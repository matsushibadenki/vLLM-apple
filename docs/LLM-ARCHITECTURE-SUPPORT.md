# LLMアーキテクチャ対応計画

更新日：2026-09-25。対象：Apple Silicon上のvLLM-Apple。関連：[roadmap](ROADMAP.md)。

## 結論

対応モデルを増やすには、モデル名ごとの実装追加に加え、**Attention／token mixer、FFN、位置表現、正規化・残差、状態cache、入出力形式を組み合わせて扱える能力管理**が必要になる。
最初にDense Transformer、local/global混在、標準MoEを広げ、次にMLAとGated DeltaNet系、Mamba系、短い畳み込み系を認定する。さらに圧縮・疎Attention、KV共有、再帰的な深さ方向の実行まで拡張する。

参照した[LLM Architecture Gallery](https://www.sebastianraschka.com/llm-architecture-gallery/)は閲覧時点で105モデルを掲載し、主にtext LLMとmultimodalモデルの言語backboneを対象としている。105モデルが105種類の独立した実行エンジンを必要とするわけではない。
本書はその一覧を調査の入口とし、技術要件は下記の公式実装・設定・ドキュメントで確認した。優先順位と実装方針は本projectへの提案であり、galleryの順位や対応率ではない。

## 対応すべきアーキテクチャ群

優先度Aは最初に認定範囲を広げる対象、BはAの基盤を使う次段階、Cは専用状態・実行契約を追加する段階。これは実装済みの表示ではない。
同一モデルが複数行に属する。例示した系列もvariant／revisionにより構造が変わるため、名前だけで機能を継承させない。

| 優先度 | アーキテクチャ／能力 | 代表モデル・系列 | 必要な実行・状態管理と一次資料 |
| --- | --- | --- | --- |
| A | Dense causal Transformer：MHA／MQA／GQA | Llama、Qwen2／Qwen3 Dense、従来のGPT系 | queryとKV head数の分離、causal mask、異なるhead dimension、biasの有無、FFN／norm差分。MQAはKV headが1の構成。[Llama](https://huggingface.co/docs/transformers/model_doc/llama)、[Qwen3](https://huggingface.co/docs/transformers/model_doc/qwen3) |
| A | Sliding-windowとglobal attentionの混在 | Gemma 2／3、Mistral系 | layerごとのwindow／global指定、window境界、absolute position、ring／rotating cache。global層のKVをwindowと共に捨てない。[Gemma 3](https://huggingface.co/docs/transformers/model_doc/gemma3) |
| A | Sparse MoE＋通常Attention | Mixtral、Qwen3 MoE | top-k routing、expert重み付け・集約、expertごとの量子化、dense／MoE混在。shared expert、group routingなどは追加能力として区別。[Mixtral](https://huggingface.co/docs/transformers/model_doc/mixtral) |
| A | MoE＋local/global＋attention sink | gpt-oss | sinkを含むsoftmax、独自FFN差分、bias、window混在を保持。sinkは先頭tokenを残すcache policyと同義ではない。MXFP4の読み込みは別の形式対応。[実装ドキュメント](https://huggingface.co/docs/transformers/model_doc/gpt_oss) |
| B | Multi-head Latent Attention（MLA）＋MoE | DeepSeek-V2／V3／R1本体 | latent cacheとRoPE成分、projection／吸収計算、shared／routed experts。展開KV実装と圧縮KV実装のメモリを区別。[DeepSeek-V3](https://huggingface.co/docs/transformers/model_doc/deepseek_v3) |
| B | Gated DeltaNet＋gated attention hybrid | Qwen3-Next、Qwen3.5 | delta-rule状態、短いconv状態、full-attention KVをlayer別に保持。chunk prefillとrecurrent decodeの一致、MoE／Dense差分。[Qwen3-Next](https://huggingface.co/docs/transformers/model_doc/qwen3_next)、[Qwen3.5](https://huggingface.co/docs/transformers/model_doc/qwen3_5) |
| B | Kimi Delta Attention（KDA）hybrid | Kimi Linear | KDA専用gate・更新式とMLAの混在。Gated DeltaNetと同じ「linear」という理由で同一kernelへ送らない。[Kimi Linear](https://huggingface.co/docs/transformers/model_doc/kimi_linear) |
| B | Selective SSM、SSM＋Attention／MoE hybrid | Mamba／Mamba-2、Jamba | selective scan／SSD、conv／SSM state、mixed layer順序。Mamba世代ごとのstate shape・演算差分を保持。[Mamba-2](https://huggingface.co/docs/transformers/model_doc/mamba2)、[Jamba](https://huggingface.co/docs/transformers/model_doc/jamba) |
| B | Gated short-convolution＋Attention | LFM2系 | causal convの履歴、gate、GQA、QK normalization。SSMではなく専用conv stateとして扱う。[LFM2](https://huggingface.co/docs/transformers/model_doc/lfm2) |
| B | Cross-layer KV sharing、K=V共有、per-layer embedding | Gemma 4の該当variant | layer間のcache所有権、producer／consumer順序、共有allocationの単一計上、追加embedding／projection。全variantで同じ共有方式とは仮定しない。[Gemma 4](https://huggingface.co/docs/transformers/model_doc/gemma4) |
| C | Indexerによる疎Attention | DeepSeek-V3.2系 | indexer state、top-k token選択、gather、causal制約、index側memory。疎AttentionとMoEの疎性は別能力。[公式実装](https://github.com/deepseek-ai/DeepSeek-V3.2-Exp) |
| C | 圧縮Attention、CSA／HCA、特殊残差接続 | DeepSeek-V4 | sliding state、compressor pool／未完了window、indexer、K=V共有、mHCを個別実装。V3のMLA adapterだけで対応扱いにしない。[DeepSeek-V4](https://huggingface.co/docs/transformers/model_doc/deepseek_v4) |
| C | 複合的な新hybrid | Qwen4-Exp／Qwen3.8-Flash-Nextの対象artifact | gated recurrence、疎Attention、MoE、残差gate、追加embeddingなどを正確なconfigから合成。ローカル専用adapterの存在と一般認定を分離。[Qwen4-Exp](https://huggingface.co/docs/transformers/model_doc/qwen4_exp) |
| C | RNN／matrix-memory系 | RWKV、xLSTM、RecurrentGemma | RWKVの世代別状態更新、xLSTMのmatrix memory、RecurrentGemmaのrecurrent／local attention。互いに別adapter・別数値検証が必要。[RWKV](https://github.com/RWKV/RWKV-LM)、[xLSTM](https://github.com/NX-AI/xlstm)、[RecurrentGemma](https://huggingface.co/docs/transformers/model_doc/recurrent_gemma) |
| C | Looped／depth-recurrent Transformer | Ouro-Thinking | 同じblockの反復、exit制御、iteration別cache identity。時間方向のRNN状態と深さ方向の反復を混同しない。[公式config](https://huggingface.co/ByteDance/Ouro-2.6B-Thinking/blob/main/config.json) |

**MoEのactive parameter数は、必要なweightメモリ容量ではない。** 常駐型では原則として全expertのweightを予算に入れる。offload型は転送bufferとI/O時間を追加し、常駐型の高速性を前提にしない。
また、R1の蒸留モデルは実際のbackboneを確認する。モデル名にR1を含むことだけでMLA／DeepSeek-MoEへ分類しない。

## 系列をまたいで必要な共通部品

以下は実装時のチェックリスト。必須のforward semanticsと任意の高速化を分けて登録する。

| 領域 | 必要な差分 | 見落とした場合の問題 |
| --- | --- | --- |
| 位置表現 | RoPEの回転layout・partial dimension・theta・scaling、YaRN等、layer別RoPE／NoPE、absolute embedding、ALiBi／relative biasを必要な系列ごとに追加 | loadに成功しても長文品質やcached decodeが壊れる |
| Attention | MHA／MQA／GQA、MLA、window、sink、QK norm、logit scale／softcap、gated output、K=V共有 | head数だけの互換判定ではforwardが一致しない |
| FFN／MoE | GELU／SiLU／SwiGLU／GeGLU、clamp、router score関数・正規化・top-k・shared expert・group／hash routing・latent projection | expert選択や出力scaleがずれる |
| Norm／residual | LayerNorm／RMSNorm、epsilon、zero-centered weight、pre／post／sandwich配置、residual scale／gate、特殊接続 | tensor shapeが一致しても数値が異なる |
| Embedding／head | tied／untied weight、embedding scale、per-layer／n-gram embedding、vocabulary padding、出力logit処理 | weight欠落、余分なメモリ、token誤生成 |
| Weight形式 | safetensors shard、MLX量子化、GGUF、MXFP4／FP8等をbackend能力として区別。packing・scale・group size・transposeを検証 | architecture対応とartifact対応を混同する |
| Tokenizer／会話形式 | BPE／SentencePiece等、BOS／EOS／stop、chat template、thinking channel、tool-call形式、incremental detokenization | forwardが正しくてもchat／tool-useが利用できない |
| MTP／speculation | 通常next-token生成と、draft head・accept／reject・rollback・sampling分布を分離 | 任意の加速機能の未対応で通常生成まで拒否する、または不正なtokenを公開する |

実装上の根拠例は[Qwen3のQK norm](https://huggingface.co/docs/transformers/model_doc/qwen3)、[Gemma 4のembedding／KV共有](https://huggingface.co/docs/transformers/model_doc/gemma4)、[gpt-ossのsink](https://huggingface.co/docs/transformers/model_doc/gpt_oss)、[Qwen3-Nextのhybrid／MTP](https://huggingface.co/docs/transformers/model_doc/qwen3_next)。上表はそれらを基にした互換性確認項目で、全モデルが全機能を必要とするという意味ではない。

## 現行コードとの対応と不足

ここでの[Done]は限定された実装の存在を示す。architecture全体の推論認定ではない。

| 状態 | 現在の実装 | 次に必要な作業 |
| --- | --- | --- |
| [Done] | [model.py](../vllm_apple/model.py)に通常KV、MLA、SSM／hybridのmetadata解析・メモリ見積もりがある | 実backendのcache layoutと照合。metadata式をkernel対応の証明にしない |
| [Done] | [StateMemorySpec](../vllm_apple/types.py)にrecurrent／window／sparse等の容量表現がある | layer別layoutとallocation共有を表現し、model inspectorからの自動導出を検証 |
| [Done] | `inspect_model_architecture`にQwen4-Exp専用required feature検出がある | それ以外のモデルには一般化したfeature導出が必要 |
| [Done] | [MLX state adapter](../vllm_apple/mlx_semantic_state.py)にbounded snapshotとopaque handleの契約がある | 各state方式でcapture／restore／release・分岐・rollbackの実効性を認定 |
| [Next] | 一般architecture registry | 現在、専用分岐以外は`required_features=()`へ進む。空集合を「任意backendで対応」の証拠にせず、unknown／unverifiedを明示する |
| [Next] | backend能力の実測・登録 | backend名だけで不足能力を補ったことにしない。固定buildの実装とprobeから対応集合を作る |
| [Next] | recurrent方式別のstate descriptor | 既存の`state_size`／`conv_kernel`前提の式を、DeltaNet／KDA／RWKV／xLSTMへそのまま流用しない |

上記はコードの静的確認結果であり、この文書作成時に新しいモデル推論や全テストを実行したわけではない。

## 提案する能力・状態の契約

モデル名の巨大な条件分岐を増やす前に、次の情報を正規化したdescriptorとして持つ。

- `model_identity`：model type、config／weight／tokenizer／template hash、revision、adapter、precision。
- `layers[]`：mixer種別、FFN種別、position、norm／residual、window、head shape、必要なoperator。
- `state_layout[]`：append-only KV、window KV、MLA latent、delta matrix、SSM、conv history、compressor／indexer、共有groupとowner。
- `execution_modes`：text／vision／audio、prefill／decode、batch、prefix reuse、speculation。必須と任意を分離。
- `backend_capabilities`：version／build、対応operator・dtype・shape、model load可否、実機証跡。

メモリ計算は次を基礎とする。これは実装提案であり、既存APIの宣言ではない。

```text
総予算 = 常駐weights
       + 一意な共有groupごとのKV／latent allocation
       + requestごとのrecurrent／conv／compressor／indexer state
       + prefill・decode・MoE dispatchのpeak workspace
       + allocator余白・OS reserve
```

local層は対応backendが実際にwindow上限で保持すると確認した場合にのみ`min(tokens, window)`を使う。MLAも実際にlatentで保持する場合にのみ圧縮容量を使う。
recurrent stateは過去token分のKVを持たなくても0 byteではなく、並列request数・分岐・snapshot数に応じて増える。memory見積もりと実測の差を認定reportへ記録する。

recurrent stateの任意位置への巻き戻しは、KVの末尾切り捨てと同等ではない。checkpointから再計算するか、backendが保証するrestore／rollbackを使う。未実装の組合せではprefix reuse／speculationを無効にして通常生成を維持する。

## 対応と認定の段階

全モデルに一個のsupported booleanを付けず、次を独立して公開する。

1. **Recognized**：configと必須演算・state layoutを理解できる。
2. **Loadable**：対象backendがartifactを読み、weight対応とメモリ適合を確認できる。
3. **Correct**：prefill／decodeが基準実装と許容誤差内で一致し、実モデルの生成品質を満たす。
4. **Service-qualified**：streaming、cancel、timeout、batch、cache isolation、三言語を対象modeごとに検証した。
5. **Performance-qualified**：対象Macでroadmapの性能・長時間安定性gateを満たす。

「architecture未対応」「形式未対応」「メモリ不足」「未検証version」「加速機能のみ未対応」は別の理由として返す。
core operatorが無い場合は明示的に拒否し、任意機能だけが無い場合は認定済みの通常生成へ戻す。

## 実装・認定の順序

- [Next] **A0：能力matrixとfixture**。主要model typeの固定configを収集し、上記descriptor、backend build、mode別状態を登録する。unknownの誤認定を防ぎ、既存認定経路への互換性を確認する。
- [Next] **A1：Dense＋window＋MoE**。小型Llama／Qwen、Gemma、メモリに収まるMoEを代表にする。gpt-ossはsinkと形式対応を独立確認する。同じ系列の全サイズを一括認定しない。
- [Later] **B1：MLAと共有KV**。latent・展開cache、KV共有・K=V、layer別位置表現、追加embeddingを確認する。大容量weightが未配置ならoperator fixtureまでと明記する。
- [Later] **B2：hybrid state**。Gated DeltaNet、KDA、Mamba、短いconvを別adapterで追加し、mixed batchとcheckpoint再開を検証する。小さい実モデルで検証可能な系列はB1と順序を入れ替えてよい。
- [Later] **C1：高度な疎・圧縮・再帰実行**。DSA、CSA／HCA、特殊residual、RWKV／xLSTM、LoopLM、複合hybridを拡張する。対応Mac・artifact・基準実装が揃ったものから進める。
- [Later] **追加範囲**。Encoder-decoderのcross-attention、multimodalのvision／audio encoder・projector・位置表現・maskを別契約で扱う。text backbone合格を画像・音声対応の根拠にしない。

MLX-LM等に同じモデル実装がある場合は、その固定versionのadapterを先に利用する。独自Metal kernelは実測で不足が判明した演算に絞る。[MLX-LM公式モデル実装](https://github.com/ml-explore/mlx-lm/tree/main/mlx_lm/models)。
[Flash Linear Attention](https://github.com/fla-org/flash-linear-attention)は更新式・参照実装の調査に使えるが、そのkernelをそのままMac／Metalで利用できると仮定しない。

## 最低限の認定テスト

| 領域 | 検証内容 |
| --- | --- |
| 数値 | reference logits／hidden state、full prefillとchunk prefill、逐次decodeの一致。量子化と非量子化は別許容差 |
| State | window前後、chunk境界、prefix編集、分岐、restore、cancel中断、異なる長さのbatch、別session隔離 |
| MoE | routing score／top-k／shared expert／集約、expert未使用・偏り、必要weight全体のmemory |
| 疎・圧縮 | index結果、未完成圧縮block、causal mask、position offset、長文でのretrieval品質 |
| Serving | tokenizer／template／EOS、SSE完走、tool-call、英語・日本語・简体中文、同時要求と失敗回復 |
| 実運用 | model＋backend＋Mac別のTTFT／TPOT／goodput／実state bytes、8時間・24時間の段階的soak |

tiny／synthetic fixtureはoperatorの正しさに使い、実モデル認定の代用にしない。対応率を出す場合は、固定catalog中のvariant・revision・modeを分母として、未配置・メモリ不足・未実装も報告する。

## English summary

Support should be composed from capabilities rather than inferred from a model name. Prioritize dense MHA/MQA/GQA, local/global attention and standard MoE; then MLA, Gated DeltaNet/KDA hybrids, Mamba/SSM, short convolutions and cross-layer KV sharing. Extend to indexed/compressed attention, specialized residual paths, RWKV/xLSTM and depth-recurrent models with separate state contracts.

- [Done] The code contains metadata/memory foundations and bounded state adapters; these are not universal inference certification.
- [Next] Build an architecture registry with explicit unknown/unverified states and qualify representative dense, windowed and MoE models.
- [Later] Add each hybrid/specialized family with reference comparisons, state lifecycle tests and real-model serving evidence.

Track recognition, loading, correctness, serving and performance independently. Quantization format, tokenizer/template and optional MTP support are separate axes. Total MoE weights and non-KV recurrent state must be budgeted. A text-backbone pass does not certify multimodal inputs.

## 简体中文摘要

应按能力组合支持模型，而不是仅根据模型名称判断兼容性。优先支持Dense MHA／MQA／GQA、局部与全局Attention混合及标准MoE；随后支持MLA、Gated DeltaNet／KDA、Mamba／SSM、短卷积和跨层KV共享；最后扩展索引／压缩Attention、特殊残差、RWKV／xLSTM及深度循环模型。

- [Done] 现有代码具备部分元数据、内存估算及有界状态适配基础，不等于全面推理认证。
- [Next] 建立显式区分未知与未验证的架构注册表，认证代表性的Dense、窗口Attention及MoE模型。
- [Later] 为各混合和特殊架构分别增加参考数值、状态生命周期和真实模型服务测试。

识别、加载、正确性、服务能力和性能应分别记录。量化格式、分词器／模板及可选MTP是独立维度。MoE总权重和非KV循环状态也占用内存。文本骨干通过测试不代表支持多模态输入。
