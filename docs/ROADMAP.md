# vLLM-Apple Runtime Roadmap

更新日：2026-09-26

## 目標と優先順位

Apple Silicon Macで、品質を維持しながら最短の応答時間と高い持続スループットを実現し、長時間の推論でもメモリ不足・停止・状態混入を起こさないLLM実行環境を目指す。
「最も高速」は全モデル・全Macに対する無条件の宣言ではなく、公開した比較条件で、品質と安定性の基準を満たす実行経路の中から最速を選べることと定義する。
Intel Macは互換性の別枠とし、Apple Siliconの性能認定を適用しない。

開発順序は **比較基盤 → 標準LLM経路 → batching／KV再利用 → 実測hot path → 長時間認定と配布** とする。
安定性の回帰検証は全段階で行う。画像・音声・動画生成、独自形式、大規模分散の追加より、日常的なchat・coding・agent用途の改善を優先する。

2026-09-26の外部レビューを踏まえ、開発単位を「固定条件で比較 → 支配時間を特定 → 1か所改善 → E2E再測定」とする。P0の基準が得られた対象では、P2と並行してP3のhot path調査を開始できる。kernel追加や独自engineの構築自体を成果指標にせず、長context・複数request・Agent・限られたメモリでのgoodputと安定性を重視する。

既存の253項目を含む詳細履歴は、編集開始時の内容をそのまま[旧roadmap](ROADMAP-history-2026-09-25.md)へ保存した。未コミットの追記も保存対象に含む。
本書が今後の優先順位を定め、旧roadmapは証跡索引として参照する。旧書の完了表示は、一般用途の性能認定を意味しない。
[設計判断](Architecture-Decision-Apple-Execution.md)のcontrol／execution分離と計測優先方針を継続する。

## ステータスと完了条件

- [Done] implemented in the current codebase — 現在のコードに実装がある。実機認定の範囲は別記する。
- [Next] high-priority unfinished work — 次の開発サイクルで取り組む未完了作業。
- [Later] planned, but not the closest next step — 依存作業の完了後に進める計画。
- [pending] 現在の筐体・環境では検証できない作業。必要なhardware／artifact／資格情報と再開条件を添え、環境が整った時に着手候補へ戻す。

旧書の`[Pending]`は履歴として保持する。本書では、優先順位待ちの`[Later]`と環境待ちの`[pending]`を区別する。現在のM4で実行可能な長時間試験や未実装項目は、時間がかかることだけを理由に`[pending]`へ移さない。
新規項目の完了にはコード／テスト、対象の実行経路、再現コマンド、認定範囲を必要とする。性能項目は実モデルの比較reportも必要とする。
以下の数値は**今後の受け入れ目標**であり、達成済みの測定値ではない。

## 現在地：再利用できる基盤と不足している証拠

| 状態 | 基盤・証拠 | 認定の限界と次の仕事 |
| --- | --- | --- |
| [Done] | memory admission、thermal／pressure対応、予約付きscheduler、process隔離、profile／rollback基盤 | 制御機構の存在だけでは実LLMの速度向上を証明しない。主経路への適用をP0で監査する |
| [Done] | [MLX server wrapper](../vllm_apple/mlx_server.py)：MLX-LM serverへの委譲、tokenize、allocator／cache計測 | wrapper独自のcontinuous batching実装ではない。backend versionごとの実効機能を測定する |
| [Done] | [semantic cache](../vllm_apple/semantic_cache.py)、[state coordinator](../vllm_apple/semantic_state.py)、[MLX state adapter](../vllm_apple/mlx_semantic_state.py) | 契約・adapterと、標準HTTP経路で実KVが再利用されることを分けて検証する |
| [Done] | [Gemma 2 2B 4-bit・30分report](evaluation/homebrew-029-text-30min-2026-09-19.json)：6,775/6,775成功 | Homebrew 0.29.0、context 1024、concurrency 1限定。KV容量再評価はunavailable。42.841 decode tok/s／平均TTFT 91.105 msは別の3-sample probe値 |
| [Done] | [Qwen3-VL persistent worker・30分report](evaluation/qwen3-vl-coreml-persistent-30min-soak-2026-09-20.json) | 固定shape／固定taskの証拠。一般VQA、標準text経路、ANE単独実行の認定へ広げない |
| [Done] | [MLX cache計測改善](vllm-mlx-review.md)、[キャンセルqueue回収](llama-cpp-review.md) | 現在の作業ツリーに実装あり。実MLX／継続キャンセル負荷での性能検証が残る |

現時点では、主要backend横断の同条件ランキング、代表モデル群の長時間SLO、全Mac世代での優位性を証明する比較資料は揃っていない。まずこの不足を埋める。

## モデル対応範囲を広げる計画

必要なarchitecture、共通operator、state契約、現行実装との差分は[LLMアーキテクチャ対応計画](LLM-ARCHITECTURE-SUPPORT.md)にまとめる。

- [Done] A0初期診断：`inspect-architecture`とJSON schema、5系列の構造recipe・synthetic fixtureを追加。未知／未検証を明示し、実行認定は付与しない。詳細は上記対応計画を参照。
- [Done] A0 recommendation統合：`inspect-model`をschema v2へ更新し、宣言一致と実機認定を分離。未知backendへの暗黙の能力付与も廃止。Gemma2／M4／MLX-LM 0.32.0の三言語smokeは3/3合格（短い算術task限定、標準backendへ未昇格）。
- [Done] A0証跡gate：30分text qualificationのidentity bindingと、`inspect-model`／managed `serve`の任意検証を追加。7日期限・model/backend/runtime/hardware変更・設定上限超過を拒否する。
- [Done] Homebrewが意図的に削除するRECORDへ対応。brew管理・分離venvを確認し、環境全体のbounded inventoryでbackend identityを検証する。
- [Done] Homebrew MLX-LM 0.32.0／Gemma 2 2B／M4でidentity付き30分text試験に合格。6,491/6,491件成功、RSS peak増加15.9 MiB、三言語・stream一致・正常終了と`inspect-model`による証跡再検証を確認。[実測report](evaluation/architecture-gemma2-homebrew-bound-2026-09-26.json)。
- [Done] 有効なidentity付き証跡を指定した通常`serve`で、MLX-LMのversion matrix範囲外だけを限定許可する。証跡なし・期限切れ・identity変更・他の互換性エラーは引き続き拒否する。
- [Done] 変更後runtimeの[30分再認定](evaluation/architecture-gemma2-homebrew-serving-bound-2026-09-26.json)で5,995/5,995件成功。新しい証跡で`--skip-backend-check`なしの[通常serve実HTTP検証](evaluation/architecture-gemma2-homebrew-managed-serve-2026-09-26.json)も合格（三言語・greedy反復・stream一致・正常終了）。
- [Next] 追加のDense／window／MoEモデルと、長文・並列負荷・cancel／recoveryを認定する。今回の通常serve確認は短いHTTP smokeであり、frontend全体の30分soakや性能優位の証明ではない。
- [Next] P0のcapability matrixへ、モデル名だけでなくlayer構成・必須operator・weight形式・state layout・backend buildを登録し、unknownを対応済みと扱わない。
- [Next] Dense MHA／MQA／GQA、local/global混在、標準MoEを代表モデルで認定する。
- [Later] MLA、KV共有、Gated DeltaNet／KDA、SSM、短いconvを個別state契約で広げ、その後に高度な疎・圧縮Attentionと再帰実行へ進む。

## 実行アーキテクチャ

```text
Swift SDK / CLI / OpenAI-compatible API
                  ↓
Control plane: admission / queue / lifecycle / telemetry / routing
                  ↓  bounded commands, request identity, cancellation
Backend-owned process: model / KV / batch scheduler / generation
                  ↓
MLX / vLLM-Metal / qualified optional backend → Metal GPU
                  ↘ qualified fixed-shape encoder → Core ML
```

- GPUをtext prefill／decodeの基準にする。CPUはtokenization・I/O・制御を担い、CPU／ANEへの演算移動は転送・同期・競合込みで改善した場合に限定する。
- modelとKVの所有者はbackend processに一本化する。control planeのpriority queueとbackendのtoken schedulerの責務を明記し、二重queueによる待ち時間を測る。
- 大きなtensorをHTTP／JSON／Swift境界で往復させない。Unified Memoryでもcopy・materialize・同期・page faultのコストは計測する。
- MLX-LM directとvLLM-Metalを優先比較し、llama.cpp Metal／GGUFとvllm-mlxを比較対象にする。比較への追加と製品backendとしての採用は別判断とする。
- 自動選択はmodel revision・precision・context・concurrency・SoC・メモリ・OS・backend buildに束縛する。実行中にbackendやKV形式を切り替えず、未認定条件は既知の安全経路へ戻す。

## P0 — 比較可能な基準と実行経路の監査 [Next]

依存：なし。次の開発サイクルの最優先。成果物はbenchmark suite、capability matrix、再現可能なbaseline report。

- [Next] 起動から最終tokenまでの経路を追い、backendごとにstreaming、usage、cancel、prefix reuse、continuous batching、chunked prefill、KV精度、structured outputの「未対応／adapterのみ／実機合格」を記録する。フラグの存在だけでは有効と判定しない。
- [Next] 既存phase probe・qualification・soakを共通reportへ接続する。直接backendとdaemon経由を同じ入力で比較し、queue、tokenize、prefill、decode、serialization／SSE、model loadを分離する。
- [Done] P0のstream通信計測：phase probeで最終生成contentと`[DONE]`到着を分離し、end-to-end／stream tailを固定容量histogramへ集計。未取得sampleは欠測として区別する。既存のdecode計算を維持し、HTTP・schema・qualification回帰49件で検証。[仕様・再現手順](PHASE-TRANSPORT-METRICS.md)。実モデルの性能比較は未実施。
- [Done] P0のbounded HTTP benchmark runner：三言語・固定workload hash、並列度1〜32、失敗を含む件数、品質とTTFT／E2E SLOを満たすgoodput、言語別集計、参考値表示付きp99を実装。関連54テスト合格。[仕様](TEXT-BENCHMARK.md)。[M4実測](evaluation/text-benchmark-m4-smoke-2026-09-26.json)のGemma 2 2B／MLX-LMで並列度1は30/30正答。並列度2は応答停止後に手動回収し、不完全な失敗runとして保存。性能優位・batching認定は付与しない。
- [Next] モデル・データ・tokenizer・chat template・量子化方式／group size・KV dtype・sampling・依存versionを固定する。reportへrevision／hash、再現コマンド、power／thermal、失敗・除外理由を保存する。
- [Next] 実MLX cacheでmetadata計測の完全性と負荷を確認し、viewの共有storageを二重計上し得る限界を明記する。queueキャンセルの長時間負荷とp95待ち時間を測る。

### 標準比較matrix

| 軸 | 最初に比較する条件 |
| --- | --- |
| Hardware | 手元のMacを正確に記録。続いて16 GB級、24–36 GB級、64 GB以上、異なるSoC世代を追加。未保有機は未評価表示 |
| Models | 配置済み2–4Bを基準に、収容可能な7–8B、14B級、MoE／hybridを順次追加。architectureごとに対応確認 |
| Workload | chat、coding、tool-use、長文読解。英語・日本語・简体中文ごとに品質を分離 |
| Context | 入力128／1K／4K／16K tokens、出力128／512 tokens。32K以上はmodel上限・admission通過時のみ |
| Load | concurrency 1／2／4／8、短長混在、到着率指定、飽和試験。容量を超える組合せは理由付きskip |
| Cache | cold model load、warm model＋prefix miss、exact prefix hit、prefix編集、cache容量超過を別集計 |
| Environment | AC／battery／low-powerを別条件とし、warm-up後と熱平衡後を比較。各backendは順番を交替して実行 |

同じbit数でもGGUFとMLXの量子化は等価とみなさない。同一artifactでの純粋な速度比較と、同一品質基準を満たす異形式の比較を分ける。
比較可能な共通モデル集合とbackend固有対応範囲を両方公開し、成功した組合せだけに母集団を狭めない。

計測はclient側TTFT、token interval由来TPOT、request p50／p95／p99、E2E latency、生成tok/s、**SLO内の成功output tokens/s（goodput）**を主指標にする。
SSE chunk数をtoken数とみなさず、usageやbackend token timestampを用いる。欠測は推定値で埋めずunavailableとする。本文最終tokenと`[DONE]`到着も分けて記録する。
RSS・allocator・KV・OS pressure・swap差分は別系列で記録し、重複するmemory値を単純加算しない。energy/tokenは取得可能な場合だけ補助指標とする。

**完了条件：** 手元のMacで少なくとも2 backend・同一品質基準のモデルについて、独立した3 runを再現できる。各主要ケースは合計100以上の完了requestを目安とし、sample数とばらつきを公開する。p99は1,000未満なら参考値扱い。比較不能・未対応条件も残す。

## P1 — 標準text経路と持続的な安定性 [Next]

依存：P0で選んだ基準経路。成果物は認定text profileと、運用上の失敗から復帰できる標準server。

- [Next] 起動時に依存ABI、実GPU選択、model／tokenizer identity、実KV capacityを確認する。取得不能は安全な上限と理由を提示し、未検証versionを自動昇格させない。
- [Next] model-owner thread／processを固定し、load／generate／cancel／closeの所有権を一貫させる。既存main-thread process経路を再利用し、モデルをrequestごとにロードしない。
- [Next] admissionにprompt＋最大出力のKV増分、prefill scratch、batch増分、allocator cache、OS reserveを反映する。最大contextと最大concurrencyを同時に保証しない。
- [Next] client切断、active／queued cancel、timeout、遅いSSE consumer、worker crash、sleep／wake、shutdownを実backendで試験する。queue・IPC・出力bufferをboundedに保つ。
- [Done] M4／Gemma 2 2B／MLX-LM 0.32.0のbatch mask不一致へ、version＋source hash限定のプロセス内互換修正を追加。[修正後HTTP試験](evaluation/text-benchmark-m4-gemma2-mask-fix-2026-09-26.json)は並列度1／2とも30/30正答。GPU上の8条件で未修正の逐次attentionと数値一致。Homebrew packageを変更せず、[専用起動経路](GEMMA2-BATCH-MASK-FIX.md)で明示適用する。
- [Next] 修正したGemma 2経路の長文、prefix編集、cancel／回復、30分以上の並列負荷を検証する。短い算術試験を一般品質や長時間安定性の認定に代用せず、標準serveへの採用は再認定後とする。
- [Next] cancelはbackendの安全な実行境界で処理する。解放完了まで予約を維持し、stream公開済みrequestの黙った再実行やtoken重複を禁止する。OOM retryは副作用と状態復元が証明できる経路だけに限定する。
- [Next] watchdogとrestart backoffを整備する。ハング時はworkerを回収し、失敗をclientへ通知して新規requestを回復する。過負荷拒否と内部障害を別集計する。

**完了条件：** 30分smokeを入口に、認定profileで8時間の混合負荷と障害注入を完走する。正常負荷では予期しないcrash／OOM／hang／状態混入0件、成功率99.9%以上を目標とする。意図したcancel／admission拒否は別に件数を出し、分母操作で成功率を上げない。
warm-up・cache充填後の同一負荷窓とidle回復時を比較し、RSS／allocator／queue／handleに継続的な増加がないことを確認する。注入後の予約・worker・一時file回収と後続正常応答も必須。

## P2 — Continuous batchingと実KV再利用 [Next]

依存：P0の計測、P1の所有権・cancel契約。成果物はinteractive／throughput profile。既存backend機能を先に利用し、不足だけを実装する。

- [Next] backendのtoken単位continuous batchingを実HTTP経路に接続・確認する。単なる複数HTTP requestやcontrol plane queueをbatching達成と数えない。
- [Next] chunked prefillで長い入力によるdecode停止を抑える。prefill token budget、active decode数、最大待ち時間を調整し、priority agingでbackground starvationを防ぐ。
- [Next] 実際のKV／recurrent stateに結び付いたprefix reuseを有効化する。model／adapter／tokenizer／template／position／cache saltをidentityへ含め、exact token prefix一致のみを再利用する。
- [Next] turn／tool境界anchor、copy-on-write、eviction／releaseをbackend所有下で検証する。共有prefixの書き換え、cancel、異なるsession間の状態混入を防ぐ。SWA／hybridはarchitecture固有の復元契約を要求する。
- [Next] hit率に加えて再計算を省けたprompt tokens、TTFT、cache byte、eviction頻度を測る。metadata上のhitだけでは昇格しない。
- [Next] chat／coding／agent／batchを比較workloadとして分ける。Agentは固定tool定義・system prompt、短いdecode、模擬tool待機、再入場を含め、再prefill token数、再入場TTFT、他requestのp95とstarvationを検証する。最初はbackend既存schedulerの設定profileで比較し、独自token schedulerの追加は不足の実測後に判断する。
- [Later] prefix trie、SSD cache、prompt類似度slot選択。RAM内reuseを認定した後に、SSD read／write、昇格前予約、摩耗、privacyと実latencyを含めて評価する。

**完了条件：** prefixなしconcurrency 1のp95 TTFT／TPOT悪化を5%以内に抑え、代表concurrency 4でgoodput 20%以上改善を目標とする。prefix hitではprefill実行token数の減少とTTFT改善を確認する。未達なら標準有効化せず、改善するprofileだけを残す。

## P3 — 計測で選ぶkernel・量子化・speculative実行 [Later]

依存：調査開始には対象経路のP0 baseline、標準採用にはP1の安定性と対象P2 workloadの回帰report。成果物は対象shapeに限定した高速化profileと安全なfallback。P2全体の完了前でも、baselineで支配時間が分かった経路の調査は並行できる。

- [Next] P0 baseline取得後、GPU profilerで同期、CPU送信、dequantize、GEMV／GEMM、attention、KV copyの支配時間を特定する。decodeのmemory帯域律速とprefillのcompute律速を分けて検証し、律速を先に決め付けない。
- [Later] upstream MLX／Metalの最適化を先に比較し、不足するhot pathに限りfused quantized GEMV／GEMM、RoPE／RMSNorm、paged／split-KV attentionを追加する。kernel単体の改善とE2E改善を別reportにする。
- [Later] model／shape／SoC別にbounded autotuningを行う。compile・warm-up時間、p95、scratch使用量まで比較し、未測定shapeは既定kernelへ戻す。
- [Later] weight 4／8-bit、KV量子化、mixed precisionを品質／速度／容量のPareto比較で選ぶ。perplexityだけでなく三言語coding、tool-use、long-context retrievalの劣化を検出する。
- [Later] speculative decodeは小さい互換draft、GPU draft、利用可能な自己draft手法を候補として比較する。acceptance、verify cost、追加KV、全体latencyを測る。greedyの一致とsamplingの分布保存は別々に検証する。
- [Later] ANEは固定shape encoderを優先する。draft導入には互換artifactとE2E改善の証拠を必要とし、CPU＋ANE＋GPUの同時利用自体を目標にしない。

既存Gemma CPU draft＋GPU verifierは[実測で約11.59倍低速](evaluation/gemma3-1b-cpu-4b-gpu-speculative-rss-2026-09-22.json)だったため既定有効化しない。より小さい互換draftやCore ML artifactが揃うまで、この構成の再試験を最優先作業にしない。

**完了条件：** 独立3 runでE2E中央値5%以上改善し、改善幅が測定ノイズを超える。対象profileのp95悪化5%以内、品質gate合格、memory budget内を満たす。未達の最適化は採用しない。
品質gateは評価前に固定し、deterministic correctness、task score、長文検索、構造化出力を分ける。量子化の許容差はsuiteごとに明記し、失敗sliceを総合平均で隠さない。

### Execution Plannerの拡張境界

- [Next] 既存の[Execution Planner設計](Architecture-Decision-Apple-Execution.md)を基に、同一backend内のphase／operator設定とworkload別profileを実測へ結び付ける。model／state所有者はbackend processに維持し、upstreamのbatching、KV管理、kernelを先に比較する。
- [Later] backendを跨ぐprefill／decode分割は、weight共有、KV／recurrent state形式、位置情報、同期、cancel／解放の契約を定義し、転送・変換・重複memory込みのE2E改善を証明できる場合だけ試す。DLPackなどの共有手段の存在だけでstate互換と判断しない。
- [Later] Apple-native execution engineは既存backendで解決できない律速が反復して確認された場合の選択肢とする。NAX／ANEを含む候補は実device能力と認定済みkernelに基づき選び、SoC名だけで有効化しない。未保有hardwareは未評価とする。

English: Keep P0 first. Once a baseline exists, profile hot paths alongside P2;
promote optimizations only after stability and workload regression checks. Add
agent re-entry workloads and prefer existing backend policies. Cross-backend phase
splitting and a native engine remain conditional research, not committed replacements.

简体中文：P0仍为最高优先级。取得基线后，可与P2并行分析热点，但优化必须通过稳定性
及工作负载回归测试。增加Agent工具等待与再次进入推理的测试，优先使用现有后端策略。
跨后端阶段拆分和原生引擎仍是有条件的研究方向，不是已确定的替代方案。

## P4 — Mac別自動選択と配布認定 [Later]

依存：P1の安定性、P2／P3で合格したprofile。

- [Later] interactiveはp95 TTFT／TPOT、throughputはgoodput、省電力は取得可能なenergy/tokenを目的にする。未知hardwareではbounded calibrationを行い、発熱やmemory pressureを性能目的より優先する。
- [Later] hardware／OS／model／backend fingerprint別の既知正常profileを配布し、変更時は再認定する。独立3回のqualificationは同じcandidateでも別process・別runで実施し、次release待ちを条件にしない。
- [Later] 各サポート構成で24時間soakを行い、OS更新、backend更新、起動失敗、profile破損、rollbackをrelease gateへ組み込む。未保有Macには専用runnerが必要。
- [Later] 英語・日本語・简体中文で起動、モデル互換性、メモリ不足、縮退理由、cancel／recoveryを同等に説明する。SDK／CLI／UIで状態とerror codeを統一する。
- [Later] 再現可能な依存lock、SBOM／license、署名・notarization、clean machine install、アンインストールを検証する。署名workflowの存在と実署名artifactの検証を分ける。実行にはDeveloper ID／notary資格情報が必要。

**完了条件：** 対応matrixの各認定セルに品質・性能・24時間安定性reportとrollback手順がある。機種横断平均だけで「最速」と表示せず、対象条件と比較日を公開する。

## RAGとLoRAの段階的な完全対応

推論経路の安定化と並行して、**外部RAG接続 → 固定LoRA推論 → 内蔵検索 → LoRA学習 → 複数LoRA運用 → 統合認定**の順に進める。「完全対応」は公開したmodel／architecture／backend／量子化の対応matrix内で、取り込みから回答、学習から配信までのライフサイクルを完結できることとする。未知の全モデルへの無条件対応を意味しない。

### R0 — 外部検索との接続

- [Done] `rag` CLIとPython APIで、検索済みチャンクから三言語の生成リクエストを構成する。資料のUTF-8 byte上限、丸ごとの除外、資料なし時のローカル回答不能、ループバックHTTP生成、出典ID／内容hashと参照IDの照合を実装。[利用手順](RAG-LORA.md)と`tests/test_rag.py`を参照。
- [Done] 引用IDの存在確認と回答の事実性を分離する。`references_valid`でも`grounding_verified=false`とし、未知参照・引用なし・生成未完了を区別する。模擬HTTP試験は実モデルの品質認定に含めない。
- [Next] 実tokenizerとchat templateでsystem／質問／資料／出力予約を含むcontext予算を検証する。モデル上限を超える資料選別と多言語境界条件をテストする。
- [Next] 英語・日本語・简体中文の実モデル評価を追加。回答正確性、引用箇所との意味的整合性、資料不足時の回答不能、悪意ある資料への耐性を別指標で記録する。構成を固定したbaselineと比較し、runtime変更後のevidenceを再取得する。

### L0 — 一つの固定LoRAを安全に配信

- [Next] 起動時に一つのimmutable adapterを読み込む経路を実装。base modelとadapterのhash、形式、rank、target modules、dtype、量子化との互換性を事前検証し、未対応の組み合わせは明示的に拒否する。
- [Next] adapter分のメモリをadmissionへ計上し、model一覧／capability／qualificationにadapter identityを含める。adapterなし・ありの数値差と三言語品質を実MLXで検証する。単にload成功しただけでは認定しない。
- [Next] 外部学習済みadapterと、別artifactとしてmergeしたモデルの導入手順を用意する。merge結果は独立modelとして再認定し、量子化前後の品質差とrollbackを確認する。

**R0／L0完了条件：** 対応matrixの少なくとも一構成で、出典付き回答と固定LoRA推論が実モデル試験に合格し、context超過・互換性不一致・メモリ不足を再現可能に処理できる。

### R1 — 取り込み・検索・回答を一体化

- [Later] 文書取り込み、parser、chunking、安定したdocument／chunk IDと版管理、差分更新・削除を実装する。まずtext／Markdownから始め、PDF等は形式別の検証後に追加する。
- [Later] embedding modelを生成modelと別capabilityとして管理し、`/v1/embeddings`、batching、次元・正規化・revision検証、メモリ競合の制御を実装する。embedding更新時には再indexを必須にする。
- [Later] 永続vector index、keywordとのhybrid検索、reranker、重複排除、引用spanを実装する。検索recallと回答品質を別々に測定し、障害時の復旧・index migration・削除反映を検証する。
- [Later] 検索前のtenant／ACL filterとrerank前の再検証、権限変更・資料削除時のcache無効化、監査を実装する。corpus revision、embedding revision、権限scopeをcache identityへ含め、別利用者の資料を混入させない。

### L1 — 学習と複数アダプター運用

- [Later] 既存`RepairAdapter`の抽象契約を実MLX LoRA学習workerへ接続する。学習データ・seed・step・memory上限、checkpoint、cancel／resume、学習前後のperplexityと三言語task品質を記録する。QLoRAはbackend／architecture／量子化ごとの対応を検証する。
- [Later] 学習と推論を別worker・budgetで管理し、GPU／unified memory競合を制御する。失敗時は既存配信を維持し、成果物の検証後に切り替え可能にする。
- [Later] request単位のadapter指定、登録／削除／切り替え、GPU常駐上限とeviction、adapter別batch groupingを実装する。受付時にimmutable identityを固定し、処理中の差し替えを防止する。
- [Later] KV／prefix cache keyにbase model、adapter hash、量子化、template等の実行identityを含める。異なるadapter間のKV共有を禁止し、同時実行・cancel・再起動で回答混入がないことを検証する。dynamic適用とmerge配信の品質・速度・メモリを比較する。

### RL2 — 完全対応の受け入れ条件

- [Later] CLI／API／SDKで、文書登録・更新・削除→権限付き検索→引用付き回答、および学習→評価→登録→配信→rollbackを通して操作できるようにする。三言語で同じ状態とerror codeを説明する。
- [Later] 対応matrix各セルで、LoRAとRAGを併用した品質、検索・rerank・prefill・decode別latency、p95 TTFT／TPOT、goodput、memoryを記録する。個別機能の合格を併用認定に代用しない。
- [Later] 文書更新、adapter切り替え、同時利用、cancel、メモリ逼迫、worker障害を含む8時間試験とrelease前24時間試験を通す。権限・adapter間の情報混入ゼロ、削除反映、復旧・rollbackを必須条件とする。品質・性能閾値は認定前にsuiteへ固定する。

**English:** [Done] External RAG request preparation, bounded local generation and reference-ID diagnostics are implemented; factual grounding and tokenizer limits are not certified. [Next] Validate real-model RAG quality/context budgets and one immutable LoRA adapter. [Later] Add ingestion, embeddings, hybrid retrieval, reranking, ACLs, actual LoRA/QLoRA training, multi-adapter serving with isolated KV caches, and combined 8/24-hour qualification. Full support is scoped to an explicit compatibility matrix.

**简体中文：** [Done] 已实现外部RAG请求构建、有界本地生成和引用ID检查，尚未认证事实依据或token预算。[Next] 验证真实模型的RAG质量与上下文预算，并支持单个固定LoRA适配器。[Later] 完成文档导入、embedding、混合检索、重排、权限、LoRA／QLoRA实际训练、多适配器与KV隔离，以及组合场景的8／24小时认证。完整支持以明确的兼容矩阵为范围。

## 環境が整った時の作業候補 [pending]

現在確認した筐体はMacBook Air／Apple M4／32 GiB。以下は本筐体だけでは認定できない条件として分離する。実装可能な共通runner・schema・fallbackは先に整備し、未保有機の性能値を推定で埋めない。

| 状態 | 作業候補 | 再開に必要な環境・確認内容 |
| --- | --- | --- |
| [pending] | M5世代のNAX kernel認定とM4との比較 | 対象M5実機、対応OS／toolchain／backend。device能力検出、correctness、対象shapeのE2E・memory・fallbackを実測 |
| [pending] | 16 GB級・64 GB以上・別SoCでの容量／thermal／24時間認定 | 各容量・SoCの実機runner。同一artifact、context、並列度、電源条件で比較し、容量超過を理由付きskip |
| [pending] | M4 32 GiBのadmission予算を超えるモデル・context・batchの認定 | 必要なRAMを備えるMacと対象artifact。小さい量子化版が収まることを、元の構成の合格に代用しない |
| [pending] | 複数Macの分散実行・通信込み性能 | 複数の対象Macと検証用network。通信・同期・障害時state回収を含めて評価 |
| [pending] | ANE draft＋GPU verifierの実モデル比較 | 互換draftのCore ML artifactと変換・実行契約。M4のANEが利用可能でも互換artifactなしでは認定しない。既存encoder検証はこの保留に含めない |
| [pending] | 署名・notarization済みreleaseとclean-machine install | Developer ID／notary資格情報と独立した検証環境。未署名のlocal buildとは別認定 |

English: `[pending]` means an unavailable test environment, with explicit resumption
requirements. Work runnable on this M4, including concurrency failures and long soaks,
remains `[Next]` or `[Later]`; it is not deferred merely because it takes time.

简体中文：`[pending]`表示缺少测试环境，必须注明恢复条件。当前M4可运行的工作，
包括并发故障排查和长时间测试，仍保留为`[Next]`或`[Later]`，不因耗时而搁置。

## 別トラックとして維持する作業 [Later]

| 作業 | 着手条件・必要な資源 |
| --- | --- |
| 一般VLM／Audio LLM | text経路の認定後。open-domain画像／音声品質、model固有projection、実artifactが必要 |
| 大容量MoE／hybrid／新architecture | 対応する実weightとメモリを持つMac。metadata対応を実推論成功と混同しない |
| 画像・音楽・動画生成 | LLM経路と別worker／budgetで維持。大容量artifact・専用quality suite・必要なMacを用意 |
| 独自数値形式・構造pruning | 既存量子化との比較で、品質維持とE2E上の必要性を示した後 |
| Multi-Mac | 2台以上の実Macと物理link測定。通信時間込みで単一Macを上回る用途があること。loopback試験は物理認定にしない |
| SSD／CPU offload | メモリ適合性を目的に評価し、最速経路とは別profileで公開 |

## 最初の実装順序

1. [Next] P0のcapability matrixと比較manifestを作り、既存の実装・probe・実HTTP接続・未認定箇所を対応付ける。
2. [Next] 配置済み小型text modelで、MLX-LM directとvLLM-Metalの直接／daemon経由baselineを取る。
3. [Next] cache計測とキャンセルqueue修正を実MLX・継続負荷で検証する。
4. [Next] P1のcancel／memory／復旧を標準経路で確認し、8時間soakを通す。
5. [Next] P2のbatchingとprefix reuseを一つずつ有効化し、同じsuiteで効果と回帰を比較する。
6. [Later] P3のprofilerで支配時間を特定し、最も効果の大きい一箇所だけを最適化する。
7. [Later] P4で機種を増やし、24時間認定と配布へ進む。

## English — execution summary

The goal is the fastest **qualified** route for each Apple Silicon Mac, model and workload, while preserving quality and sustained stability. This is a roadmap, not a claim of measured leadership.

- [Done] Reuse the existing memory admission, scheduler, process isolation, profiling and rollback foundations. Existing Gemma and vision evidence covers limited configurations, not general certification. The original roadmap, including in-progress edits, is preserved in the linked history.
- [Next] P0: audit real backend capabilities and establish reproducible, quality-matched direct-versus-proxy benchmarks. Report client TTFT, TPOT, tail latency, SLO goodput, memory and failures across English, Japanese and Simplified Chinese.
- [Next] P1: qualify the standard text route, cancellation, bounded queues, memory admission and failure recovery under an eight-hour mixed workload.
- [Next] P2: validate backend-owned continuous batching, chunked prefill and actual KV reuse. Target 20% higher concurrency-four goodput with no more than 5% regression in single-request p95 TTFT/TPOT.
- [Later] P3: optimize measured hot paths, quantization and speculative decoding only after quality and end-to-end gates. Target at least 5% reproducible E2E improvement beyond measurement noise.
- [Later] P4: certify hardware-specific profiles with 24-hour soaks, dependency pinning, rollback, signing and clean-machine installation. Additional hardware, compatible artifacts and signing credentials are explicit prerequisites.

All targets are prospective. Unsupported combinations, insufficient samples and failed candidates remain visible; no benchmark result is extrapolated to all Macs or models.

## 简体中文 — 执行摘要

目标是在保持质量和持续稳定性的前提下，为每台Apple Silicon Mac、每个模型和工作负载选择经过认证的最快路径。本路线图不代表已经取得性能领先。

- [Done] 复用现有内存准入、调度器、进程隔离、性能记录和回滚基础。Gemma及视觉模型的证据仅适用于已测配置。包含未提交修改的旧路线图已完整保存。
- [Next] P0：核实真实后端能力，建立同等质量、可复现的直接访问与代理访问基准，记录TTFT、TPOT、尾延迟、满足SLO的有效吞吐、内存及失败。质量测试覆盖英语、日语和简体中文。
- [Next] P1：认证标准文本路径，完成取消、有界队列、内存准入、故障恢复及8小时混合负载测试。
- [Next] P2：验证后端拥有的连续批处理、分块prefill和真实KV复用。目标为并发4时有效吞吐提升20%，单请求p95 TTFT／TPOT退化不超过5%。
- [Later] P3：根据测量结果优化热点、量化和推测解码。只有质量通过且端到端提升至少5%、超过测量噪声时才采用。
- [Later] P4：进行各硬件配置的24小时稳定性认证，固定依赖，验证回滚、签名及全新环境安装。额外硬件、兼容模型文件和签名凭据是明确的前提条件。

以上数字均为未来验收目标。保留不支持的组合、样本不足和失败候选，不把局部结果推广到所有Mac或模型。

## 一次資料と更新方針

2026-09-26参照。[最新確認と採用判断](upstream-review-2026-09-26.md)。以下は比較候補の技術資料であり、このrepositoryでの性能認定の証拠ではない。実装時は参照commitとlicenseを固定する。

- [MLX-LM公式](https://github.com/ml-explore/mlx-lm)：Apple Silicon向け生成・量子化・streamingの比較基準。
- [vLLM-Metal公式](https://github.com/vllm-project/vllm-metal)：vLLMのApple Silicon plugin。使用buildの実効機能は個別に確認する。
- [llama.cpp公式](https://github.com/ggml-org/llama.cpp)：Metal／GGUF経路の比較対象。
- [vllm-mlx upstream](https://github.com/waybarrios/vllm-mlx)：continuous batchingとcache設計の比較対象。upstreamの性能値を本projectの実測として引用しない。

各開発サイクルで本書の状態・証跡・次の作業を更新する。詳細な実験履歴はevaluationと個別reviewへ置き、roadmapを実装履歴の追記だけで肥大化させない。
