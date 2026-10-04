# RAG / LoRA integration

## 日本語

現在は、**外部検索済みのテキストを使うRAGブリッジ**を利用できます。LoRAの実学習・アダプター読み込み・切り替えは未実装です。完全対応までの計画は[roadmap](ROADMAP.md#ragとloraの段階的な完全対応)を参照してください。

入力JSONは `question`、`language`（`en` / `ja` / `zh`）、`documents` を持ちます。各documentには一意な `id`、`title`、`text` が必要です。検索側で権限確認済みの資料だけを、関連度順に渡してください。

```sh
# リクエストを組み立てるだけ。ネットワーク接続なし。
python -m vllm_apple rag examples/rag/input.json

# 起動済みのローカルOpenAI互換サーバーで生成する。
python -m vllm_apple rag examples/rag/input.json \
  --url http://127.0.0.1:8000 --model default_model \
  --max-source-bytes 32768 --max-tokens 256
```

`--model` は接続先が受け付けるモデル名に合わせてください。URLはループバックのoriginのみ指定でき、`/v1` は付けません。認証が必要な場合はPython APIの `answer_rag(..., session_token=...)` を使用できます。トークンをコマンドライン引数に渡す機能はありません。

- 予算に収まらないチャンクは丸ごと除外し、後続の小さいチャンクを検討します。除外IDは `omitted_document_ids` に返します。資料が残らなければHTTP生成せず、指定言語で回答不能を返します。
- `max-source-bytes` は資料部分のUTF-8予算です。`--url`と`--context-tokens`を指定すると同じbackendの`/tokenize`で質問・指示・chat template・generation promptを含む実token数を取得し、出力予約込みで収まるまで低優先度の資料を丸ごと除外します。資料ゼロでも基本promptが超過する場合は生成前に拒否します。tokenize失敗時にbyte予算へ黙って戻ることはありません。成功時だけ`token_budget_verified=true`となります。指定上限はモデル上限とserverのadmission上限以下にしてください。backendのモデル変更は検出していないため、固定modelのserverを使います。予算指定なしの従来経路はfalseを維持します。
- system role非対応template（配置済みGemma 2など）は`--no-system-role`で指示と資料JSONを一つのuser turnへ構成します。自動fallbackではなく明示設定です。
- `[S1]` のような参照IDを元の資料ID・タイトル・SHA-256へ対応付けます。`references_valid` は参照IDが実在するという意味で、回答内容の正しさや引用の意味的整合性を保証しません。`grounding_verified` は常に `false` です。
- 結果statusは `references_valid` / `abstained` / `invalid_references` / `missing_citations` / `incomplete`。CLI終了コードは通常成功・回答不能が0、引用不備・未完了が1、入力・通信エラーが2です。モデルが返す資料不足の判定自体は未検証です。
- CLI入力・HTTP応答は各1 MiBまで、入力チャンクは最大64件。HTTPは非streaming、既定timeoutは30秒、リダイレクトは追従しません。
- 資料内の命令を無視する指示を付けますが、prompt injection耐性の認定ではありません。ACL、検索、embedding、reranking、取り込み・削除は呼び出し側の責任です。準備モードの標準出力には資料本文が含まれます。生成結果にも資料IDとタイトルが含まれるため、保存・共有先に注意してください。

単体・模擬HTTPに加え、[実tokenizer試験](evaluation/rag-real-token-budget-m4-2026-10-04-r2.json)で三言語の境界・資料除外・基本prompt超過拒否を検証しました。[実Gemma HTTP試験](evaluation/rag-real-http-m4-2026-10-04-r2.json)は三言語の固定support codeの引用回答と資料なし時の回答不能に限定します。一般のgrounding・悪意ある資料耐性・性能・長時間安定性は未認定です。初回tokenizer probeはTransformersのdict返却を誤計数し不合格、初回HTTP probeはSIGTERM停止で不合格として保存し、修正後を別reportにしました。runtime変更後は対象構成で再認定してください。

```sh
vllm-apple rag retrieved.json --url http://127.0.0.1:8080 \
  --context-tokens 4096 --max-tokens 256 --no-system-role
```

Pythonのoffline準備では`prepare_rag(..., context_tokens=..., token_counter=...)`へ、
実際に生成するtemplateと同じ完全requestを数える関数を渡します。

2026-10-04：RAG対象15 tests、全回帰1430 tests成功（11 skip）、Ruff・差分検証成功。
実機probeの再現コマンド：

```sh
/opt/homebrew/opt/vllm-metal/libexec/bin/python scripts/probe_rag_token_budget.py \
  --model models/gemma-2-2b-it-4bit --output /tmp/rag-token-budget-new.json
/opt/homebrew/opt/vllm-metal/libexec/bin/python scripts/probe_rag_http.py \
  --model models/gemma-2-2b-it-4bit --output /tmp/rag-http-new.json
```

既存outputは上書きしません。HTTP probeは所有するbackendをSIGINTで停止し、例外時も
終了・失敗reportを記録します。性能計測中や他のGPU試験と並行して実行しないでください。

## English

The external-retrieval RAG bridge accepts `question`, `language` (`en`, `ja`, `zh`), and ordered `documents` with unique `id`, `title`, and `text`. Use the commands above to prepare a request offline or generate through a running loopback server. For authentication, the Python `answer_rag` API accepts `session_token`.

Oversized chunks are skipped whole; no selected sources means abstention without generation. The UTF-8 source budget alone is not a tokenizer guarantee. With `--url --context-tokens`, the same backend tokenizes the complete chat template and reserves output tokens, dropping lowest-priority whole sources until it fits. Base prompt overflow or failed tokenization blocks generation. Only a successful check sets `token_budget_verified=true`; choose a limit within the fixed model and server limits. Use `--no-system-role` for templates such as the installed Gemma 2 that reject system roles. Citation diagnostics validate IDs, not factual grounding; `grounding_verified` remains false. Exit codes: 0 for preparation, valid references or abstention; 1 for citation issues or incomplete generation; 2 for input/transport errors.

Retrieval, permissions, ingestion, embeddings, reranking and deletion remain external. Prompt instructions do not certify injection resistance. Prepared output contains source text; result metadata contains source IDs and titles. Actual tokenizer boundaries and three-language fixed support-code HTTP smoke were verified on M4/Gemma 2; general grounding, performance and sustained stability remain unqualified. Actual LoRA training/loading/switching is planned. Runtime changes require fresh evidence.

## 简体中文

当前提供外部检索RAG桥接：输入包含 `question`、`language`（`en`、`ja`、`zh`）和按相关度排序的 `documents`，每项必须包含唯一的 `id`、`title`、`text`。以上命令可离线准备请求，或连接已启动的本地服务器生成回答。需要认证时，可使用Python `answer_rag` 的 `session_token` 参数。

超出预算的片段整体跳过；没有资料时不调用生成。UTF-8预算本身不保证token上限。
指定`--url --context-tokens`后，同一backend计算完整chat template的真实token数，预留输出，
并从低优先级开始整体移除资料。基本prompt超限或tokenize失败时拒绝生成，不自动退回字节预算。
只有检查成功才设置`token_budget_verified=true`，上限必须在固定model和server限制内。
不支持system role的template可显式指定`--no-system-role`。引用检查不验证事实依据，
`grounding_verified`仍为false。退出码：成功或无法回答为0；引用问题或未完成为1；输入或通信错误为2。

检索、权限、导入、embedding、重排与删除仍由调用方负责。提示词不代表通过注入防护认证。
已在M4／Gemma 2验证真实tokenizer边界和三语言固定support code的引用回答及无资料时的拒答。
通用grounding、性能和长期稳定性未认证。LoRA仍在路线图中，runtime变化后需要新的认证证据。
