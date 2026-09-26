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
- `max-source-bytes` はシリアライズした資料部分のUTF-8予算です。質問・system prompt・chat templateを含むトークン数ではなく、モデルのcontext上限を保証しません。`token_budget_verified` は常に `false` です。
- `[S1]` のような参照IDを元の資料ID・タイトル・SHA-256へ対応付けます。`references_valid` は参照IDが実在するという意味で、回答内容の正しさや引用の意味的整合性を保証しません。`grounding_verified` は常に `false` です。
- 結果statusは `references_valid` / `abstained` / `invalid_references` / `missing_citations` / `incomplete`。CLI終了コードは通常成功・回答不能が0、引用不備・未完了が1、入力・通信エラーが2です。モデルが返す資料不足の判定自体は未検証です。
- CLI入力・HTTP応答は各1 MiBまで、入力チャンクは最大64件。HTTPは非streaming、既定timeoutは30秒、リダイレクトは追従しません。
- 資料内の命令を無視する指示を付けますが、prompt injection耐性の認定ではありません。ACL、検索、embedding、reranking、取り込み・削除は呼び出し側の責任です。準備モードの標準出力には資料本文が含まれます。生成結果にも資料IDとタイトルが含まれるため、保存・共有先に注意してください。

検証範囲は単体テストとローカルの模擬HTTPサーバーです。実モデルでの回答品質・性能・長時間安定性は未認定です。runtimeソース変更後は既存のidentity-bound evidenceを流用せず、認定対象の構成で再取得してください。

## English

The external-retrieval RAG bridge accepts `question`, `language` (`en`, `ja`, `zh`), and ordered `documents` with unique `id`, `title`, and `text`. Use the commands above to prepare a request offline or generate through a running loopback server. For authentication, the Python `answer_rag` API accepts `session_token`.

Oversized chunks are skipped whole; no selected sources means local abstention without generation. The UTF-8 source budget is **not a tokenizer/context-window guarantee**. Citation diagnostics validate reference IDs only, never factual grounding. Both `token_budget_verified` and `grounding_verified` (generation results) remain false. Exit codes: 0 for preparation, valid references or abstention; 1 for citation issues or incomplete generation; 2 for input/transport errors.

Retrieval, permissions, ingestion, embeddings, reranking and deletion remain external. Prompt instructions do not certify injection resistance. Prepared output contains source text; result metadata contains source IDs and titles. Tests use a mock HTTP server; real-model quality and performance remain unqualified. Actual LoRA training/loading/switching is planned in the roadmap. Runtime changes require fresh identity-bound qualification evidence.

## 简体中文

当前提供外部检索RAG桥接：输入包含 `question`、`language`（`en`、`ja`、`zh`）和按相关度排序的 `documents`，每项必须包含唯一的 `id`、`title`、`text`。以上命令可离线准备请求，或连接已启动的本地服务器生成回答。需要认证时，可使用Python `answer_rag` 的 `session_token` 参数。

超出预算的片段整体跳过；没有可用资料时直接返回无法回答，不调用模型。UTF-8资料字节预算**不保证符合模型的token或上下文上限**。引用检查只验证引用ID存在，不验证回答的事实依据。`token_budget_verified` 和生成结果中的 `grounding_verified` 始终为false。退出码：准备成功、引用ID有效或无法回答为0；引用问题或生成未完成为1；输入或通信错误为2。

检索、权限、文档导入、embedding、重排与删除仍由调用方负责。提示词不代表已通过注入防护认证。准备输出包含资料正文，结果元数据包含资料ID和标题。当前仅完成模拟HTTP测试，尚未认证真实模型的质量与性能。LoRA实际训练、加载及切换仍在路线图中。runtime变化后必须重新获取与身份绑定的认证证据。
