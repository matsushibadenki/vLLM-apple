# P1 request timing — 2026-10-08

## 日本語

[Done] 実験P1経路に要求単位の数値時刻を追加。`/vllm-apple/resources`の`request_timings`から確認できる。
originは`generate`でresponse queueを作成した時刻であり、socket接続・header／body読込の開始ではない。
個々の時間差はmonotonic clock、他の記録との照合用開始時刻はunix clockを用いる。

| stage | 計測境界 |
| --- | --- |
| scheduler_dequeued | upstream schedulerが最初にqueueから要求を取り出した時刻 |
| context_received | HTTP threadがGenerationContextまたはerrorを受領した時刻 |
| first_response_ready | upstreamが最初のResponseをresponse queueへ入れる直前 |
| first_response_received | HTTP threadが最初のResponseを受領した時刻 |
| first_sse_write_started／returned | 最初の`data:`フレームのwrite呼出前後。keepalive／DONE／headerを除く |
| handler_finished | completion handlerのfinally到達。正常終了とfaultを含む |

SSE writeのreturnはsocket側の書込完了であり、clientの受信確認ではない。
role等のフレームが先行する場合、最初のSSE writeはtoken到達と一致しない。
tokenize、prefill、GPU kernel、body読込の時間は個別に測れていない。
context受領とResponse準備は異なるthreadなので、準備が先行する場合もある。
cancel・error・非stream要求では一部stageがなく、ゼロで補わない。

入力本文、生成内容、request ID、token、tensorを保存しない。backend内の数値sequenceだけを付与し、
完了履歴は最大64件、完了総数は別counterで保持。snapshotは独立したcopy。
markは最初の時刻だけを保存し、同じstageで上書きしない。
P1 profile有効時だけ要求traceとSSE writerを使用する。

[実モデルsmoke](evaluation/p1-request-timing-smoke-m4-2026-10-08.json)：
品質はwarmup 3/3、長文12/12、並列短文30/30。SLOはwarmup 2/3、長文12/12、短文30/30。
記録52件のうち全stageを持つ48件で、dequeue→context受領→Response受領、Response準備→受領、
SSE write前→後→handler終了の順序を確認。残る4件の欠測を合格値に補完しない。
cancel、queued cancel、timeout、slow consumer、header枯渇・回収、正常停止とruntime/model identity不変を確認。

対象13 tests成功（1 skip）。全Python回帰1468 tests成功（11 skip）、Ruff・diff check成功。
これは診断機能の検証で、速度向上・warmup SLO達成・長時間安定性・省電力の認定ではない。
計測による追加clock読込／lock処理があり、E2E計測影響の前後比較は未実施。

[Next] 短時間の同条件負荷でこの要求別時刻を収集し、queue待ち／backend初回結果待ち／HTTP受領・書込の遅延を分離する。
tokenize／prefill分離とcollectorへの要求identity接続、計測負荷の比較を続ける。
根因が確認できてから最小修正を選ぶ。長時間campaignは09:00 JST開始を維持する。

## English

[Done] Added bounded per-request host timing to the opt-in P1 route: scheduler dequeue, context receipt,
first Response readiness/receipt, first SSE data write and handler finish. Retains 64 completed numeric records;
no prompts, output, request IDs or tensors. Missing fault/nonstream stages remain absent.
Origin is response-queue creation, not socket/body ingress. SSE write return is not client receipt;
first data may be a role frame. Tokenization and GPU phases remain unseparated.
Actual smoke: long 12/12 and short 30/30 quality/SLO; warmup quality 3/3 and SLO 2/3.
Verified stage ordering for 48 complete records out of 52. Full regression: 1468 tests, 11 skipped; Ruff passed.
[Next] Use request timing to separate latency causes and measure instrumentation overhead; no speed/stability qualification yet.

## 简体中文

[Done] 在显式P1路径添加有界请求级host计时：scheduler dequeue、context接收、首个Response准备／接收、
首个SSE data写入和handler结束。最多保留64条完成数值记录，不保存输入、输出、request ID或tensor。
fault／非stream缺失stage不补零。起点为response queue创建，不是socket／body入口。
SSE write返回不代表client收到，首个data可能是role frame；tokenize及GPU phase仍未分离。
实际smoke：长文12/12、短文30/30质量及SLO通过；warmup质量3/3、SLO 2/3。
52条记录中48条完整stage顺序核对成功。全回归1468项成功（跳过11项），Ruff通过。
[Next] 利用请求级计时分离延迟原因并比较计测开销；速度和长期稳定性尚未认证。
