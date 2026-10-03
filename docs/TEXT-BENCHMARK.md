# Bounded text HTTP benchmark

[M4 two-client trickle validation](evaluation/wrapper-multi-trickle-body-m4-2026-10-03.json)
used queue capacity one, two connections declaring 100-byte bodies, and one-byte
uploads every 200 ms. Both requests returned 408 at about 10.002 seconds despite
continued uploads; an additional request returned 503. Inflight returned to zero,
no generation started, and three recovery generations passed quality. This verifies
the absolute body deadline and capacity recovery in this short test, not total HTTP
thread limits, header-read deadlines, or long-run stability.

上記M4試験は待機枠1、100 bytesを宣言する2接続、200 msごとに1 byteを送る条件で実施した。
送信が続いていても双方約10.002秒で408、追加要求503、最終使用枠0、不要な生成開始0、
後続3/3品質合格を確認した。短時間のbody絶対deadlineと枠回収の検証であり、
全HTTP thread上限、header読取りdeadline、長時間安定性の認定ではない。

上述M4测试使用等待名额1、声明100 bytes的2个连接、每200 ms发送1 byte。
持续上传未延长deadline，两者约10.002秒返回408，追加请求503，最终名额0，未启动多余生成，
后续3/3通过质量检查。这只验证短时body绝对deadline及名额回收，不认证全部HTTP thread上限、header读取deadline或长期稳定性。

Body reads now recheck the absolute deadline after each chunk, including the final
chunk; receiving a complete body after the deadline cannot bypass rejection.
[M4 slow-body smoke](evaluation/wrapper-slow-body-m4-2026-10-03.json) sent one byte
of a declared 100-byte body. With zero waiting slots, an additional request returned
503, the incomplete body returned 408 after about 10.002 seconds, and admission
returned to zero without starting generation. Three subsequent generation requests
passed quality. CPU regression tests ran concurrently; this is recovery evidence,
not a performance comparison. Multiple slow clients, trickle uploads and total
HTTP thread limits remain unverified.

bodyの最終chunkを含め、読取り後にも絶対deadlineを確認し、期限後の完全bodyを拒否する。
上記M4試験では100 bytes宣言のbodyを1 byteだけ送り、待機枠0で追加要求503、
未完bodyは約10.002秒後に408、生成開始なしで使用枠0に戻り、後続3/3品質合格を確認した。
CPU回帰を並行実行した回復確認であり、性能比較ではない。複数slow client、trickle送信、全HTTP thread上限は未検証。

每次读取chunk后（包括最终chunk）重新检查绝对deadline，拒绝期限后才完整到达的body。
上述M4测试声明100 bytes但只发送1 byte；等待名额0时追加请求返回503，不完整body约10.002秒后返回408，
未启动生成且名额归零，后续3/3通过质量检查。CPU回归同时运行，因此只是恢复验证，不是性能比较。
多个slow client、trickle上传和全部HTTP thread上限尚未验证。

Serialized admission now serves FIFO after body preparation finishes. This orders
ready requests, not connections or slow body uploads. Cancelled/timed-out tickets
are removed and followers notified. FIFO and cancelled-head tests passed 200
repetitions. [M4 regression](evaluation/wrapper-fifo-queue-cancel-m4-2026-10-03.json)
confirmed a queued disconnect did not start generation, slots returned to zero,
and three recovery requests passed quality. This is HTTP adapter scheduling;
backend token scheduling, long-run fairness and slow uploads remain unqualified.

直列化admissionはbody準備完了後のFIFOで処理する。接続順や遅いbody送信の開始順ではない。
取消／timeout ticketを除去して後続へ通知し、FIFOと先頭取消testは200回反復合格。
上記M4回帰で待機切断後の不要生成なし・最終枠0・後続3/3品質合格を確認した。
HTTP adapterの順序制御であり、backend token scheduler・長時間の公平性・遅いuploadは未認定。

串行admission按body准备完成后的FIFO顺序处理，不按连接建立或慢body上传的开始顺序。
取消／timeout ticket移除后通知后续请求；FIFO及取消队首测试重复200次通过。
上述M4回归确认等待断开后未启动多余生成、最终名额0、后续3/3通过质量检查。
这是HTTP adapter调度，后端token scheduler、长期公平性及慢上传尚未认证。

Regression validation also found a Qwen4 socket-race test cleanup failure.
The test now shuts down clients before joining workers and closes server sockets
afterward; the race case passed 200 repetitions. Qwen4 runtime behavior is unchanged.

回帰検証でQwen4 socket競合テストのcleanup失敗を確認した。client shutdown後にworkerをjoinし、
最後にserver socketをcloseするよう修正し、該当テスト200回反復合格。Qwen4 runtimeの動作は変更していない。

回归验证发现Qwen4 socket竞争测试的cleanup失败。测试先shutdown client，再join worker，
最后close server socket；该测试重复200次通过，Qwen4 runtime行为未修改。

Streaming BrokenPipe/ConnectionReset now increments `active_disconnects` separately
from queued cancellation, releases admission, and closes the handler without an
expected-disconnect traceback. Older telemetry may omit this counter; omission is
not zero. MLX-LM 0.32.0's completion finally calls `ctx.stop()`.
[M4 active-disconnect smoke](evaluation/wrapper-active-cancel-m4-2026-10-03.json)
received generated content before closing the client, observed admission return to
zero in approximately 51.5 ms, and passed three recovery requests. This is observed
adapter release time, not GPU stop latency. Non-stream cancellation, half-close,
and long-run cancellation remain unverified.

stream中のBrokenPipe／ConnectionResetは待機取消と別の`active_disconnects`へ記録し、枠を解放してhandlerを終了する。
旧telemetryがcounterを返さない場合はゼロ扱いにしない。MLX-LM 0.32.0のcompletion finallyは`ctx.stop()`を呼ぶ。
上記M4試験は生成content受信後に切断し、約51.5 msでadapter使用枠0、後続3/3品質合格を確認した。
これはadapter枠回収の観測時間で、GPU停止latencyではない。非stream取消、half-close、長時間負荷は未検証。

stream期间的BrokenPipe／ConnectionReset单独计入`active_disconnects`，释放名额并结束handler。
旧telemetry缺少该counter时不视为零。MLX-LM 0.32.0的completion finally调用`ctx.stop()`。
上述M4测试接收生成content后断开client，约51.5 ms观察到adapter名额归零，后续3/3通过质量检查。
这是adapter回收观测时间，不是GPU停止latency。非stream取消、half-close及长期负载尚未验证。

`generation_admission` exposes locked counters for inflight/active requests,
generation starts, cancellations, queue timeouts, rejections and body preparation
failures. The benchmark validates and preserves these counters; unsupported
endpoints/bypass return null. Inflight includes body preparation and waiting.
[M4 real-model queued disconnect](evaluation/wrapper-queue-cancel-m4-2026-10-03.json)
observed two inflight requests, then one cancellation without starting a second
generation. The first stream finished (200/DONE), slots returned to zero, and
three recovery requests passed quality. This does not verify active cancellation,
long soaks, FIFO or half-close. Admission's own snapshot is locked; the complete
memory/admission response remains non-atomic.

`generation_admission`に使用枠／active／生成開始／取消／queue timeout／拒否／body準備失敗を公開する。
counterはlock付きsnapshotで、benchmarkが非負整数を検証し保存する。未対応とbypassはnull。
使用枠にはbody準備と待機も含む。上記M4実model試験は使用枠2の後に待機clientを切断、
取消1・生成開始は先行1件のみ、先行200／DONE、最終使用枠0、後続3/3品質合格を確認した。
生成中cancel・長時間soak・FIFO・half-closeは未認定。admission単独snapshotはlock付きだが、応答全体はnon-atomic。

`generation_admission`公开使用名额／active／生成开始／取消／queue timeout／拒绝／body准备失败计数。
计数快照由lock保护，benchmark验证非负整数后保存；未支持及bypass为null，使用名额包含body准备和等待。
上述M4真实模型测试在使用2个名额后断开等待client，确认取消1次且只有先行生成启动，
先行200／DONE、最终名额0，后续3/3通过质量检查。生成中cancel、长时间soak、FIFO及half-close未认证。
admission独立快照由lock保护，但整个响应仍为non-atomic。

Serialized generation now reserves a capacity slot before buffering the request
body, limited to 8 MiB and an absolute 10-second read deadline. It requires one
Content-Length and rejects Transfer-Encoding. Invalid framing, oversized bodies,
and read timeouts return 400, 413, and 408 respectively and close the connection.
The body is replayed to MLX-LM after admission; socket timeout and input stream
are restored. This removes unread body bytes that previously hid EOF during waiting.
Real-socket tests verify cancellation and slot recovery; [M4 generation regression](evaluation/text-benchmark-body-admission-m4-2026-10-03.json)
passed 6/6 at client concurrency two. Half-close, active generation cancellation,
and long model queues remain unqualified. Body buffering is skipped by the experimental
concurrent-generation bypass.

直列化生成は容量枠を確保後、最大8 MiB・読取り絶対deadline 10秒でbodyを先読みする。
Content-Lengthは1つ必須、Transfer-Encodingは拒否。不正framingは400、size超過は413、timeoutは408で
connectionを閉じる。admission後にMLX-LMへbodyを再生し、socket timeoutと入力streamを復元する。
未読bodyがEOF検出を妨げる経路を解消し、実socketテストでcancelと枠回収を確認した。
上記M4正常生成はclient c2で6/6合格。half-close・生成中cancel・長いmodel queueは未認定。
実験用同時生成bypassでは、このbody先読みも行わない。

串行生成先预留容量名额，再读取body（最大8 MiB，绝对deadline 10秒）。要求单一Content-Length，拒绝Transfer-Encoding。
无效framing返回400、超出size返回413、timeout返回408并关闭连接。admission后向MLX-LM重放body，恢复socket timeout和输入stream。
解决未读body隐藏EOF的问题；真实socket测试验证cancel和名额回收，上述M4正常生成在client c2下6/6通过。
half-close、生成中cancel和长model queue尚未认证。实验性并发bypass不执行body预读。

Admission now checks cancellation while waiting (poll interval up to 50 ms) and
again before generation. Cancelled requests release both waiting and worker slots.
The wrapper uses a best-effort, non-consuming EOF/socket-error probe and does not
write an error to a detected disconnected peer. This is partial cancellation:
unread request body bytes can hide EOF, TCP read-side EOF is treated as withdrawal,
and active generation is not interrupted. Half-closed clients that still expect a
response are not qualified. Callback and recovery tests pass; real-model disconnect
and long-queue behavior remain unverified.

待機中（poll間隔最大50 ms）と生成直前にcancelを確認し、検出時は待機枠とworker枠を解放する。
wrapperはbyteを消費しないEOF／socket errorのbest-effort確認を行い、検出済み切断先へerrorを書かない。
部分的なcancel対応であり、未読bodyがEOFを隠す場合がある。TCP read-side EOFも取消とみなし、
応答を待つhalf-close clientは未認定。生成中の処理は中断しない。
callback・回復テストは合格、実model切断と長時間queueは未検証。

等待期间（poll间隔最大50 ms）及生成前检查cancel，检测后释放等待和worker名额。
wrapper使用不消耗byte的EOF／socket error尽力检测，不向已检测断开的peer写入error。
这只是部分cancel支持：未读body可能隐藏EOF，TCP读侧EOF也视为取消；仍等待响应的half-close client未认证。
不终止正在进行的生成。callback和恢复测试通过，真实模型断开及长队列行为尚未验证。

Serialized wrapper admission defaults to eight waiting generation requests and a
30-second queue timeout. `--generation-queue-capacity` accepts 0–128;
`--generation-queue-timeout` accepts positive finite seconds up to 300. Full queues
or expired waits return HTTP 503 with Retry-After and close the connection.
Timeouts and handler errors release admission slots. This bounds admitted generation
waiters, not all HTTP threads or inference duration. Ready requests use FIFO;
waiting EOF checks are polled and do not qualify half-close behavior. The experimental concurrency
bypass also bypasses these admission limits.
[M4 overload smoke](evaluation/text-benchmark-admission-m4-2026-10-03.json) with zero
waiters processed one request, rejected five with 503, then passed three recovery
requests. It is a recovery check, not a performance comparison.

直列化wrapperは生成待機8件・待機30秒を既定とする。`--generation-queue-capacity`は0〜128、
`--generation-queue-timeout`は有限の正数で最大300秒。満杯／待機timeoutは503＋Retry-Afterで
connectionを閉じ、timeout・handler例外後も枠を解放する。生成待機の上限であり、全HTTP threadや
推論時間の上限ではない。準備済み要求はFIFO、待機EOFはpollで確認しhalf-closeは未認定。実験用同時生成bypassはこの制限も迂回する。
上記M4待機枠0試験は1件処理・5件503拒否、後続3/3品質合格。性能比較ではなく回復確認である。

串行wrapper默认允许8个生成等待请求、等待30秒。`--generation-queue-capacity`范围0〜128，
`--generation-queue-timeout`为有限正数且不超过300秒。满队列或等待timeout返回503及Retry-After并关闭连接。
timeout或handler异常后释放名额。这只限制生成等待者，不限制所有HTTP thread或推理时长。
准备完成的请求使用FIFO，等待EOF通过poll检查，half-close未认证；实验性并发bypass也绕过这些限制。
上述M4零等待名额测试处理1个请求、拒绝5个请求（503），随后3/3恢复请求通过质量检查，不是性能比较。

The MLX wrapper now serializes generation POSTs by default. On M4 / Gemma 2 2B /
MLX-LM 0.32.0, [two-client concurrent generation](evaluation/text-benchmark-c2-memory-m4-2026-10-03.json)
timed out on all six measured attempts despite successful warmup and memory probes.
[With the serialization gate](evaluation/text-benchmark-c2-serialized-memory-m4-2026-10-03.json),
all six passed quality/SLO and 16/16 memory probes succeeded. Telemetry remains
accessible during generation. `--allow-concurrent-generation` bypasses the gate
for experimental tests. This is sequential HTTP adapter processing, not qualified
backend batching. Queue cancellation, overload bounds and long soaks remain unverified.

MLX wrapperの生成POSTを既定で直列化した。M4／Gemma 2 2B／MLX-LM 0.32.0のclient並列度2試験は、
修正前に本測定6件がすべてtimeoutしたが、直列化後は6/6品質／SLO合格、telemetry 16/16成功。
生成中もtelemetryは取得できる。`--allow-concurrent-generation`は実験用にgateを無効化する。
HTTP adapterの順次処理であり、backend batching認定ではない。queue cancel、過負荷制限、長時間soakは未検証。

MLX wrapper默认串行处理生成POST。M4／Gemma 2 2B／MLX-LM 0.32.0的双client测试中，
修复前6次正式请求全部timeout；串行化后6/6通过质量和SLO检查，telemetry 16/16成功。
生成期间仍可获取telemetry。`--allow-concurrent-generation`仅用于实验性绕过gate。
这是HTTP adapter顺序处理，不是后端batching认证。queue cancel、过载限制和长时间soak尚未验证。

Memory snapshots now declare `snapshot_consistency=non_atomic`; absent declarations
remain `unspecified`. Counters are not guaranteed to represent the same instant.
[M4 concurrent telemetry smoke](evaluation/text-benchmark-concurrent-memory-m4-2026-10-03.json)
passed 30/30 measured generation quality/SLO checks and 50/50 memory probes, with
100 ms waits between probes and a cap of 200. This overlaps one generation worker
with telemetry; it does not qualify multiple concurrent generations or performance
impact. CPU regression tests also ran during this final smoke, so timing is not a
performance comparison. All reported counters remain advisory.

メモリsnapshotに`snapshot_consistency=non_atomic`を明示し、旧endpointの欠落値は`unspecified`とする。
counterが同一時点を表す保証はない。上記M4同時取得試験は測定30/30品質／SLO合格、
取得後100 ms待機・上限200回のprobe 50/50成功。単一生成workerとtelemetryを重ねた試験であり、
複数同時生成や性能影響の認定ではない。最終試験中はCPU回帰テストも動作しており、速度比較には使わない。

内存快照显式声明`snapshot_consistency=non_atomic`，旧endpoint缺失该字段时保留`unspecified`。
各计数不保证来自同一时刻。上述M4测试30/30生成通过质量和SLO检查，probe 50/50成功，
每次采集后等待100 ms且最多200次。这是单一生成worker与telemetry并行的测试，不能认证多请求并行生成或性能影响。
最终测试期间CPU回归测试也在运行，因此不可用于速度比较。

[M4 tokenize / idle telemetry validation](evaluation/wrapper-tokenize-memory-smoke-m4-2026-10-03.json)
matched English/Japanese/Simplified Chinese token counts (22/20/19) to actual generation
usage, with 3/3 quality checks passed. An invalid model returned HTTP 400 and the
next valid tokenize request succeeded. Thirty idle memory probes took mean 9.672 ms
and maximum 12.663 ms. These are HTTP collection durations, not measured inference
interference, performance gains, or concurrent snapshot consistency.

上記M4試験でtokenize結果は英語22・日本語20・简体中文19 tokenとなり、実生成usageと一致、
3/3品質合格。不正modelはHTTP 400で拒否され、その後の正常要求は成功した。
idle memory取得30回は平均9.672 ms／最大12.663 ms。HTTP取得時間であり、
推論への干渉、性能改善、並列snapshot一貫性の検証ではない。

上述M4测试中，英语／日语／简体中文tokenize计数22／20／19与实际生成usage一致，3/3通过质量检查。
无效model返回HTTP 400，后续正常请求成功。30次idle memory采集平均9.672 ms、最大12.663 ms。
这是HTTP采集耗时，不能证明推理干扰、性能提升或并发快照一致性。

MLX-LM 0.32.0 wrapper compatibility is repaired for a single local worker, using
an explicit telemetry handler. [M4 wrapper validation](evaluation/text-benchmark-wrapper-memory-m4-2026-10-03.json)
passed 3/3 warmups and 30/30 measured quality/SLO checks. Both memory probes
observed 7,774,208 KV bytes from `backend_lru_accounting`. Token counts remain null
and traversal completeness false: this is backend bookkeeping, not an array audit.
Concurrent snapshot consistency, collection overhead, the tokenize endpoint and
other new API versions remain unverified. No speed improvement is claimed.

MLX-LM 0.32.0のwrapper互換性を単一local workerで修正し、telemetry handlerを明示接続した。
上記M4実測はwarmup 3/3・測定30/30品質／SLO合格、前後のKV容量7,774,208 bytesを取得した。
取得元は`backend_lru_accounting`で、token数null・traversal完全性falseを維持する。
backend管理値であり配列監査ではない。並列snapshot一貫性、取得負荷、tokenize endpoint、他の新API versionは未検証。速度改善は主張しない。

已修复MLX-LM 0.32.0单一本地worker的wrapper兼容性，并显式连接telemetry handler。
上述M4测试warmup 3/3及正式测量30/30通过质量和SLO检查，前后均获取7,774,208 KV bytes。
来源为`backend_lru_accounting`，token数量保留null，遍历完整性为false；这是后端管理值而非数组审计。
并发快照一致性、采集开销、tokenize endpoint和其他新API版本尚未验证，不主张速度提升。

`prompt_cache_usage` aggregates validated response usage `prompt_tokens_details.cached_tokens`.
Missing, boolean, negative, or larger-than-prompt values remain unavailable; warmup is excluded.
The reuse ratio covers observed prompt tokens only, not missing attempts. Eviction counts
remain null. [M4 MLX-LM smoke](evaluation/text-benchmark-cache-usage-m4-2026-10-03.json)
passed 30/30 measured requests and reported 580/610 reused prompt tokens. This is
backend-reported reuse, not an independent KV correctness or performance certification.
The stock memory endpoint returned 404. The wrapper failed startup on installed
MLX-LM 0.32.0 (`prompt_cache_size` missing); compatibility repair remains pending.

`prompt_cache_usage`は応答usageの`prompt_tokens_details.cached_tokens`を検証して集計する。
欠測・bool・負数・prompt token数超過は未取得に残し、warmupを除外する。
再利用比率の分母は取得済みprompt tokenのみで、evictionはnull。
上記M4試験は30/30品質・SLO合格、backend報告で580/610 prompt token再利用を取得した。
KV正確性や性能改善の独立認定ではない。標準memory endpointは404。
実wrapperはMLX-LM 0.32.0で`prompt_cache_size`欠落により起動失敗し、互換性修正が残る。

`prompt_cache_usage`验证并汇总响应usage中的`prompt_tokens_details.cached_tokens`。
缺失、bool、负值及超过prompt token数的值保留为未获取，排除warmup。
复用比例仅以已获取的prompt token为分母，eviction为null。上述M4测试30/30通过质量和SLO检查，
后端报告580/610 prompt token复用；这不是KV正确性或性能提升的独立认证。
标准memory endpoint返回404。MLX-LM 0.32.0的wrapper因缺少`prompt_cache_size`启动失败，兼容性修复尚未完成。

`--collect-backend-memory` optionally reads the wrapper's `/v1/vllm-apple/memory`
before and after measurement, outside the goodput timer. Reads are limited to
64 KiB and one second; redirects are refused. Only validated capacity counters
and traversal completeness are stored. Unsupported endpoints and invalid payloads
remain unavailable. KV hits/evictions are null, not inferred from token counts.
Allocator cache bytes are not prefix-cache hit statistics. Support beyond the
0.32.0 single-worker smoke and collection overhead remain unverified.

`--collect-backend-memory`でwrapperの`/v1/vllm-apple/memory`を測定前後に任意取得する。
goodput計測窓外で、64 KiB・1秒に制限しredirectを拒否する。検証済み容量counterと
traversal完全性のみ保存し、未対応・不正payloadは欠測にする。KV hit／evictionはnullで、
token数から推測しない。allocator cache容量はprefix hit統計ではない。0.32.0単一workerのsmokeを超える対応と取得負荷は未検証。

`--collect-backend-memory`可在测量前后读取wrapper的`/v1/vllm-apple/memory`，位于goodput计时窗口之外。
读取限制为64 KiB及1秒，拒绝重定向。只保存验证后的容量计数和遍历完整性；未支持或无效数据保留为缺失。
KV hit／eviction为null，不根据token数量推断。allocator cache容量不是prefix命中统计。超出0.32.0单一worker smoke的支持和采集开销尚未验证。

M4実機確認（2026-10-03）：[Gemma 2 2B／MLX-LM direct report](evaluation/text-benchmark-distribution-m4-2026-10-03.json)でwarmup 3/3、測定30/30が品質・SLO合格。
TTFT平均165.639 ms／最大256.815 ms、E2E平均225.022 ms／最大348.379 msで、各histogramは30 sample・欠測0。
3点のsystem観測も取得でき、温度nominal／AC Power／automatic、メモリpressure推定normalだった。
単一路線の短時間smokeであり、速度改善・ばらつき原因・cache hitは認定しない。artifact／build digest未指定のため比較資格も付与しない。

M4 validation (2026-10-03): the linked Gemma 2 2B / MLX-LM direct report passed
3/3 warmups and 30/30 measured quality and SLO checks. TTFT mean/max were
165.639/256.815 ms; E2E mean/max were 225.022/348.379 ms. Both histograms contain
30 samples with no missing observations. All three system snapshots were saved.
This single-route smoke does not establish faster inference, variance causes,
or cache hits. Artifact/build digests were omitted, so comparison is unqualified.

M4实机验证（2026-10-03）：上述Gemma 2 2B／MLX-LM direct报告中，warmup 3/3和正式测量30/30通过质量及SLO检查。
TTFT平均／最大为165.639／256.815 ms，E2E平均／最大为225.022／348.379 ms；两个直方图均有30个样本且无缺失。
三次system快照均已保存。这是单一路径短时验证，不能证明速度提升、波动原因或缓存命中。
未指定artifact／build digest，因此不授予比较资格。

`latency_distributions` saves constant-memory TTFT and E2E histograms: disjoint
buckets include their upper boundary; a final null boundary denotes overflow.
Counts include completed responses that fail quality or SLO, and exclude warmup.
Failed attempts and missing E2E `[DONE]` timestamps remain unavailable, never zero
latency. Mean/max are null when no samples exist. These distributions describe
client arrival times, not internal backend phases or cache hits.

`latency_distributions`は固定容量のTTFT／E2E histogramを保存する。各bucketは上限を含み、
最後のnull上限はoverflowを示す。品質・SLO不合格の完了応答も含み、warmupは除外する。
失敗とE2Eの`[DONE]`欠測は未取得件数に残し、遅延ゼロとして扱わない。
sampleなしのmean／maxはnull。backend内部phaseやcache hitの計測ではなく、クライアント到着時間を示す。

`latency_distributions`保存固定容量的TTFT／E2E直方图。各区间包含上限，最后的null上限表示溢出。
统计包含质量或SLO不合格的完整响应，不包含warmup。失败和E2E的`[DONE]`缺失保留为未获取次数，
不会被计为零延迟。没有样本时mean／max为null。这是客户端到达时间，不是后端内部阶段或缓存命中测量。

## 日本語

`python -m vllm_apple.text_benchmark`は、既に起動したloopback HTTP backendに同じ英語・日本語・简体中文の算術promptを送り、並列負荷と通信latencyを計測する。モデルの起動・download・更新は行わない。

```bash
.venv/bin/python -m vllm_apple.text_benchmark \
  --base-url http://127.0.0.1:19096 \
  --model /absolute/path/to/model --backend mlx_lm_direct \
  --hardware-fingerprint Apple-M4-32GiB \
  --warmup-requests 3 \
  --requests 30 --concurrency 2 --max-tokens 16 \
  --ttft-slo-ms 1000 --e2e-slo-ms 5000 > /tmp/text-benchmark.json
```

認証が必要なら`--session-token-file`、RSSを観測するならbackendの`--target-pid`を指定する。token・endpoint・prompt・生成本文はreportに保存しない。モデル識別子は保存するので、共有するreportのローカルパスには注意する。

`--collect-operating-context`でwarmup前・測定直前・測定直後の3点にUTC時刻、温度状態、
電源供給元、電源モードを保存する。`--target-pid`を併用すると対象processの起動後経過秒数も記録する。
PIDには推論workerを指定する。daemon frontendのPIDではworkerの経過時間を表さない。
probeは各command 1秒timeoutで、goodputの計測窓外に実行する。取得不能は`unknown`／`null`で残す。
各snapshotの`system`に1／5／15分のload average、logical CPU数、利用可能メモリの推定値と取得元を保存する。load averageはCPU使用率やbackend単独の負荷ではない。`pressure_estimate`は利用可能メモリ比率8%／18%から算出した推定で、OSの正式なpressure判定ではない。取得不能はnull、保守的なメモリfallbackは`available_is_fallback`で区別する。これらは原因調査用で、比較gateや自動調整の認定条件には追加しない。

これは3点の観測であり、途中の温度変化、cache hit率、processの入れ替わりは認定しない。
比較器の`--require-operating-context`で観測条件のgateを有効にできる。両routeの3点で温度nominal、
同じ既知の電源供給元・電源モード、UTC時刻の順序、測定前の同一UTC日、worker ageの非減少と
測定前の経過時間差5秒以内を要求する。欠測・条件変化は`blocked_operating_context`と理由を保存する。
5秒は比較条件として固定した許容値であり、最適値の実測ではない。gate通過後も連続温度安定性や
性能優位の認定には広げない。[M4観測smoke](evaluation/benchmark-context-m4-2026-10-01.json)
では温度nominal、Battery Power、automaticと観測process自身のage 0秒を取得した。推論性能は未測定である。

- 固定数のworkerだけを作るclosed-loop方式。requestは1〜100,000、並列度は1〜32かつrequest数以下。request数に比例するfutureや生成本文を保持しない。
- HTTP／usage／途中切断の失敗は件数とcodeを保存する。完了率の分母から除外しない。
- goodputは、前後の空白を除いた答えが`2`で、TTFTと`[DONE]`までの時間が指定SLO以内のrequestのusage output tokensを、負荷実行全体のwall timeで割る。並列requestの時間を加算して分母にしない。
- 言語別の試行・完了・品質・SLO合格件数を保存する。全件正答でCLI終了code 0、失敗／品質不合格があれば1。SLO未達はreportで別に判定する。
- workload hashはprompt・期待値・warmup件数・request数・並列度・sampling・timeout・SLOを固定する。endpointを含めないのでdirect／daemon間で照合できるが、同一artifactやbackend buildの証明にはならない。
- 検証済みのmodel integrity root digestとbackend環境digestがある場合だけ、`--artifact-identity-sha256`と`--backend-build-sha256`を同時指定できる。片方だけ、64桁小文字hex以外は拒否する。指定なしの既存reportはidentity未検証のままである。
- Python APIの`cases=`には最大64件の`(label, prompt, expected)`を渡せる。labelは重複不可、64 bytes以下。prompt 8 MiB、expected 1 KiBを上限とし、内容をreportへ保存せずworkload hashへ結合する。CLIは固定三言語workloadだけを実行する。
- p99はhistogram上限であり、測定完了数1,000未満は参考値。backend内部token timestamp、queue、tokenize、allocator、swap、thermal、energyは未取得と明記する。RSSを指定していない場合、phase profileの既存0値は実測ゼロではない。
- `--warmup-requests`は0〜100件。同じ三言語caseを順番に送ってからtimerとprofilerを開始する。warmupの試行・完了・品質・errorをreportへ保存し、測定request数とgoodputから除外する。比較器は両routeのwarmup件数が同じで全件成功した場合だけ比較を許可し、失敗時は`blocked_warmup_failure`にする。
- warmupは起動直後のcache条件を近づけるが、prefix hit率やcache容量を直接固定しない。`cache_policy=backend_managed_conditioned`はcold／warm／prefix hitや純粋なdecode速度の認定ではない。warmup 0件では従来どおりcache非制御である。HTTPタイムアウトはsocket待機の上限で、request全体のhard deadlineではない。

M4 smokeの起動条件は[実測report](evaluation/text-benchmark-m4-smoke-2026-09-26.json)の`command`に保存した。Homebrewの既存MLX-LM serverでGemma 2 2Bを起動し、3件warmup後、並列度1と2で各30件実行する。試験後は起動したbackendだけを終了・回収する。これは配線と三言語算術の確認であり、一般品質、continuous batching、性能優位、長時間安定性を認定しない。

同じidentity引数とworkloadで取得したdirect／daemon reportは次で比較する。

```bash
.venv/bin/python -m vllm_apple.text_benchmark_comparison \
  --direct /path/to/direct.json --proxy /path/to/proxy.json \
  > /tmp/text-route-comparison.json
```

比較器は入力を各4 MiB以下のregular fileに限定し、元reportのSHA-256も保存する。workload、
concurrency、SLO、artifactまたはbackend buildが違えば比較自体を拒否する。両経路の全件品質が
合格し、identityが検証済みで、goodputが取得でき、各routeが最低30件ある場合だけ
`comparable`とする。30件未満は比率を保存して`blocked_insufficient_samples`とする。
p99は1,000件未満なら別途`p99_reference_only=true`を維持する。この結果も単独では
performance qualificationにしない。

[2026-09-28のM4短時間比較](evaluation/text-route-comparison-m4-2026-09-28.json)では、同じ
Gemma 2 artifact root SHA-256とMLX-LM環境SHA-256を束縛し、directとdaemonが各9/9正答・
SLO合格だった。goodputはdirect 15.611、daemon 15.979 tokens/s、比は1.023544、p99 histogram
上限は双方500 ms。sampleは9件、closed-loop並列度1、cache非制御であり、約2.4%の差は測定
noiseと起動後cache状態を分離できない。性能改善や一般的なproxy costとして採用しない。
runtime変更を旧30分証跡がidentity mismatchとして正しく拒否したため、daemonは認定を伴わない
`--skip-backend-check`で起動した。比較器も最低30件のgateにより
`blocked_insufficient_samples`を返す。全processは試験後に回収し、reportの`qualification`はfalseである。

同条件を30件へ増やした[baseline report](evaluation/text-route-comparison-30req-m4-2026-09-28.json)
は、両経路30/30正答・SLO合格、direct 17.452、daemon 17.657 goodput tokens/s、比1.011736で
最低sample gateに合格した。p99 histogram上限は双方500 msだが、30件なので
`p99_reference_only=true`である。単一のclosed-loop c1 runかつcache非制御のため、約1.2%差を
性能改善や一般的なproxy overheadとして扱わない。反復・順序交替・cache条件固定が次の比較条件である。

反復比較は3〜21個のpair reportを渡す。

```bash
.venv/bin/python -m vllm_apple.text_benchmark_series \
  /path/to/pair-1.json /path/to/pair-2.json /path/to/pair-3.json \
  > /tmp/text-route-comparison-series.json
```

seriesは同じworkload／artifact／backend buildだけを受け付け、direct-firstとproxy-firstの両方を
必要とし、順序件数差を1以内に制限する。proxy/direct goodput比の中央値・最小・最大・相対幅を
保存し、相対幅5%超を`blocked_variance`とする。[M4 3×30件series](evaluation/text-route-comparison-series-3x30-m4-2026-09-28.json)
は順序2:1、比1.011736／1.005702／1.006541、中央値1.006541、相対幅0.5995%で
`comparable_stable_reference`だった。全90件／routeが品質・SLO合格だが、各pairはcache非制御の
c1でp99も参考値である。約0.65%差を性能優位として認定せず、series自体も`qualification=false`とする。

[warmup 3件後のM4比較](evaluation/text-route-comparison-warm3-30req-m4-2026-09-29.json)は、
両routeでwarmup 3/3、測定30/30が品質・SLO合格した。direct 18.448、daemon 18.783
goodput tokens/s、proxy/direct比1.018188、p99 histogram上限は双方250 msで`comparable`だった。
これはwarmup実装と比較gateの実機確認である。direct-firstの単一pair、closed-loop c1、backend管理cache、
各30件の参考p99に限定されるため、約1.8%差を性能改善として認定しない。

[warmup固定の3×30件series](evaluation/text-route-comparison-series-warm3-3x30-m4-2026-09-30.json)では、
各routeのwarmup 9/9と測定90/90が品質・SLO合格し、実行順序はdirect-first 2／proxy-first 1だった。
proxy/direct goodput比は1.018188／1.071009／0.956578、中央値1.018188、相対幅11.2387%で
5% gateを超え、`blocked_variance`となった。warmupだけでは変動を抑えられない。
このseriesは2日間にまたがり、power／thermal、起動後経過時間、実際のcache hit率を束縛していない。
差の原因を特定したとは扱わず、性能改善は未認定とする。

2026-10-01の[operating context付き3×30件series](evaluation/text-route-comparison-series-context-3x30-m4-2026-10-01.json)
では、各routeをfresh processで起動し、worker age 30秒以降にwarmupを開始した。測定前ageは
30〜31秒、全18観測でnominal／AC Power／automatic、各routeのwarmup 9/9と測定90/90が
品質・SLO合格し、3 pairともcontext gateを通過した。goodput比は1.006013／0.984572／1.170932、
相対幅18.5246%で`blocked_variance`となった。電源・温度・ageの一致だけでは変動を解消できない。
CPU負荷やcache hitは未取得なので原因を特定せず、性能改善は未認定とする。
[起動flagsと再現条件](evaluation/text-context-series-protocol-m4-2026-10-01.json)を保存した。
各benchmarkに`--collect-operating-context --target-pid <worker-pid>`を付け、比較器には
`--require-operating-context`を指定する。daemonはbackend check skipを明示し、資格認定を付与しない。

### M4で見つかった並列失敗

MLX 0.32.1／MLX-LM 0.32.0、Gemma 2 2B、serverのprompt／decode concurrencyを2に設定した。並列度1は30/30正答。並列度2の最初の試験は30秒timeoutが続いたためbackendを手動停止した。停止後の接続失敗を含み、比較baselineとして使用しない。

[短い独立再試験](evaluation/text-benchmark-m4-c2-repro-2026-09-26.json)は、3件warmup後に5秒timeout・並列度2・3件を実行し、1件成功、2件失敗。backendの`_generate`threadから`batch_generator.next()`、Gemma 2 attentionの`mx.where`へ進み、次の例外でthreadが終了した。

```text
ValueError: [broadcast_shapes] Shapes (2,1,1,20) and (2,4,2,1,20) cannot be broadcast.
```

これはこのmodel／build／負荷条件での失敗であり、全MLXモデルへの一般化はしない。既存の並列度1の認定を並列度2へ広げない。mask／batch／cache契約の調査、upstream修正候補との照合、修正後のcancel／回復と並列再試験を`[Next]`とする。Homebrew管理下のpackageは変更していない。実行した子processは両試験で回収済みで、SIGTERM終了を自然な正常終了の認定には数えない。

## English

The context-gated M4 3×30 series passed all three pair gates, 9/9 warmups and 90/90
measured requests per route. All 18 snapshots were nominal / AC Power / automatic;
starting worker ages were 30–31 seconds. Ratios were 1.006013, 0.984572, and 1.170932.
The 18.5246% spread exceeded 5%, yielding `blocked_variance`. Matching these observations
did not establish stable performance. CPU load and actual cache hits remain unobserved.

Use `--collect-operating-context` to record UTC time, thermal state, power source,
and power mode before warmup, before measurement, and after measurement. With
`--target-pid`, worker process age is also recorded. Probes run outside the goodput
window; unavailable values remain unknown/null. Three snapshots do not establish
continuous thermal stability or cache hits. `--require-operating-context` requires
nominal thermal snapshots, identical known power conditions, ordered UTC timestamps,
the same UTC measurement date, nondecreasing worker ages, and starting ages within
five seconds. Missing or changed conditions yield `blocked_operating_context` with
reasons. The five-second tolerance is a fixed comparison policy, not a measured optimum.

Each snapshot also stores `system`: 1/5/15-minute load averages, logical CPU count,
and estimated available memory with its source and fallback flag. Load average is
neither CPU utilization nor backend-only load. `pressure_estimate` uses available
memory ratio thresholds of 8% and 18%; it is not the OS pressure classification.
Missing observations are null. These diagnostic fields do not change comparison
gates or qualify automatic tuning.

Run a bounded closed-loop workload against an existing local HTTP backend. Reports
retain failures, per-language arithmetic quality, client timings, and quality-gated
SLO goodput using total wall time. Optional bounded warmups run before measurement;
their counts, quality, and errors are reported, bound into the workload, and excluded
from measured goodput. Workload hashes enable input matching but do not verify artifact
identity. Warmups condition backend-managed caches but do not certify a cache state;
internal timings are unavailable.
This smoke benchmark does not certify general quality, batching or performance superiority.
Verified model and backend SHA-256 identities can be bound explicitly. The comparison
tool rejects mismatched workloads or identities, retains failures, and blocks a performance
conclusion when identity, quality, or required metrics are unavailable.
The M4 identity-bound smoke passed 9/9 requests on both routes. Direct and daemon
goodput were 15.611 and 15.979 tokens/s respectively, a 1.023544 ratio. Nine
closed-loop samples with uncontrolled caches cannot establish a performance gain;
the comparison is mechanically blocked by the 30-request minimum.
The follow-up 30-request baseline passed 30/30 on both routes. Direct and daemon
goodput were 17.452 and 17.657 tokens/s, a 1.011736 ratio. This single uncontrolled-cache
run is comparable input, not evidence of a performance improvement; p99 remains reference-only.
An order-balanced 3×30 series produced proxy/direct goodput ratios of 1.011736,
1.005702, and 1.006541. The median was 1.006541 with 0.5995% relative spread.
This is stable reference evidence for the bounded c1 workload, not a performance qualification.
A warmup-conditioned M4 pair passed 3/3 warmups and 30/30 measured requests on each
route. Direct and daemon goodput were 18.448 and 18.783 tokens/s, a 1.018188 ratio.
This direct-first single pair validates the warmup gate; it does not establish a performance gain.
The three-pair warmup series passed 9/9 warmups and 90/90 measured requests per route,
with direct-first and proxy-first orders represented. Goodput ratios were 1.018188,
1.071009, and 0.956578; the 11.2387% relative spread exceeded the 5% gate and
yielded `blocked_variance`. Power, thermal state, process age, and actual cache hits
remain unbound, so no performance gain is certified.

## 简体中文

M4的context gate三组比较全部通过条件检查，每条route的warmup 9/9及测量90/90均通过。
18次快照均为nominal／AC Power／automatic，测量前worker运行时间为30〜31秒。
比值为1.006013、0.984572及1.170932，相对范围18.5246%超过5%，结果为`blocked_variance`。
这些观测条件一致仍不能证明稳定性能；CPU负载及实际缓存命中尚未获取。

使用`--collect-operating-context`可在warmup前、正式测量前后记录UTC时间、温度状态、
电源来源和电源模式。配合`--target-pid`可记录推理worker启动后的秒数。
探测在goodput计时窗口之外运行；无法获取的值保留为unknown／null。
三次快照不能证明持续温度稳定或缓存命中。`--require-operating-context`要求温度均为nominal、
已知电源条件一致、UTC时间有序、测量日期一致、worker运行时间不减少且测量前差值不超过5秒。
缺失或变化会产生`blocked_operating_context`及原因。5秒是固定比较策略，尚未实测其最优性。

每次快照还在`system`中保存1／5／15分钟平均负载、逻辑CPU数量、可用内存估计及来源。
平均负载不是CPU使用率或后端独占负载。`pressure_estimate`使用可用内存比例8%／18%的阈值，
不是操作系统正式的内存压力判定。缺失值为null，保守回退由`available_is_fallback`标识。
这些字段仅用于诊断，不改变比较gate，也不认证自动调优。

对已启动的本地HTTP后端运行有界闭环负载，保留失败、各语言算术质量、客户端延迟，
并按总墙钟时间计算满足质量及SLO条件的有效吞吐。可选的有界warmup在正式测量前运行；
报告保存其数量、质量和错误，将数量绑定到工作负载，并从测量吞吐中排除。
工作负载哈希用于核对输入，不证明模型文件相同。warmup只能使后端管理的缓存条件更接近，
不代表缓存状态认证；后端内部计时不可用。
此冒烟测试不代表一般质量、连续批处理或性能优势认证。
可以显式绑定已验证的模型及后端SHA-256 identity。比较工具拒绝不同workload或identity，
保留所有失败，并在identity、质量或必要指标缺失时阻止性能结论。
M4 identity-bound短测中direct和daemon均通过9/9个请求；goodput分别为15.611和15.979
tokens/s，比值为1.023544。仅9个闭环样本且cache未受控制，不能证明性能提升。
比较器也会通过每条route至少30个请求的gate自动阻止该结论。
后续30请求baseline中两条route均通过30/30，goodput分别为17.452和17.657 tokens/s，
比值为1.011736。该单次、cache未受控制的结果只表示输入可比较，不证明性能提升；p99仍仅供参考。
交替顺序的3×30 series得到1.011736、1.005702和1.006541三个proxy/direct goodput比，
中位数为1.006541，相对范围为0.5995%。这只是有界c1工作负载的稳定参考，不属于性能认证。
加入3次warmup后的M4比较中，两条route均通过3/3次warmup及30/30个测量请求。
direct与daemon goodput分别为18.448和18.783 tokens/s，比值为1.018188。
该direct-first单组结果只验证warmup gate，不证明性能提升。
固定warmup的三组比较中，每条route的warmup 9/9及测量90/90均通过，并包含两种执行顺序。
goodput比值为1.018188、1.071009和0.956578；相对范围11.2387%超过5%的门槛，
结果为`blocked_variance`。电源、温度、进程启动后的时间及实际缓存命中率尚未绑定，
因此不能认定性能提升。
