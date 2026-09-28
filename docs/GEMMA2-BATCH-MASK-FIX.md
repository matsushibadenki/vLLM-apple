# Gemma 2 batch mask compatibility

## 日本語

MLX-LM 0.32.0の確認済みGemma 2実装は、Grouped Query Attentionでscoreを
`[B, KV heads, repeats, Q, K]`へ分ける一方、batch用maskを`[B, 1, Q, K]`のまま適用する。
batch軸がhead軸へ誤って対応し、並列度2ではbroadcast例外で生成threadが終了した。

`mlx_gemma2_compat`は、共有headの4次元maskだけを`[B, 1, 1, Q, K]`へreshapeして
元のattentionへ渡す独立実装。bool／加算maskとも値を変えず、weight、RoPE、cache、
softcap、softmaxは元の演算を使う。2次元mask、非GQA、maskなしの経路は変更しない。
対応しないhead別4次元maskは明示拒否する。

適用条件はMLX-LM `0.32.0`と`models/gemma2.py`のSHA-256
`64b0935b06fe2c4d5d4ed23a9cf62deb6218c55a88b9403a657afe9e2be8f251`の一致。
一致しなければ起動を拒否し、新版への暗黙適用を防ぐ。インストール済みファイルは書き換えない。
通常の`serve`／`mlx_server`は変更せず、専用の実験用entry pointでだけ適用する。

repository rootから、MLX入りPythonで起動する：

```bash
PYTHONPATH=. /opt/homebrew/opt/vllm-metal/libexec/bin/python \
  -m vllm_apple.mlx_gemma2_compat \
  --model models/gemma-2-2b-it-4bit --host 127.0.0.1 --port 19096 \
  --decode-concurrency 2 --prompt-concurrency 2 --prompt-cache-size 4
```

引数解析とserver lifecycleはupstreamへ委譲する。[benchmark手順](TEXT-BENCHMARK.md)で
3件warmup後、並列度1／2で各30件を測定した。[実測report](evaluation/text-benchmark-m4-gemma2-mask-fix-2026-09-26.json)は
両方30/30正答・HTTP失敗0。試験processはSIGTERM後に回収済み。速度向上や自然shutdownの認定ではない。

GPU数値回帰はbatch 2／3 × query長1／4 × bool／加算maskの8条件で、各batch行の異なる
maskを使用し、未修正の逐次attentionと`atol=rtol=1e-5`で一致した。実行コマンド：

```bash
VLLM_APPLE_TEST_GEMMA2_MASK=1 /opt/homebrew/opt/vllm-metal/libexec/bin/python \
  -m unittest tests.test_mlx_gemma2_compat
```

数値回帰は小型F32 attention、HTTP試験は配置済み4-bit実モデルの短い算術promptに限定する。
既存の並列度1用証跡を、
この新しい実行経路の証跡として流用しない。rollbackは専用起動をやめ、既存の認定経路へ戻す。

追加の短時間qualificationは、約2K prompt tokensの同一prefix末尾を3種類に編集した
並列12件と、三言語算術の並列100件を実行した。いずれも完了・品質・設定SLOが全件合格し、
継続負荷のgoodputは19.283 output tokens/s、観測peak RSSは2,232,991,744 bytesだった。
長文側p95 TTFTはhistogram範囲外の`>5000 ms`、最大7,249.752 msであり、短い入力の
性能値として扱わない。SSEを最初のdata受信後にclientから閉じ、1秒後の正常応答も確認した。
初回の切断試験ではupstream serverからcancel完了通知を得られなかった。そのため確認済み
`server.py`のsource hashに限定し、`X-VLLM-Apple-Request-ID`でactive generation contextと
request専用response queueを一時登録する実験用bridgeを追加した。次のendpointはactive requestへ
`stop()`を通知し、同じrequestのresponse queueへ終了を送る。GPU batchからのremoveはupstreamの
safe pointで行われ、cancelled requestのpartial cacheを保存しない。

```text
DELETE /vllm-apple/requests/{request-id}
```

未知・完了済みIDは404、active IDは202。IDはASCII英数字と`._-`の1〜64文字に制限し、
重複IDの新規requestはその新規contextだけを停止する。標準MLX serverにはendpointを追加しない。
[2026-09-27 report](evaluation/gemma2-batch-mask-cancel-slow-m4-2026-09-27.json)では最初のSSE data後に
cancelし、HTTP 202から0.759 msで`[DONE]`を観測した。1 KiB receive bufferで最初の読み取りを
1秒止めたslow consumerも48,651 bytesを完走し、その後の正常応答に合格した。

再現コマンド：

```bash
.venv/bin/python scripts/qualify_gemma2_batch_mask.py \
  --python /opt/homebrew/opt/vllm-metal/libexec/bin/python \
  --model models/gemma-2-2b-it-4bit \
  --output docs/evaluation/gemma2-batch-mask-cancel-slow-m4-2026-09-27.json \
  --port 19100 --sustained-requests 100 --long-requests 12
```

runnerは固定数workerを使い、reportをatomic保存する。仮想環境launcherのsymlinkを保持して
`sys.prefix`を変えず、local modelだけをofflineで起動する。これは約36秒の回帰試験であり、
30分認定、cancel反復負荷、一般的な長文品質は未認定。

30分の混合負荷は次の明示的な時間gateで実行できる。

```bash
.venv/bin/python scripts/qualify_gemma2_batch_mask.py \
  --python /opt/homebrew/opt/vllm-metal/libexec/bin/python \
  --model models/gemma-2-2b-it-4bit \
  --output docs/evaluation/gemma2-batch-mask-30min-m4-2026-09-27.json \
  --port 19102 --sustained-requests 100 --long-requests 12 \
  --duration-seconds 1800 --require-30-minute-window
```

[2026-09-27の30分report](evaluation/gemma2-batch-mask-30min-m4-2026-09-27.json)は
通常応答3162/3162正答、cancel 310/310、slow consumer 32/32、正常終了を確認した。
RSSは開始2,219,130,880 bytes、peak 2,239,053,824 bytes、終了2,023,309,312 bytesだった。
ただし長prefixの102件がTTFT 10秒SLOを超え、SLO内は3060/3162だったため、総合判定は
**不合格**である。最終長prefix窓のTTFTは平均8026.096 ms、最大11894.346 msだった。
安定性の証拠としては利用できるが、30分SLO認定や標準serveへの昇格には利用しない。

[設定matrix](evaluation/gemma2-prefill-settings-m4-2026-09-27.json)では
`prompt_concurrency=2 / prefill_step_size=512`が短時間の最大TTFTを
7579.311 msから7363.242 msへ改善したが、同時長prompt 2件の5分試験はSLO内496/510、
`prompt_concurrency=1`でも500/510に留まった。そこで短文は並列度2を維持し、約2K-tokenの
長promptだけを並列度1へ制限するprofileを追加した。これはbackendのdecode concurrencyを
下げず、workload admissionだけを実機容量に合わせる。

```bash
.venv/bin/python scripts/qualify_gemma2_batch_mask.py \
  --python /opt/homebrew/opt/vllm-metal/libexec/bin/python \
  --model models/gemma-2-2b-it-4bit \
  --output docs/evaluation/gemma2-batch-mask-long-c1-prefill512-30min-m4-2026-09-27.json \
  --port 19118 --sustained-requests 100 --long-requests 12 \
  --duration-seconds 1800 --require-30-minute-window \
  --decode-concurrency 2 --prompt-concurrency 2 \
  --prefill-step-size 512 --long-concurrency 1
```

[再認定report](evaluation/gemma2-batch-mask-long-c1-prefill512-30min-m4-2026-09-27.json)は
1800.211秒で通常応答3924/3924件が正答かつSLO内、cancel 384/384、slow consumer 39/39、
切断後回復、SIGINT正常終了に合格した。開始時からthermalは全385 sampleで`fair`、
`serious`／`critical`は0件。RSSは2,216,345,600 bytesから2,241,167,360 bytesへ
24,821,760 bytes増加した。このprofileはM4 Air／32 GiB、Gemma 2 2B 4-bit、短文c2、
約2K-token長prompt c1に限定して30分認定する。長prompt c2や一般品質は認定しない。

障害注入の短時間gateも追加した。`X-VLLM-Apple-Request-ID`付きrequestはcontext生成前から
pending registryへ登録し、cancel済み項目をsource-hash確認済みのupstream `_next_request`
入口で破棄する。dequeueとの競合でcontext生成へ進んだ場合も、active化時に`stop()`して
同じ安全境界で除去する。`X-VLLM-Apple-Timeout-Ms`は1〜600,000 msに制限し、request IDを
必須として同じcancel経路を使う。

[queued cancel／timeout report](evaluation/gemma2-queued-cancel-timeout-m4-2026-09-27.json)は、
queued requestのDELETE 202と`request_state=queued`、model実行前のHTTP 404終了、active
blockerの停止、100 ms timeoutの122.589 msでのstream完了、後続回復、正常終了に合格した。

既存`BackendProcess`にはreview済みcompat moduleだけを`python -m`で管理できる限定起動契約を
追加した。[worker crash report](evaluation/gemma2-worker-restart-qualified-m4-2026-09-27.json)は
SIGKILL 3/3回を検出し、0.25／0.5／1.0秒backoff後に各2.596〜2.798秒で別PIDをreadyへ戻し、
毎回の算術品質と最終shutdownに合格した。これは明示的supervisor restartの証拠であり、daemon
watchdog、inflight request replay、sleep／wakeの認定ではない。

続いて`BackendSupervisor`を追加し、poll、最大restart回数、0.25秒からのbounded exponential
backoff、restart failure／exhaustion／PID／ready snapshot、shutdown競合防止を実装した。
[自動watchdog report](evaluation/gemma2-worker-watchdog-qualified-m4-2026-09-27.json)では、
手動`restart()`を呼ばずSIGKILL 3/3回を検出し、backoff込み2.953／3.418／3.710秒で別PIDを
readyへ復帰、restart failure 0、毎回の算術品質、最終shutdownに合格した。その後daemonの
起動・停止とnative v2 tuning／restore／quarantine rollbackを同じsupervisor transactionへ
接続した。planned restart中はwatchdogがprocessの再稼働を再確認し、二重restartを行わない。
restart中の新規requestはupstreamへ送らず`503 backend_unavailable`、接続後のworker消失も
同じretryable errorへ変換する。proxyはinflight requestを自動再送しない。

watchdogはrestart成功・失敗・上限到達をdaemon eventへ橋渡しする。各restart失敗では
raw logを保存せずdigestだけを含むprivate crash diagnosticを生成し、上限到達時は
`backend_exited`の構造化failureをserviceへ設定する。mock worker試験で2回のrestart失敗と
exhaustionを固定した。[追加M4 fault report](evaluation/gemma2-worker-watchdog-fault-qualified-m4-2026-09-28.json)
では、SIGKILL 3/3回のrestart中に新規clientが実際に`503 backend_unavailable`を受け、
3.096〜3.738秒で別PID・readiness・算術品質を回復し、正常終了した。

8時間認定に向け、同じrunnerへ`--require-8-hour-window`を追加した。このgateは28,800秒未満を
拒否し、active／queued cancel、timeout、slow consumerを試験中に反復する。各cycle後に
原子的checkpointを保存し、RSSは最大512点の時系列から後半の傾きを算出する。8時間gateでは
16 MiB/時以下かつ後半増加64 MiB以内をallocator plateauとする。`--require-sleep-wake`指定時は
suspend gapの観測ゼロを不合格にする。[45秒smoke](evaluation/gemma2-8hour-runner-smoke-m4-2026-09-28.json)
は通常応答75/75と4種類のfault各7/7に合格したが、8時間認定ではない。worker crashの同一窓
反復注入と実8時間runは未完了である。

## English

An explicit, version- and source-hash-bound launcher reshapes shared-head batched
Gemma 2 masks for grouped-query scores. It preserves upstream computation and never
modifies installed packages. M4 HTTP smoke passed 30/30 requests at concurrency 1
and 2; eight GPU cases matched unpatched per-row attention within 1e-5 tolerance.
A short M4 regression passed 12 shared-prefix edits, 100 sustained concurrent
requests, explicit active cancellation, a bounded slow consumer, and recovery after
a client-side disconnect. A 30-minute mixed run completed 3,162/3,162 correct
responses, 310/310 cancellations, and 32/32 slow consumers with clean shutdown and
no RSS growth. It failed qualification because 102 long-prefix requests exceeded
the 10-second TTFT SLO (3,060/3,162 within SLO). Managed serving is unchanged.

A workload-aware profile keeps short requests at concurrency 2 and admits the
approximately 2K-token long prompts at concurrency 1, with a 512-token prefill step.
Its 30-minute rerun passed 3,924/3,924 responses within quality and latency SLOs,
384/384 cancellations, 39/39 slow consumers, recovery, and clean shutdown. All 385
thermal samples were `fair`; none were `serious` or `critical`. Qualification is
limited to this M4 Air, model, and workload envelope.

The bounded fault gate now cancels requests while they are still queued and supports
a 1–600,000 ms request timeout through the same cancellation path. Real HTTP tests
passed queued cancellation, a 100 ms timeout, and subsequent recovery. Three injected
SIGKILL crashes were restored under the existing managed process lifecycle with new
PIDs and correct recovery responses. Automatic daemon watchdog and in-flight replay
remain unqualified.

`BackendSupervisor` now provides bounded polling, exponential backoff, restart
limits, lifecycle locking, and observable restart state. Without manual restart
calls, an M4 test recovered from three SIGKILL crashes in 2.95–3.71 seconds including
backoff, with zero restart failures and correct responses. Daemon startup, shutdown,
native-v2 tuning, restore, and rollback now share the same supervisor transaction.
New requests during restart receive `503 backend_unavailable`; transport failures are
not replayed automatically, avoiding duplicate in-flight execution.

Watchdog restart successes, failures, and exhaustion now flow into daemon events.
Every failed restart persists a private crash diagnostic containing only bounded log
metadata and a digest; exhaustion also sets a structured `backend_exited` runtime
failure. A mock worker test covers two failed attempts and exhaustion. The additional
M4 fault run observed `503 backend_unavailable` during all three injected crashes,
then recovered a new PID, readiness, and response quality in 3.096–3.738 seconds.

The soak runner now has an explicit 28,800-second gate, atomic per-cycle checkpoints,
repeated active/queued cancellation, timeout and slow-consumer checks, and a bounded
512-sample RSS series. The eight-hour gate requires the latter half to remain at or
below 16 MiB/hour and 64 MiB total growth. Optional sleep/wake qualification fails
when no suspend gap is observed. A 45-second M4 smoke passed 75/75 normal responses
and 7/7 of each fault check; it is not eight-hour evidence. Repeated worker crashes
inside the same window and the actual eight-hour run remain unfinished.

## 简体中文

专用启动入口仅对版本和源码哈希匹配的Gemma 2补齐批处理mask的分组维度，保留上游计算，
不修改已安装的软件包。M4上的并发度1和2各通过30/30次HTTP测试，8组GPU数值测试与未修改的
逐条attention在1e-5容差内一致。追加测试通过12次共享前缀编辑、100次并发持续请求，
并通过显式活动请求取消、有界慢速客户端及客户端中断后的恢复。30分钟混合负载完成了
3162/3162个正确响应、310/310次取消和32/32个慢速consumer，正常退出且RSS没有增长。
但有102个长前缀请求超过10秒TTFT SLO（SLO内为3060/3162），因此总体认证失败。
该结果仅作为稳定性证据，不用于升级标准serve。

新的工作负载profile保持短请求并发度2，只把约2K-token的长prompt限制为并发度1，并使用
512-token prefill step。30分钟复测中3924/3924个响应全部通过质量及延迟SLO，取消384/384、
慢速consumer 39/39，并通过恢复及正常退出。385个thermal sample全部为`fair`，没有
`serious`或`critical`。该认证仅适用于本机M4 Air、当前模型及此工作负载范围。

有界故障测试现在可以在请求仍处于queue时取消，并通过同一取消路径支持1–600,000 ms超时。
真实HTTP测试通过了queued取消、100 ms超时及后续恢复。现有managed process生命周期在3次
SIGKILL后都以新PID恢复，并返回正确答案。daemon自动watchdog和处理中请求重放仍未认证。

新增的`BackendSupervisor`提供有界轮询、指数backoff、重启次数上限、生命周期锁和状态快照。
无需手动调用restart，M4测试在三次SIGKILL后均于2.95–3.71秒内恢复，重启失败为0且回答正确。
daemon启动、停止、native-v2调优、恢复及rollback现在共用同一个supervisor transaction。
重启期间的新请求返回`503 backend_unavailable`；传输中断的请求不会被自动重放，以避免重复执行。

watchdog的重启成功、失败和次数耗尽现在都会发送到daemon事件。每次重启失败都会保存private
crash diagnostic，其中只包含有界日志元数据和摘要；次数耗尽时还会设置结构化的
`backend_exited` runtime failure。mock worker测试覆盖了两次重启失败及exhaustion。
新增M4故障测试在三次SIGKILL的重启期间都观测到`503 backend_unavailable`，随后在
3.096–3.738秒内恢复新PID、readiness和回答质量，并正常退出。

长时间runner现在提供明确的28,800秒gate、每个cycle的原子checkpoint、反复active／queued
取消、timeout、slow consumer，以及最多512点的RSS时间序列。8小时gate要求后半段趋势不超过
16 MiB/小时且总增长不超过64 MiB。启用sleep／wake认证时，如果没有观测到suspend gap则失败。
45秒M4 smoke通过了75/75个正常响应和每类7/7次故障检查，但不属于8小时认证。同一窗口内
反复注入worker crash及实际8小时运行仍未完成。
