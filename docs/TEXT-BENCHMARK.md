# Bounded text HTTP benchmark

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
