# Bounded text HTTP benchmark

## 日本語

`python -m vllm_apple.text_benchmark`は、既に起動したloopback HTTP backendに同じ英語・日本語・简体中文の算術promptを送り、並列負荷と通信latencyを計測する。モデルの起動・download・更新は行わない。

```bash
.venv/bin/python -m vllm_apple.text_benchmark \
  --base-url http://127.0.0.1:19096 \
  --model /absolute/path/to/model --backend mlx_lm_direct \
  --hardware-fingerprint Apple-M4-32GiB \
  --requests 30 --concurrency 2 --max-tokens 16 \
  --ttft-slo-ms 1000 --e2e-slo-ms 5000 > /tmp/text-benchmark.json
```

認証が必要なら`--session-token-file`、RSSを観測するならbackendの`--target-pid`を指定する。token・endpoint・prompt・生成本文はreportに保存しない。モデル識別子は保存するので、共有するreportのローカルパスには注意する。

- 固定数のworkerだけを作るclosed-loop方式。requestは1〜100,000、並列度は1〜32かつrequest数以下。request数に比例するfutureや生成本文を保持しない。
- HTTP／usage／途中切断の失敗は件数とcodeを保存する。完了率の分母から除外しない。
- goodputは、前後の空白を除いた答えが`2`で、TTFTと`[DONE]`までの時間が指定SLO以内のrequestのusage output tokensを、負荷実行全体のwall timeで割る。並列requestの時間を加算して分母にしない。
- 言語別の試行・完了・品質・SLO合格件数を保存する。全件正答でCLI終了code 0、失敗／品質不合格があれば1。SLO未達はreportで別に判定する。
- workload hashはprompt・期待値・request数・並列度・sampling・timeout・SLOを固定する。endpointを含めないのでdirect／daemon間で照合できるが、同一artifactやbackend buildの証明にはならない。
- 検証済みのmodel integrity root digestとbackend環境digestがある場合だけ、`--artifact-identity-sha256`と`--backend-build-sha256`を同時指定できる。片方だけ、64桁小文字hex以外は拒否する。指定なしの既存reportはidentity未検証のままである。
- Python APIの`cases=`には最大64件の`(label, prompt, expected)`を渡せる。labelは重複不可、64 bytes以下。prompt 8 MiB、expected 1 KiBを上限とし、内容をreportへ保存せずworkload hashへ結合する。CLIは固定三言語workloadだけを実行する。
- p99はhistogram上限であり、測定完了数1,000未満は参考値。backend内部token timestamp、queue、tokenize、allocator、swap、thermal、energyは未取得と明記する。RSSを指定していない場合、phase profileの既存0値は実測ゼロではない。
- backend cacheは制御しない。cold／warm／prefix hitの認定や純粋なdecode速度比較ではない。HTTPタイムアウトはsocket待機の上限で、request全体のhard deadlineではない。

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

### M4で見つかった並列失敗

MLX 0.32.1／MLX-LM 0.32.0、Gemma 2 2B、serverのprompt／decode concurrencyを2に設定した。並列度1は30/30正答。並列度2の最初の試験は30秒timeoutが続いたためbackendを手動停止した。停止後の接続失敗を含み、比較baselineとして使用しない。

[短い独立再試験](evaluation/text-benchmark-m4-c2-repro-2026-09-26.json)は、3件warmup後に5秒timeout・並列度2・3件を実行し、1件成功、2件失敗。backendの`_generate`threadから`batch_generator.next()`、Gemma 2 attentionの`mx.where`へ進み、次の例外でthreadが終了した。

```text
ValueError: [broadcast_shapes] Shapes (2,1,1,20) and (2,4,2,1,20) cannot be broadcast.
```

これはこのmodel／build／負荷条件での失敗であり、全MLXモデルへの一般化はしない。既存の並列度1の認定を並列度2へ広げない。mask／batch／cache契約の調査、upstream修正候補との照合、修正後のcancel／回復と並列再試験を`[Next]`とする。Homebrew管理下のpackageは変更していない。実行した子processは両試験で回収済みで、SIGTERM終了を自然な正常終了の認定には数えない。

## English

Run a bounded closed-loop workload against an existing local HTTP backend. Reports
retain failures, per-language arithmetic quality, client timings, and quality-gated
SLO goodput using total wall time. Workload hashes enable input matching but do not
verify artifact identity. Cache state is uncontrolled; internal timings are unavailable.
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

## 简体中文

对已启动的本地HTTP后端运行有界闭环负载，保留失败、各语言算术质量、客户端延迟，
并按总墙钟时间计算满足质量及SLO条件的有效吞吐。工作负载哈希用于核对输入，
不证明模型文件相同。缓存状态未受控制，后端内部计时不可用。
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
