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
- p99はhistogram上限であり、測定完了数1,000未満は参考値。backend内部token timestamp、queue、tokenize、allocator、swap、thermal、energyは未取得と明記する。RSSを指定していない場合、phase profileの既存0値は実測ゼロではない。
- backend cacheは制御しない。cold／warm／prefix hitの認定や純粋なdecode速度比較ではない。HTTPタイムアウトはsocket待機の上限で、request全体のhard deadlineではない。

M4 smokeの起動条件は[実測report](evaluation/text-benchmark-m4-smoke-2026-09-26.json)の`command`に保存した。Homebrewの既存MLX-LM serverでGemma 2 2Bを起動し、3件warmup後、並列度1と2で各30件実行する。試験後は起動したbackendだけを終了・回収する。これは配線と三言語算術の確認であり、一般品質、continuous batching、性能優位、長時間安定性を認定しない。

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

## 简体中文

对已启动的本地HTTP后端运行有界闭环负载，保留失败、各语言算术质量、客户端延迟，
并按总墙钟时间计算满足质量及SLO条件的有效吞吐。工作负载哈希用于核对输入，
不证明模型文件相同。缓存状态未受控制，后端内部计时不可用。
此冒烟测试不代表一般质量、连续批处理或性能优势认证。
