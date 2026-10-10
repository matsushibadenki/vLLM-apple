# P1 efficiency candidates — 2026-10-07

## Current status / 現在 / 当前 — 2026-10-10

🟢 [Done] [SPM表再利用とHTTP参照解放](P1-SPM-REUSE.md)を実装。修正後[90秒report](evaluation/p1-spm-final-m4-2026-10-10.json)は正常負荷420/420品質・SLO、RSS slope −19.84 MB/hour、allocator/thread/FD、awake、identity、正常停止が合格。warmup 3/3、初期長文12/12・短文30/30も合格。90秒は30分/8時間の認定ではない。
🟠 [Next] 2026-10-11 09:00 JST、compact/prefill512の単発30分→全gate合格時8時間。既存r11は不合格のまま、専用job回収済み。定期更新はユーザー指示で停止したまま。以下の過去の起動・定期予定は履歴であり現在の予約ではない。
English: The corrected pinned P1 passes a 90-second check (420/420 quality/SLO, RSS/resource/awake/identity/shutdown gates). Long-run qualification remains unfinished; one-off October 11 at 09:00 JST. Historical schedules below are superseded; recurring updates remain stopped.
简体中文：修正后固定版本P1通过90秒试验（420/420质量/SLO、RSS/资源/awake/identity/shutdown）。长期认证仍未完成，10月11日09:00 JST单次试验；下方旧日程仅为历史，定期更新仍停止。

## 日本語

[Done] sourceを固定したMLX-LM 0.32.0／MLX 0.32.1／M4 32 GiB／Gemma 2 2B 4-bit向けに、
明示選択する効率化候補を実装した。既定は`baseline`。自動採用・本番／性能認定は行わない。

| 設定 | schedulerが新しい要求を確認する処理時間の目安 | allocatorの解放済みcache上限 | idle時 |
| --- | --- | --- | --- |
| baseline | 500 ms | 256 MiB | 従来の100 ms間隔のqueue待ち |
| responsive | 50 ms | 256 MiB | 新要求または停止通知までqueueで待機 |
| compact | 50 ms | 64 MiB | responsiveと同じ |

50 msはGPU処理を中断するdeadlineではない。一つの長いprefill stepを途中で割り込ませる設定でもない。
新しいidle待機は単一processの空schedulerだけに適用し、処理中の非blocking readと分散経路は維持する。
shutdown時はqueueへ通知し、待機threadを起こす。allocator cacheは再利用用の解放済みbufferであり、
モデル本体やKV cacheとは別の予算。少なくすると再allocationの費用が増える可能性がある。
[MLX公式仕様](https://ml-explore.github.io/mlx/build/html/python/_autosummary/mlx.core.set_cache_limit.html)を参照し、実装の互換性は配置済みversionで検証した。

[Done] [各候補3回の比較](evaluation/efficiency-m4-2026-10-07/summary.json)：
起動順を変え、各回新しいworkerを使用した。concurrency 2の短文60件・長文12件、計648件の品質とSLOが成功。
9試験のcancel／故障系確認と正常終了も成功。runtime／runner／model identityは候補間で一致する。

| 設定 | 各runの短文平均E2Eの中央値 | run間の範囲 | 長文平均E2Eの中央値 |
| --- | --- | --- | --- |
| baseline | 905.305 ms | 729.176–1782.382 ms | 1390.620 ms |
| responsive | 526.159 ms | 511.033–1186.635 ms | 795.375 ms |
| compact | 722.034 ms | 582.950–1299.770 ms | 1396.821 ms |

responsiveの短文中央値は41.88%小さいが、測定範囲は重なり、変動はP3の5%基準を大幅に超える。
因果的な速度改善、TPOT p95、一般品質、全用途の最速を認定しない。起動warmupはこの648件に含めない。

[Done] [固定要求後の資源計測](evaluation/efficiency-m4-2026-10-07/resources.json)：各候補8件の品質確認後、
idle snapshotで保持cacheの最大値はbaseline 255.32 MiB、responsive 231.51 MiB、compact 63.77 MiB。
compactはこの観測で約75%少ない。active allocator最大値は約1878 MiBでほぼ同じ。
RSSはbaseline約745 MiB、他候補約2093 MiBと逆転しており、総使用メモリ削減の証拠にしない。
資源probeはprocess停止のみ確認し、正常shutdownの根拠は上記9試験を用いる。

[Done] [実schedulerのidle測定](evaluation/efficiency-m4-2026-10-07/idle.json)：3秒のread入口回数29→1、
process CPU時間3.156→0.752 ms。モデル推論なしの短時間測定であり、ワット値ではない。
最初の測定fixtureにはload_defaultがなくthreadが終了したため無効とし、
`idle-invalid-fixture.json`に保存。fixture修正後はthreadの生存と停止を確認した。
[pending] [電力計測](evaluation/efficiency-m4-2026-10-07/power-availability.json)：powermetricsは管理者権限を要求。
権限を持つ計測環境または外部電力計でjoule/request・idle wattを測るまで省電力率は未認定。

checkoutからpreviewを起動する場合は、明示的に選択できる。

```sh
/opt/homebrew/opt/vllm-metal/libexec/bin/python -m vllm_apple.local_text --model models/gemma-2-2b-it-4bit --efficiency compact --check --language ja
/opt/homebrew/opt/vllm-metal/libexec/bin/python -m vllm_apple.local_text --model models/gemma-2-2b-it-4bit --efficiency compact --language ja
```

`--efficiency responsive`で受付優先候補、`--efficiency baseline`で従来設定。
previewはconcurrency 1で、上記concurrency 2の性能数値をそのまま当てはめない。
この変更前に作成したwheelには新optionを含めていない。新候補はcheckoutで利用する。

再現コマンド（各outputは未作成のpathを使う）：

```sh
.venv/bin/python scripts/compare_p1_efficiency.py --python /opt/homebrew/opt/vllm-metal/libexec/bin/python --model models/gemma-2-2b-it-4bit --output-directory docs/evaluation/efficiency-new
python3 scripts/summarize_p1_efficiency.py docs/evaluation/efficiency-new
.venv/bin/python scripts/probe_p1_efficiency_resources.py --python /opt/homebrew/opt/vllm-metal/libexec/bin/python --model models/gemma-2-2b-it-4bit --output docs/evaluation/efficiency-resources-new.json
/opt/homebrew/opt/vllm-metal/libexec/bin/python scripts/measure_p1_idle.py docs/evaluation/efficiency-idle-new.json
```

[Done] 全Python回帰1454 tests成功（11 skip）、Ruff成功。compactの実hardware事前確認も成功。
[Next] [r10](evaluation/p1-stability-m4-2026-10-07-r10/state.json)でcompact／prefill512を固定して30分、
全条件合格時だけ8時間を実行する。実行中のruntime変更や追加GPU負荷を避ける。
r9は品質3009/3009、SLO3000/3009、cache増加87,461,321 bytesで資源gate未達。
RSSは合格したがP1は不合格、r9の8時間は未開始。r10へ自動的に資格を移さない。

## English

[Done] Added explicit `baseline`, `responsive` and `compact` candidates for the pinned M4/Gemma2 backend.
Responsive checks new work after a 50 ms scheduler budget and blocks on the queue while idle; shutdown wakes it.
Compact also reduces the free allocator cache budget from 256 to 64 MiB. Baseline remains the default.
This budget does not interrupt GPU work or reduce model/KV storage.

Three fresh workers per candidate, with rotated order, passed 648 short/long arithmetic quality and SLO checks plus
fault checks and clean exit. Median short-response run means were 905.305, 526.159 and 722.034 ms respectively.
Overlapping ranges and large variation prevent P3 speed qualification. TPOT p95 and general quality are unqualified.
Separate fixed-request snapshots observed up to 255.32/231.51/63.77 MiB of retained cache. Active allocation was
almost unchanged; RSS did not show a reduction. Idle scheduler read entries fell from 29 to 1 over three seconds.
These are not electrical-power measurements; powermetrics needs administrator privileges in this environment.

Use the checkout command above with `--language en` and an explicit `--efficiency` choice. Previously built wheels
lack this option. Preview concurrency is one; comparison performance at concurrency two does not transfer automatically.
[Next] R10 tests compact for 30 minutes, then eight hours only after every gate passes. Full regression: 1,454 tests,
11 skipped. [pending] Joules/request and idle watts require an authorized power sampler or external meter.

## 简体中文

[Done] 为固定M4／Gemma2后端加入显式选择的baseline、responsive及compact候选。
responsive将scheduler检查新请求的时间预算设为50 ms，空闲时等待queue通知，停止时唤醒。
compact同时把allocator空闲cache预算从256降至64 MiB。默认仍为baseline；预算不会中断GPU计算，
也不减少模型及KV存储。

每个候选以不同启动顺序运行3个新worker，共648个短／长算术请求通过质量及SLO检查，故障检查与正常退出也通过。
短请求各轮平均E2E的中位数分别为905.305／526.159／722.034 ms。范围重叠且波动较大，因此不认证P3加速。
TPOT p95及一般质量仍未认证。另行固定请求后的cache最大观测为255.32／231.51／63.77 MiB，
active allocator基本不变，RSS没有显示减少。3秒内idle read入口次数从29降至1，这不是功率测量。

从checkout使用上面的命令，选择`--language zh`及`--efficiency`；旧wheel尚无该option。
preview并发为1，不直接套用并发2的比较数字。[Next] r10以compact进行30分钟验证，仅全部通过后进入8小时。
全回归1454项通过（11项跳过）。[pending] joule/request及idle watt需授权的功率采样环境或外部仪表。

## Schedule update / 開始時刻の変更 / 开始时间调整

2026-10-07のユーザー指定で、開始済みr10を制御停止した。次回は10月8日09:00 JSTに新規runで開始し、
30分の全条件合格時だけ8時間へ進む。以降も長時間試験の一連の実行は日本時間の朝9時開始。
English: R10 was stopped on user request. The next new campaign starts October 8 at 09:00 JST; eight hours follow only if the 30-minute stage passes all gates.
简体中文：r10按用户要求停止。下一轮新试验在10月8日日本时间09:00开始，30分钟阶段全部通过后才进入8小时阶段。
