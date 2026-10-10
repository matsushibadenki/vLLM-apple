# P2 numerical parity and parallel response — 2026-10-10

## 日本語

🟢 [Done] 数値差をKVコピー破損と分離した。実Gemma 2／float16では、LRU再利用とfresh分割計算のlogit差は0だが、分割と全prompt一括計算は最大0.03125–0.046875違った。同じ4分割条件をfloat32で計算すると最大差0.000034–0.000042となり、既定`atol=rtol=1e-3`に合格した。許容値を緩めず、argmax一致だけで合格にしない。

P2専用の`--compute-dtype float32`を追加し、floating parametersとKVをfloat32にする。同じtoken数のKV要素の保存幅はfloat16の2倍であり、RSS／省電力の改善は未認定。packed integer weightsは変更しない。dtypeをcache namespaceに含め、異なる精度のKVを混在させない。実験用P2起動の既定はfloat32、scheduler budget 50 ms、SPM語彙表共有とimmutable RMSNorm補正重み共有。標準daemon／P1経路とinstalled packageは変更しない。scheduler budgetは長いGPU操作を途中で中断するdeadlineではない。

🟢 [Done] [最終KV試験](evaluation/p2-extended-kv-final-m4-2026-10-10.json)は16/16条件合格：prefix 19/127/511/513/1025 tokens、1/4-token suffix、owner不変、異なるprefix長をnative batch幅2/4へmerge。全prompt対照との最大差0.000119209、argmax一致、packed integer parameters保持を確認。profileはMLX 0.32.1／MLX-LM 0.32.0／M4 32 GiB／Gemma 2 2B 4-bitに限定する。1025 tokens超の数値正確性、SWA wraparound、他architectureを認定しない。

🟢 [Done] ユーザーが高負荷作業を停止した後の[同じfloat32精度の6 fresh-worker比較](evaluation/p2-float32-profile-comparison-m4-2026-10-10-quiet/comparison.json)。順序はbaseline/optimized/optimized/baseline/baseline/optimized、各worker c1 60件＋c4 60件、warmup 3件は別集計。baselineはcache無効・500 ms budget・SPM/RMSNorm共有なし、optimizedはcache有効・50 ms・両共有あり。各workerのsource/model identity不変、SIGINT停止exit 0を確認した。

| 観測 | float32 baseline (3 workers) | float32 optimized (3 workers) |
| --- | --- | --- |
| c4 品質 | 180/180 | 180/180 |
| c4 SLO (TTFT 1000 ms / E2E 5000 ms) | 154/180 | 180/180 |
| c4 goodput平均 | 12.440 tokens/s | 44.097 tokens/s |
| c4 平均E2Eのrun間範囲 | 1073.830–1154.895 ms | 331.106–421.238 ms |
| c4 TTFT p95のrun間範囲 | 1009.005–1095.926 ms | 202.741–263.712 ms |
| c1 TTFT p95のrun間範囲 | 243.522–260.016 ms | 50.829–57.345 ms |
| c1 TPOT p95のrun間範囲 | 13.357–14.702 ms | 13.836–15.056 ms |
| 最大実decode幅 | 各4 | 各4 |
| 実cache reuse tokens（warmupを含む123要求） | 各0 | 各2331 |

goodputはSLOを満たした正答token数÷全経過時間。baselineのSLO未達を分母から除外しない。改善はprofile全体の効果であり、cacheだけの速度改善率ではない。全720正常要求の品質は合格、optimized 360/360はSLOも合格。client p95は60要求のraw durationからnearest-rankで算出し、TPOTは要求ごとのclient平均token間隔。個々のGPU token latencyやcoarse histogram上限とは別の値。

🟢 [Done] ROADMAPの「prefixなしc1」を別の[3 fresh-worker試験](evaluation/p2-uncached-latency-m4-2026-10-10/report.json)で確認した。cacheを明示無効化し、各workerでcache entry／reuse tokenが0、品質・SLO 180/180、identity不変、exit 0を確認。TTFT p95は118.454–119.464 ms、TPOT p95は13.816–13.994 ms。最悪TPOT 13.994 msは最良baseline 13.357 msの5%以内（14.025 ms）。同精度c4 goodputは3.545倍で20%改善目標を満たす。[対象workloadの統合判定](evaluation/p2-optimization-validation-2026-10-10.json)はtrue。CPU回帰1,493 tests合格（31 skip）。これは事前のprefixなし条件を検証したもので、以下のcache有効c1の失敗や旧float16比較を合格へ書き換えない。

🟠 [Next] cache有効c1も含めた比較scriptのperformance gateはfalse。各3 workerの最悪optimized c1 TPOT p95 15.056 msを最良baseline 13.357 msに対して5%以内（14.025 ms）とする条件に未達。高負荷停止前の比較も不合格で保存し、停止後の結果でも判定を緩めない。旧float16 baselineとの比較もTPOTが悪化し、float16標準経路のreplacementを認定しない。同精度対照でprecision変更とbatching/KV効果を分離したが、旧失敗を上書きしない。静かな条件でも残る単独TPOT費用を改善してから採用判定する。このM4で試験可能なので[Next]であり[Pending]ではない。

測定中に別アプリの高CPU負荷を観測したが、これだけを遅延の根因と断定しない。MLP compilationとmixed量子化係数は実測で採用しなかった。共有候補の構築回数監査で、MLX Moduleが属性を`__dict__`ではなくmappingに保持することを検出し、`getattr`へ修正。関連CPU testは修正前fail／後pass、実workerでRMSNorm補正105個・SPM表1個の一度構築を確認。重みを変更するにはmodelをreloadする。RMSNorm共有単独の速度・ワット値改善は未認定。

P2全体、P1安定性、一般chat/coding品質、長時間fairness/cancel、他SoC／容量のqualificationは未完了。元のP1 10月11日09:00 JST単発予約とruntime manifestは維持する。定期更新は復活させない。

### 再現

GPU試験の競合がないことを確認し、各outputは新しいpathを使用する。

```sh
PYTHONPATH=. /opt/homebrew/opt/vllm-metal/libexec/bin/python \
  scripts/probe_p2_kv_extended.py --model models/gemma-2-2b-it-4bit \
  --output /tmp/p2-new-extended-kv.json
PYTHONPATH=. .venv/bin/python scripts/compare_p2_mlx.py \
  --python /opt/homebrew/opt/vllm-metal/libexec/bin/python \
  --model models/gemma-2-2b-it-4bit --baseline-compute-dtype float32 \
  --output-directory /tmp/p2-new-profile-comparison
PYTHONPATH=. .venv/bin/python scripts/probe_p2_uncached_latency.py \
  --comparison /tmp/p2-new-profile-comparison/comparison.json \
  --output-directory /tmp/p2-new-uncached-latency
```

旧精度対照は`--baseline-compute-dtype float16`で明示する。numerical/performance reportはqualification=falseのまま。P2 backendはopt-in。使用したMLX-LM MITの[license本文](licenses/MLX-LM-MIT.txt)を保持する。

## English

🟢 [Done] Float16 full-versus-segmented differences also occur without LRU reuse; fresh segmented and reused KV match exactly. A source-pinned float32 P2 profile resolves the fixed numerical cases without relaxing `atol=rtol=1e-3`. Sixteen real KV branch/mixed-prefix batch cases pass through 1025 tokens; maximum logit error is 0.000119209. Packed integer weights remain unchanged; dtype belongs to cache identity.

Six fresh-worker, same-float32 trials preserve quality for 720/720 normal requests. Concurrency-four SLO improves from 154/180 to 180/180 and mean goodput from 12.440 to 44.097 tokens/s. E2E run means improve from 1073.830–1154.895 to 331.106–421.238 ms. Actual decode width four and 2331 reused tokens per enabled worker are observed. This compares whole profiles, not cache-only speed. The optimized profile uses 50 ms admission checks and shared immutable SPM/RMSNorm data.

🟢 [Done] The ROADMAP single-request condition explicitly excludes prefix reuse. Three additional fresh, cache-disabled workers pass 180/180 quality/SLO with zero cache entries/reused tokens, unchanged identity and clean exit. TTFT p95 is 118.454–119.464 ms; TPOT p95 is 13.816–13.994 ms, within the unchanged 5% limit of 14.025 ms. Together with 3.545× same-float32 c4 goodput and the 16 numerical cases, the fixed-workload ROADMAP gate passes. CPU regression: 1493 tests, 31 skips.

🟠 [Next] The separate cache-enabled-c1 comparison gate still fails: c1 request-average client TPOT p95 ranges 13.357–14.702 ms at baseline versus 13.836–15.056 ms optimized. Earlier float16 comparisons also regress TPOT. Keep both failures; neither float16 replacement nor overall P2 is qualified. Other app load was observed but is not a proven sole cause. Production/P1 remain unchanged; no recurring updates are restored. The quiet rerun also fails the unchanged gate. Fix the remaining single-request cost before promotion.

## 简体中文

🟢 [Done] float16的全prompt与分段差异在不使用LRU时也存在；fresh分段与KV复用完全一致。固定source的float32 P2 profile在不放宽`atol=rtol=1e-3`的情况下通过16个真实KV分支及不同prefix长度的batch测试，覆盖至1025 tokens，最大logit差0.000119209。packed整数weight未改，dtype纳入cache identity。

同float32精度的6个fresh worker正常请求质量720/720通过。并发4的SLO从154/180升至180/180，goodput均值12.440→44.097 tokens/s；各run平均E2E范围1073.830–1154.895→331.106–421.238 ms。确认真实decode宽度4，每个启用cache的worker复用2331 tokens。比较的是整个profile，不是cache单独加速；采用50 ms admission检查及immutable SPM/RMSNorm数据共享。

🟢 [Done] ROADMAP的单请求条件明确不使用prefix复用。另行3个fresh、cache关闭的worker通过180/180质量及SLO，cache entry与复用token为0，identity不变且正常停止。TTFT p95为118.454–119.464 ms，TPOT p95为13.816–13.994 ms，符合不变的5%上限14.025 ms。结合同float32并发4 goodput提高3.545倍及16项数值测试，该固定workload的ROADMAP gate通过。CPU回归1493 tests通过，31 skip。

🟠 [Next] 单独的cache启用c1比较gate仍失败：单请求client平均TPOT的p95，baseline为13.357–14.702 ms，optimized为13.836–15.056 ms。旧float16对照也有TPOT退化，保留全部失败记录，不认证float16替换或整个P2。观测到其他应用负载，但不据此断言唯一根因。标准/P1未修改，不恢复定期更新；低负载重测也未通过原gate；修复单请求费用后再考虑晋升。
