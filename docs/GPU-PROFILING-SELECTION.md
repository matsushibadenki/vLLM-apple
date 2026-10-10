# GPU profiling and measured setting selection — 2026-10-10

追加実装：🟢 [Done] [GPU core count・電源／thermal条件の照合とP4接続](HARDWARE-P4-CONNECTION.md)。以下はr3取得時点の実測履歴であり、新sourceの性能資格へ転用しない。
English: The hardware/conditions and P4 connection are now implemented; the r3 measurements below remain historical, not qualification for the changed sources.
简体中文：已实现hardware／运行条件及P4连接；以下r3测量保留为历史，不认证变更后的source。

## 日本語

🟢 [Done] `experiments.p3_mlx`で、実Gemma 2のGPU計測→候補範囲の選択→独立HTTP取得→既存P3 gate→実起動設定まで接続した。対象はM4 32 GiB／Gemma 2 2B 4-bit／float32／MLX 0.32.1／MLX-LM 0.32.0。標準daemonへ資格を付与せず、明示的な実験起動で適用する。

Metal System Trace＋Metal GPU Countersで、512-token prefillとbatch-one decodeを各12回収集。`time-info`のmach clockを使い、Pythonのmonotonic windowとtarget PIDのGPU区間を対応させる。depth-zero Active Compute区間のunionを集計し、入れ子区間や他processの仕事を二重計上しない。shader分類はShader Timelineのsampled durationであり、GPU wall timeと別の分母を使う。quantized matmulとfused dequantizationは分離できない。copy/indexing shaderの全時間をKV copyと解釈しない。CPU graph構築とevaluation/waitを分けるが、待機時間を全て同期の無駄と数えない。

GPUの支配的な量子化行列演算に応じて、今回の自動調整範囲はprefill step 512／128に制限した。候補の実測前にhardware／OS build／RAM／model全file／native binary・MLX-LM source／workload／sampling／precision／cache policy／P3 policyを固定する。通常要求のE2Eはprofilerを付けないfresh workerで測定し、profiler付きの時間を速度改善率へ流用しない。

独立したworkerはbaseline/candidate/candidate/baseline/baseline/candidateの順で、各c1/c4 30要求＋warmup 3要求を実行する。英語・日本語・简体中文の固定reference-prefix arithmeticを使い、cacheを無効にしてchunked prefillの違いをcache hitで隠さない。品質・SLO・warmup・identity・shutdown・実設定確認をquality slicesに含める。メモリは実MLX peak-active値であり、RSSや物理memory全体のpeakではない。native allocator上限8 GiB・cache上限256 MiBは既存profileと同じ。hostの独占は未確認で、変動gateを維持する。

`launch`は保存されたwinnerを信用せず、毎回P3判定を再計算し、現在のidentityに一致するallowlistの引数だけを実workerへ適用する。3-run E2E中央値5%以上・run範囲分離・変動5%以内・p95悪化5%以内・品質／SLO／memory条件の失敗はbaseline保持。scope変更・証拠改変・不明設定は起動を拒否する。通常の起動は明示的なopt-in、`--smoke`は同じcommandで実要求と設定値を確認して停止する。P1／P2の標準採用前提はfalseのまま。

### 実GPU結果

[最終GPU report](evaluation/p3-gpu-profile-m4-2026-10-10-r3.json)。各12 window、target PID限定、identity不変、native workload exit 0。

| 計測 | 512-token prefill | batch-one decode |
| --- | --- | --- |
| profiler付きwall中央値 | 1,222.209 ms | 21.001 ms |
| graph構築中央値 | 2.316 ms | 1.064 ms |
| GPU Active Compute union／wall（全window集計） | 98.916% | 84.540% |
| 量子化行列＋fused dequantのsampled shader時間割合 | 60.132% | 85.362% |

この固定shapeの主要な仕事はGPU上の量子化行列演算。decodeが帯域律速、prefillがcompute律速というroofline判定は未取得counterから推定しない。CPU→GPU submission latencyは前のGPU仕事のqueue待ちを含み、全量をCPU送信の無駄とは扱わない。graph、CPU、GPUは重なるため、各時間を足してE2Eを再構成しない。

### 実測選択と実起動

[6-worker evidence](evaluation/p3-real-selection-m4-2026-10-10-r3/evidence.json)、[選択結果](evaluation/p3-real-selection-m4-2026-10-10-r3/selection.json)、[実起動receipt](evaluation/p3-selected-launch-m4-2026-10-10.json)。全workerのidentity不変・exit 0、通常要求の品質360/360を確認。

| 観測 | baseline step 512 | candidate step 128 |
| --- | --- | --- |
| c4 30要求のelapsed seconds（各3 fresh workers） | 27.147 / 57.144 / 59.756 | 38.852 / 40.688 / 47.980 |
| c1＋c4 品質 | 180/180 | 180/180 |
| c1＋c4 SLO（TTFT 1000 ms / E2E 5000 ms） | 30/180 | 4/180 |
| MLX peak-active最大 | 3,274,946,992 bytes | 3,102,405,456 bytes |

候補のc4 elapsed中央値は28.8%低いが、両profileのrun間変動が5%超、範囲が重なり、SLO／warmup・p95条件にも未達。baseline自体も資格を満たさない。selectorは`baseline_retained=true`、candidate不採用、`standard_adoption_eligible=false`を返す。メモリ値や中央値だけでは採用しない。現状のfallbackは既存の実験設定を保持するもので、SLO認定済みのprofileへ昇格した意味ではない。host負荷・thermalを変動の根因とは未確認のまま扱う。

選択後の明示起動ではbaseline step 512、decode 4、prompt 2、cache無効を実workerの応答で確認。三言語9/9品質・SLO、identity不変、exit 0で停止した。候補128を新しい既定へ切り替えていない。

[最終接続監査](evaluation/p3-integration-validation-2026-10-10-r2.json)は合格。全回帰1,501 tests成功（31 skip）、関連15 testsとRuff・diff検査成功。P1の固定runtime／runner identityは一致し、10月11日09:00 JSTの単発長時間試験は維持する。

r3取得時点のhardware fingerprintはSoC／RAM／architecture／OS buildで、GPU core binや個体識別・thermal/powerは未収集。この筐体の診断に限定し、別のM4の性能を認定しない。新仕様ではGPU core countと電源／thermalを収集し、旧r3証拠を適用時に拒否する。

### 再現

他のGPU試験・長時間試験と重ねず、各output directory／receiptは新しいpathを使う。raw traceは大きいため、Git対象外の`qualification-results`へ保存する。

```sh
PYTHONPATH=. .venv/bin/python -m experiments.p3_mlx.profile \
  --python /opt/homebrew/opt/vllm-metal/libexec/bin/python \
  --model models/gemma-2-2b-it-4bit \
  --output-directory qualification-results/p3-new-gpu
PYTHONPATH=. .venv/bin/python -m experiments.p3_mlx.collect \
  --python /opt/homebrew/opt/vllm-metal/libexec/bin/python \
  --model models/gemma-2-2b-it-4bit \
  --gpu-profile qualification-results/p3-new-gpu/report.json \
  --output-directory qualification-results/p3-new-selection
PYTHONPATH=. .venv/bin/python -m experiments.p3_mlx.launch \
  --python /opt/homebrew/opt/vllm-metal/libexec/bin/python \
  --model models/gemma-2-2b-it-4bit \
  --evidence qualification-results/p3-new-selection/evidence.json \
  --receipt qualification-results/p3-new-launch.json --smoke
```

🟠 [Next] 帯域律速／compute律速を分けるhardware counterの定量評価、異なるshape・一般chat/coding/Agent品質、sampling、energy/token、標準daemonへの採用とP4全matrix／24時間認定。現在のM4で試せる未達は[Next]に留める。
⭕️ [Pending] 未保有SoC／独立Macの実機測定。速度改善や省電力を未取得counterから推定しない。

### 取得仕様と出典

2026-10-10 JST参照：[Apple Metal tools](https://developer.apple.com/metal/tools/)はCPU/GPU timelineの用途、[Apple GPU counters](https://developer.apple.com/videos/play/tech-talks/10001/)はGPU timestampとcounterの測定範囲を確認。[MLX Metal debugger](https://ml-explore.github.io/mlx/build/html/dev/metal_debugger.html)はnative GPU計測の補助資料。参照先docsの0.32.3表記をinstalled 0.32.1の資格へ流用せず、依存更新しない。nativeの実file hashesはreportへ記録する。MLX-LM由来部分は既存の[MIT license本文](licenses/MLX-LM-MIT.txt)を保持する。

初回のHTTP factory接続不具合とHomebrew alias照合失敗は履歴として保存。前者をnative handler classの段階で包むよう修正し、後者はidentityのfile keyだけを実体pathへ正規化した。起動に使うvenv executableのsymlinkはresolveしない。どちらも性能試験の合格扱いにはしない。

## English

🟢 [Done] An opt-in pipeline now connects real Metal profiling, a bounded candidate dimension, independent HTTP trials, the existing P3 gate and effective worker settings. Target: M4 32 GiB, Gemma 2 2B 4-bit, float32, MLX 0.32.1/MLX-LM 0.32.0. GPU intervals are filtered by target PID, intersected with monotonic workload windows and unioned without nested double counting. Shader Timeline sampled shares are distinct from GPU wall occupancy; fused dequantization is not separated from quantized matmul.

The observed matrix work guides prefill-step 512/128 trials. Policy, workload, source/model/native binaries and hardware identity are fixed before six serial fresh-worker trials. Quality, SLO, warmup, shutdown, identity and effective settings are explicit slices. Memory is actual MLX peak active, not RSS. Separate unprofiled HTTP acquisition avoids treating profiler overhead as an E2E improvement. The launcher recomputes the gate, rejects stale identity and applies only allowlisted settings; failures retain baseline. Application requires this explicit experimental launch and does not qualify production/P1/P4.

Actual six-worker acquisition passes 360/360 answer quality but fails SLO, spread and p95 gates. Candidate 128 is rejected despite a 28.8% lower c4 median; baseline 512 remains unqualified. The explicit selected launch confirms effective settings, 9/9 quality/SLO, unchanged identity and clean exit. Full regression: 1501 tests pass, 31 skips; 15 targeted tests and Ruff pass. The final integration audit passes and pinned P1 sources remain unchanged. No inference speedup is certified. Hardware identity does not yet distinguish GPU core bins or individual Macs.

🟠 [Next] Quantify bandwidth versus compute limits, broader workloads/shapes, energy and standard adoption. ⭕️ [Pending] Other unavailable hardware. See the commands and evidence above; no recurring updates are restored.

## 简体中文

🟢 [Done] 已连接真实Metal GPU测量、限定候选范围、独立HTTP试验、现有P3 gate及worker实际启动配置。范围为M4 32 GiB、Gemma 2 2B 4-bit、float32、MLX 0.32.1／MLX-LM 0.32.0。按目标PID过滤GPU事件，与monotonic workload窗口对应，取区间并集避免嵌套重复计算。Shader Timeline采样时间份额与GPU wall占用率分开；fused反量化不能与量化矩阵运算独立计时。

量化矩阵热点用于选择prefill step 512／128候选。先固定policy、workload、硬件及source／model／native binary identity，再依次运行6个fresh worker。质量、SLO、warmup、shutdown、identity及实际配置均为独立slice。内存指标是实际MLX peak-active，不是RSS。E2E由不启用profiler的独立HTTP试验获取。启动器重算gate、拒绝旧identity，仅应用allowlist参数；失败保留baseline。仅用于明确选择的实验启动，不认证标准daemon／P1／P4。

实际6-worker测量通过360/360回答质量，但SLO、波动与p95条件未通过。候选128虽有28.8%的c4中位数下降，仍拒绝采用；baseline 512也未认证。明确选择后的真实启动确认配置生效，9/9质量及SLO通过，identity不变且正常退出。完整回归1501 tests通过、31 skip；相关15 tests及Ruff通过，最终连接审计通过，P1固定source保持不变。不认证推理加速；hardware identity尚未区分GPU core bin或Mac个体。

🟠 [Next] 量化带宽／compute瓶颈、广泛workload与shape、energy及标准采用。⭕️ [Pending] 尚未拥有的其他硬件。复现命令与证据见上，不恢复定期更新。
