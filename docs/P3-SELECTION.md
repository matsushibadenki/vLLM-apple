# P3 measurement selection / 計測選択 / 测量选择

## 日本語

[Done] `vllm_apple.p3_selection`にkernel・量子化・speculative共通のE2E選択判定を実装。
`python3 scripts/select_p3_candidate.py evidence.json`で収集済みJSONを評価する。
実行設定の変更は行わない。既存の内部時間autotunerやspeculative profileの
`qualified`だけでは、このP3判定に合格したことにならない。

入力は`policy`、`baseline`、`candidates`。policyはscopeのSHA256、事前固定した
quality_slices、maximum_memory_bytesを持つ。scopeにはhardware・OS・model・backend
version／source hash・shape・precision・sampling・workloadを収集側で固定する。
各candidateはcandidate_id、kind、trialsを持つ。kindはbaseline／kernel／quantization／speculative。
各trialはacquisition_id、scope_sha256、policy_sha256、e2e_seconds、ttft_p95_seconds、
tpot_p95_seconds、peak_memory_bytes、quality（slice名とboolの組の配列）を持つ。
policy_sha256は`SelectionPolicy.digest`。閾値を結果取得後に変更しない。

独立取得の確認は収集者の`independent_acquisition_verified`で明示する。
IDの重複は拒否するが、異なるIDだけで独立取得を認定しない。baselineと候補それぞれ
3回以上、全品質slice合格、メモリ上限、run間変動5%以内を要求する。
E2E中央値5%以上改善し、候補の最遅runがbaselineの最速runより速いことも要求する。
TTFT／TPOTは候補の各runのp95がbaseline各runより5%を超えて悪化しない条件にする。
証拠が不足・失敗した候補は理由を残してbaselineに戻す。

P1／対象P2の検証済み前提は`prerequisites_verified`で別途指定する（既定false）。
候補が性能判定に合格しても前提未達ならstandard_adoption_eligibleはfalse。
automatic_applicationは常にfalse。report_idは全trialとpolicyに結び付く。

2026-10-04検証：新規7テスト、関連22テスト、全回帰1418テスト成功（11 skip）。
Ruff成功。sandbox内ではsocket bind制限で回帰失敗したため、ローカルsocketを許可して再実行した。

[Next] 実backendの計測収集へ接続し、対象候補の独立3回E2E、品質suite、
compile／warmup／scratch／内部phaseを取得する。本実装のテストはCPU上の判定検証であり、
kernel高速化や量子化品質、sampling分布保存を実測で証明したものではない。
P1の30分試験はSLO未達、P2も性能／数値一致未達。P3全体は未完了で標準採用しない。
[Later] [pending] 他SoC・ANE・互換draft／追加量子化artifactの評価には対象環境とartifactが必要。

## English

[Done] A common, scope-bound E2E selector evaluates kernel, quantization and speculative
candidate evidence. Run `python3 scripts/select_p3_candidate.py evidence.json`.
It requires at least three independently verified runs for both baseline and candidate,
all predefined quality slices, bounded memory, at most 5% E2E spread, at least 5%
median improvement, separated run ranges and at most 5% TTFT/TPOT p95 regression.
Every failed slice remains visible. Missing evidence retains the baseline.
Separate prerequisite verification controls standard adoption eligibility; configurations
are never applied automatically. Distinct acquisition IDs alone do not prove independence.
Internal autotuner scores and legacy speculative profile qualification do not replace this gate.

[Next] Connect real backend collectors and measure candidate quality and E2E performance.
CPU contract tests establish no speedup. P1 and P2 have outstanding failed gates, so P3
is not complete and standard adoption remains disabled.
[Later] [pending] Other SoCs, ANE and additional compatible artifacts require their own environments.

## 简体中文

[Done] 实现kernel、量化和推测解码共用的、限定适用范围的E2E选择判定。
运行`python3 scripts/select_p3_candidate.py evidence.json`。基线和候选各需至少3次
经确认的独立测量、所有预先固定的质量分项通过、内存不超预算、E2E波动不超过5%、
中位数改善至少5%，且候选最慢一轮快于基线最快一轮。TTFT／TPOT的p95退化不得超过5%。
失败分项不会被平均值隐藏，缺少证据时保留基线路径。不同测量ID本身不能证明独立性。
标准采用还需单独验证P1及相关P2前提，不会自动应用配置。
内部autotuner和旧speculative profile的qualified不能替代此判定。

[Next] 对接实际后端采集器，完成质量及E2E性能测量。CPU判定测试不证明性能改善。
P1、P2仍有未通过项目，因此P3整体未完成，暂不采用为标准路径。
[Later] [pending] 其他SoC、ANE及额外兼容模型文件需要对应环境与artifact。
