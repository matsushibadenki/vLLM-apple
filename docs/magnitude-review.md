# Magnitude review — 2026-10-01 (Asia/Tokyo)

Reference: [Magnitude](https://github.com/magnitudedev/magnitude), commit
`72800340624c63d08c4b83e3e391a81471176453`. The latest release returned by the
official API was CLI `@magnitudedev/cli@0.2.3` (2026-10-01T02:25:53Z);
this is not an inference-engine compatibility version. The root
[license](https://github.com/magnitudedev/magnitude/blob/72800340624c63d08c4b83e3e391a81471176453/LICENSE)
is Apache-2.0. No upstream code or dependency was imported.

## 日本語

[native tuning実装](https://github.com/magnitudedev/magnitude/blob/72800340624c63d08c4b83e3e391a81471176453/inference-v4/engine/model-executor/src/native/tuning/mod.rs)
は準備時のbounded探索、device／toolchain／実装digest／探索定義によるcache key、
確認sampleの大きなばらつきの除外、defaultに対する改善marginを持つ。
[tile sizing資料](https://github.com/magnitudedev/magnitude/blob/72800340624c63d08c4b83e3e391a81471176453/info/inference-tile-sizing-autotuning.md)
は提案・未測定の研究noteであり、採用済み実装の証拠とは分けて読む。

[Done] 既存のevidence-only `tune_runtime_configuration`へ、prefill／decodeそれぞれの
`(max-min)/median`を指定上限と比較する任意のstability gateを追加した。
`maximum_relative_spread=0.05`なら5%超を除外する。`baseline_configuration`指定時は、
最速候補がbaselineに対して既定2%を超える改善を示さなければbaselineを維持する。
baseline自身がcorrectness／memory／stabilityに失敗した場合は選択を拒否する。
sample全体のdigestと選択policy・memory budgetをreport IDへ結合し、同じ中央値でも
異なる証拠のreportを混同しない。数値は本projectのpolicyであり、Magnitudeの値の移植ではない。

[Next] backend／model／shape別の独立確認測定とE2E回帰へ接続する。
現在の変更は選択ロジックのsynthetic検証であり、実LLMの速度改善は未測定。
runtime autotunerはevidence-only APIで、daemonの自動設定変更へは接続していない。

[Later] device・toolchain・kernel source・探索policy・shapeに束縛した結果cacheを、
既存Metal tuning profileと比較する。agent待機中のKV解放と共通prefix共有も検証候補。
Magnitude engineの直接導入は既存MLX state／KVとの互換契約と実測が必要。
upstreamの速度・memory節約の宣伝値を本projectの性能証拠には使用しない。

## English

[Done] Adopted noise rejection and conservative baseline retention as optional
policies in the existing evidence-only runtime tuner. Each phase is checked
independently; a baseline must pass all gates. Full sample evidence and selection
policy bind the report identity. No upstream implementation was copied.
[Next] Obtain independent backend/model/shape confirmation measurements and E2E
regressions. Actual inference gains remain unmeasured; daemon settings do not change.
[Later] Evaluate identity-bound selection caches, idle-session KV release, and shared
prefixes. Upstream benchmark claims are not local qualification evidence.

## 简体中文

[Done] 在现有evidence-only runtime tuner中加入可选的波动过滤及保守baseline保留策略。
prefill与decode分别检查；baseline必须通过全部gate。完整sample证据及选择策略绑定report ID。
没有复制上游实现。[Next] 获取各backend／model／shape的独立确认测量及E2E回归。
实际推理收益尚未测量，daemon配置不会自动改变。[Later] 评估identity绑定的结果缓存、
空闲会话KV释放和共享prefix。上游性能宣传不代表本项目的认证证据。
