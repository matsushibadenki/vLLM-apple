# Synthetic RAG quality suite

## 日本語

[Done] `scripts/rag_quality_suite.py`は英語・日本語・简体中文の12ケースを生成する。
各言語で正しい資料、回答に必要な情報を含まない資料、悪意ある追加資料、長い資料中の事実を評価する。
資料なし時のローカル回答不能とは別に、情報不足ケースにも資料を渡し、実生成による回答不能を要求する。
固定support codeの完全一致、期待する参照、根拠資料の保持、token予算確認、生成実行を別項目で採点する。
追加の事実、誤ったコード、未知参照、根拠資料の脱落、欠測ケースは合格にしない。

これは固定タスクの厳密な採点であり、自由記述の意味的groundingや一般的なprompt injection耐性を認定しない。
悪意ある追加資料は一種類のみ。runtime identityもこの採点だけでは検証しない。
[Next] P1試験終了後に固定model／buildで各payloadを`answer_rag`へ渡し、実モデル結果を収集する。
長い資料がcontext予算で除外された場合も、根拠が残っていなければ当該ケースは失敗とする。
[Later] 攻撃種類、長文中の位置、引用の意味的整合性、baseline独立反復の評価を広げる。

```sh
python3 scripts/rag_quality_suite.py --output /tmp/rag-cases.json
python3 scripts/rag_quality_suite.py --results /tmp/rag-results.json --output /tmp/rag-score.json
```

結果入力は`[{"id":"en-supported","result":<answer_ragの返却dict>}, ...]`。
既存outputの上書きは拒否する。12ケースが揃わない場合は失敗。suite hashをreportに保存する。
CPU採点・既存RAG HTTP回帰20テスト成功。模擬結果の成功は実モデルの成功とは扱わない。

## English

[Done] Twelve synthetic cases cover supported answers, nonempty insufficient sources,
an injected extra source and a fact inside a long source in English, Japanese and Simplified Chinese.
Scoring separately checks exact task answers, references, retained supporting documents, verified
token budgets and actual generation. Missing cases, wrong codes and extra claims fail.
The commands above export payloads and score a list of `{id, result}` using `answer_rag` results.
Existing output files cannot be overwritten; the report includes a suite hash.
Twenty CPU/scoring and existing mock HTTP tests pass. This does not certify model quality,
general semantic grounding, broad injection resistance or runtime identity.
[Next] Collect real model results after the active P1 test. [Later] Expand attacks,
fact positions, semantic citation assessment and independent baseline repetitions.

## 简体中文

[Done] 12个合成案例覆盖英语、日语和简体中文的正确资料、非空但信息不足的资料、
恶意追加资料以及长资料中的事实。分别检查固定答案、引用、根拠资料保留、token预算及实际生成。
遗漏案例、错误代码、额外事实、根拠资料被删除均判失败。上述命令可导出payload并为
`answer_rag`返回的`{id, result}`列表评分；拒绝覆盖已有文件，并记录suite hash。
20项CPU评分及既有模拟HTTP回归通过；不能据此认定实际模型质量、一般语义grounding、
广泛的注入抵抗能力或runtime identity。
[Next] 等P1试验结束后收集实际模型结果。[Later] 扩展攻击类型、事实位置、引用语义评估及baseline独立重复。

## 2026-10-05 actual HTTP result / 実測結果 / 实测结果

[Done] `probe_rag_http.py --quality-suite`を実HTTP collectorへ接続。
[実測report](evaluation/rag-quality-m4-2026-10-05.json)：M4のGemma 2 2B 4-bitで
12ケースを実行、backend正常終了、errorなし。厳密合格4/12。
EN長文・JA通常／injection・ZH通常／injectionは指定回答形式を逸脱。
JA／ZH長文は引用欠落、ZH情報不足は営業時間を回答し、回答不能に失敗。
全ケースでtoken予算確認・根拠資料保持・injection marker不在。ただし一般耐性の証明ではない。
[Next] prompt／資料構成の候補を固定baselineと比較し、形式、引用欠落、情報不足の誤答を改善する。
閾値を緩和しない。この12ケースの失敗は本筐体で修正・再測定可能であり[pending]にしない。

English: The actual HTTP collector ran all twelve cases with clean backend exit and no collector
error. Strict score: 4/12. Five cases violated the requested answer format; Japanese/Chinese long
sources lacked citations, and Chinese insufficient-source answering returned office hours instead
of abstaining. Token budgets and retained support passed. General grounding remains unqualified.
[Next] Compare prompt/source candidates against this fixed baseline without relaxing scoring.

简体中文：实际HTTP collector完成12例，backend正常退出，无collector错误，严格评分4/12。
5例未遵守回答格式；日语／中文长资料缺少引用，中文信息不足案例回答营业时间而未拒答。
所有案例均确认token预算及根拠保留，但一般grounding仍未认证。
[Next] 在不放宽评分的情况下比较prompt／资料构成候选与固定baseline。
