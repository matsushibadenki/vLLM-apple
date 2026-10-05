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

## Candidate audit / 候補監査 / 候选审计

2026-10-05：[r2](evaluation/rag-quality-m4-2026-10-05-r2.json)は8/12、
[r3](evaluation/rag-quality-m4-2026-10-05-r3.json)は7/12、両方正常終了。
ただしbaselineで成功していた回答不能がr2では英語、r3では日本語で悪化したため、
両候補を不採用とし、runtime promptは元に戻した。採点基準は変更していない。
[Done] `compare_reports`で同一suite・全ケース・正常終了を要求し、総合点が増えても
既存合格ケースの悪化を拒否する。[比較証拠](evaluation/rag-quality-candidate-comparison-2026-10-05.json)。
[Next] 別のprompt／資料構成を検証する。今回の2候補から性能や一般品質改善は主張しない。

English: Candidates scored 8/12 and 7/12 but regressed English and Japanese abstention,
respectively. Both were rejected and the runtime prompt restored. The comparison gate requires
matching suites, complete cases, clean termination, and no loss of baseline passes, even when the
total rises. General quality and performance remain unqualified.

简体中文：候选得分8/12及7/12，但分别使英语和日语的拒答退化。两者均未采用，
runtime prompt已恢复。比较gate要求suite一致、案例完整、正常退出，并拒绝既有成功案例退化，
即使总分提高也不接受。一般质量及性能仍未认证。

## Adopted limited improvement / 限定改善の採用 / 采用有限改进

2026-10-05：[r4](evaluation/rag-quality-m4-2026-10-05-r4.json)と
[独立process再試験](evaluation/rag-quality-m4-2026-10-05-r4-repeat.json)は両方5/12。
baselineの合格4ケースは維持、中国語long_sourceが新たに合格。
[Done] system roleを使わない経路で、資料の後に元のinstructionとquestionを再提示する。
末尾の条件も完全chat templateのtoken計測へ含め、context不足時の資料除外でも保持する。
合成固定taskでの改善であり一般grounding／広範な攻撃耐性／性能改善ではない。
[Next] 残る7ケース（形式、情報不足時の誤答）の改善と、固定task以外の評価を進める。
追加prompt分のcontextを消費する。runtime sourceが変わったため旧RAG／P1のidentity証拠を
新buildの認定へ転用しない。P1は元から未認定で、次回試験でidentityを取り直す。

English: Two fresh backend processes both scored 5/12, preserving all four baseline passes and
adding Chinese long-source success. The no-system-role path now repeats the original instruction
and question after source data. Complete-template token budgeting includes the added text and
preserves it when dropping whole sources. This consumes extra context; it is a limited synthetic
task improvement, not general grounding or performance certification. Seven cases remain [Next].
Old runtime-identity evidence cannot certify the changed build.

简体中文：两个独立backend进程均为5/12，保留baseline全部4个成功案例，新增中文长资料成功。
无system role路径在资料后重复原instruction及question，新增文本计入完整template的token预算，
删除资料时仍保留这些条件。这会消耗额外context，仅证明合成任务的有限改善，不代表一般grounding
或性能认证。剩余7例为[Next]。旧runtime identity证据不能认证新的build。

Validation / 検証 / 验证：全回帰1441 tests成功（11 skipped）、Ruff・diff check成功。

## 2026-10-05 continued audit / 継続監査 / 后续审计

[r5](evaluation/rag-quality-m4-2026-10-05-r5.json)は6/12だがEN／JA回答不能が悪化し不採用。
[r6](evaluation/rag-quality-m4-2026-10-05-r6.json)は既存5件を維持して6/12、正常終了。
ただし[独立再試験](evaluation/rag-quality-m4-2026-10-05-r6-repeat.json)は回答6/12を再現しても
SIGINT後15秒以内に終了せずkill、returncode=-9。候補は採用せず、r4 promptへ復元した。
[Done] collectorはshutdown.graceful／forced_kill／sigint_deadline_secondsを保存し、
生成時のerrorなしと終了失敗を分離する。模擬deadline超過の回帰を含む7テスト・Ruff成功。
[Next] 実backendの終了遅延原因と、既存回答不能ケースを維持する品質改善を調査する。
P1／一般grounding／性能は未認定で、これらの失敗は[pending]にしない。

English: R5 regressed abstention and was rejected. R6 scored 6/12 without losing baseline
passes; its independent repeat reproduced those answers but exceeded the 15-second SIGINT
shutdown deadline and was killed (exit -9). R6 was not adopted; the r4 prompt was restored.
The collector now reports graceful/forced shutdown separately from generation errors, with a
mock deadline regression test. Shutdown diagnosis and quality improvement remain [Next].

简体中文：r5的拒答退化，未采用。r6在保留baseline成功案例的情况下达到6/12，
但独立重试虽重复相同答案，SIGINT后15秒仍未退出，被强制结束（-9）。未采用r6，
已恢复r4 prompt。collector现在分别记录正常／强制退出及生成错误，并加入模拟deadline回归。
退出延迟原因及质量改善仍为[Next]，不归为[pending]。

## Shutdown diagnosis / 終了診断 / 退出诊断

[Done] `mlx-server --diagnostic-signal`はSIGUSR1で全Python threadのstackをstderrへ保存する。
診断は明示opt-in、locals／request本文／tensorを採取しない。stackにはfile pathを含む。
collectorの`--shutdown-diagnostics`はこのoptionを接続し、SIGINT後15秒を超えた場合のみ
SIGUSR1を送り、0.5秒の記録猶予後に既存のkillへ進む。stack_dump_requestedはsignal送信の
記録であり、stack保存成功や原因特定の証明ではない。signal対応のPOSIX環境に限定。

[実機smoke](evaluation/rag-shutdown-diagnostics-m4-2026-10-05.json)は既存6ケース・正常終了合格。
今回の終了遅延は未再現。CPU実process試験でsignal後もworkerが生存しstackを保存することを
確認し、模擬終了deadline試験でSIGUSR1要求とkillの記録を確認した。
[Next] 長いsuiteで遅延を再現した際にstackを監査し、join待ち／GPU待ちなどの原因を特定する。
今回の診断実装だけで原因解消やR0／P1認定とは扱わない。

```sh
/opt/homebrew/opt/vllm-metal/libexec/bin/python scripts/probe_rag_http.py \
  --model models/gemma-2-2b-it-4bit --quality-suite --shutdown-diagnostics \
  --output /tmp/rag-diagnostic-new.json
```

English: Opt-in `--diagnostic-signal` records Python thread stacks on SIGUSR1, without
locals, request bodies or tensors; file paths are included. The collector's
`--shutdown-diagnostics` requests a dump only after the 15-second SIGINT deadline, then allows
0.5 seconds before its existing kill fallback. A requested dump does not prove successful capture.
The real six-case smoke shut down cleanly; the previous delay was not reproduced. CPU process
and mock timeout tests verify signal survival and diagnostic requests. Root cause remains [Next].

简体中文：显式`--diagnostic-signal`在SIGUSR1时记录Python thread stack，不记录locals、
请求正文或tensor，但包含file path。collector的`--shutdown-diagnostics`仅在SIGINT超过15秒
时请求stack，等待0.5秒后执行既有kill回退。请求记录不能证明stack已保存或原因已确定。
实机6例正常退出，未重复之前的延迟。CPU进程及模拟timeout验证signal不终止worker及诊断请求。
根因调查仍为[Next]，未认证R0／P1。

Validation / 検証 / 验证：全回帰1443 tests成功（11 skipped）、Ruff・diff check成功。
