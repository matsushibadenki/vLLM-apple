# Bounded trial cleanup — 2026-10-09

## 日本語

🟢 [Done] `compare_p1_prefill.py`の試験processを独立session／process groupで起動する。timeoutまたは例外時、SIGINTで既存qualifierのfinally（report保存・supervisor停止）を最大45秒待ち、同じgroupに残ったprocessをSIGKILLで回収する。runnerをreapする。正常終了でもgroup内の残存子processを回収。独自sessionへ離脱した外部processまで回収する契約には広げない。

timeoutは合格にせず、専用`.timeout.json`をexclusive作成して後続試験を中止する。raw report／checkpointを変更しない。180秒＋停止猶予45秒＋reap最大5秒は試験の終了上限で、推論deadlineではない。既存MLX workerは親のgroupを継承する。無関係なprocess名で検索してkillしない。

Problem: 前回の一時的な比較driverはsubprocess.run(timeout=180)でrunnerを強制終了し、最終reportを保存できなかった。旧prefill比較scriptにもSIGINT停止猶予超過時の回収が不足していた。
Root cause: timeoutと停止を無視するprocessへの段階的な回収が不足。
Evidence: 前回の64 MiB timeout監査と今回の実process tests。
Changed files: trial_process.py、compare_p1_prefill.py、test_trial_process.py。
Change: owned group、SIGINT猶予、残存process回収、timeout別receipt、後続停止。
Before: 45秒猶予超過でrunnerを残す可能性。After: 残存group停止とrunner reap。
CPU impact: 正常時の終了waitのみ。速度改善率未測定。
GPU impact: GPU／kernel／同期変更なし。GPU実機のtimeout再試験は実施していない。
Memory impact: timeoutで残存processが資源を保持し続ける可能性を防ぐ。推論RSS削減未認定。
I/O impact: timeout receipt追加、既存raw証跡保護。
Energy impact: 未測定。
Correctness verification: 実processでexit code保持、SIGINT handlerの最終証跡保存、SIGINT無視processの強制終了・reapを検証。比較summaryの非昇格testも通過。
Regression risk: groupの範囲に限定。45秒で正常停止できなければ強制停止として扱い、正常shutdownやP1資格を付与しない。
Keep / Revert: 回収修正をKeep。既存の未完了試験を遡って認定しない。

🟠 [Next] 今後の短時間比較はこのbounded runnerを利用し、実KV・allocation候補を比較する。長時間campaignは09:00 JST開始、定期更新は停止したまま。

## English

🟢 [Done] Prefill comparison trials now run in owned process groups. On timeout/exception, allow up to 45 seconds for SIGINT-driven report preservation and supervisor shutdown, then kill remaining group members and reap the runner. Timeout writes a separate exclusive receipt, preserves raw evidence and stops subsequent trials. Real CPU-only process tests verify exit codes, graceful final evidence and cleanup of a runner that ignores SIGINT. This covers inherited groups, not processes that detach into separate sessions. No GPU timeout rerun, speed, energy or historical qualification claim. 🟠 [Next] Use this bounded runner for future short comparisons. Long campaigns start at 09:00 JST; recurring updates remain stopped.

## 简体中文

🟢 [Done] prefill比较试验使用独立、专属process group。timeout／异常时先SIGINT等待最多45秒，允许保存report及停止supervisor，再强制回收group残存process并reap runner。timeout另存exclusive receipt，保留raw证据并停止后续试验。CPU-only实际process tests验证exit code、正常停止的最终证据及忽略SIGINT的runner回收。范围仅为继承group的process，不包括自行脱离session的process。本次未重做GPU timeout，不认证速度、电能或历史试验。🟠 [Next] 短时间比较使用bounded runner；长时间测试09:00 JST，定期更新保持停止。

検証 / Validation / 验证：全回帰1486 tests（11 skip）、Ruff・diff check成功。[Receipt](evaluation/p1-trial-cleanup-validation-2026-10-09.json).
