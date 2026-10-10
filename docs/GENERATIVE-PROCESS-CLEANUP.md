# Generative worker process cleanup — 2026-10-10

🟢 [Done] 生成workerの専用process groupを、leaderの正常終了後も回収する。TERM後にleaderを待機し、残存groupへKILLを送りreapする。存在しないgroupは許容するが、permission errorを無視して回収成功とは扱わない。

Problem: 終了済みleaderのpoll結果によりcleanupを省略し、孫processが残っていた。
Root cause: leader終了とprocess group終了を同一視していた。
Evidence / Before: 実workerがbackground childを起動して終了すると、childが後からfileへ書き込む再現testが失敗した（1 test、1.339秒）。
After: 同じtestを含む生成adapter／bounded trialの9 testsが3.753秒で成功。変更fileのRuff成功。全回帰1485 testsが38.534秒で成功（31 skip）。削除済みmodelを必要とする実機能力をこの結果で再認定しない。
Change: `vllm_apple/generative_subprocess_adapter.py`のfinallyで常に専用groupを回収。`tests/test_generative_subprocess_adapter.py`へ実childの残存を検出する回帰testを追加。
CPU / GPU / Memory / I/O / Energy impact: 不要なbackground childの残存を防止。推論速度、RSS、ワット値の改善率は未測定。追加GPU呼び出しなし。正常worker終了時にもgroupのsignal syscallを行う。
Correctness: 正常telemetry、非zero exit、invalid telemetry、timeout、line上限、trial回収を検証。既存timeout／品質基準は変更しない。
Regression risk: 専用group内のchildはworker終了時に終了する契約。独立sessionへ離脱したchildの回収はこのgroup方式では保証しない。
Keep / Revert: Keep。再現した不要processの残存を解消。過去のPermissionErrorや起動timeoutすべての原因が解消したとは扱わない。

English: 🟢 [Done] Always clean the worker's dedicated process group, including after a successful leader exit. A real-process regression reproduced a surviving background child before the fix; nine targeted tests pass afterward. Permission failures remain visible. Inference speed, RSS and power savings are unmeasured; this does not qualify P1 stability.

简体中文：🟢 [Done] 在worker正常退出后仍回收专用进程组。修复前实际子进程残留测试失败，修复后9项相关测试通过。权限错误仍明确报告。推理速度、RSS及功耗改善未测量；不代表P1稳定性认证。
