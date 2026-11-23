# Stable dependency review — 2026-11-23

## 日本語

⭕️ [Pending] Strataの静穏比較と競合するため、今回の依存更新・互換性検証を次回へ延期。
PID34953で `scripts/compare_strata_experts.py --quiet-host --quiet-timeout 180 --cycles 512 --warmup-cycles 8 --runs 4 --memory-suite --resident-bytes 20000` が稼働。出力予定は `docs/evaluation/strata-phased-partial-2026-11-23-085307-jst.json`。初回にworker PID35307も確認、再確認時もparent PID34953は稼働していた。process停止・設定変更は行っていない。

🟢 [Done] 依存定義・lock、CI冒頭、既存差分、[前回記録](dependency-update-2026-11-16.md)と公式レジストリを読み取り確認。project直下および直接の親に追加の適用対象AGENTS.mdは見つからず、チャットの指示を適用。P1 r11はstatus scriptでfailed／runner_alive=false。これは過去の未認定結果であり、現在のStrata比較の合否を表さない。

Ruffは定義0.17.5に対して最新stable0.17.6を確認。setuptools84.2.1、PyInstaller6.22.3、MLX0.32.4、MLX-LM0.32.2、MLX-Gen0.38.0、tokenizers0.23.3、Torch2.15.0、torchao0.18.0、NumPy2.5.5、Core ML tools9.0、Diffusers0.43.0、mflux0.24.0、Transformers5.24.0が公式PyPIのstable候補。最新Transformers／Torch／Diffusers／mfluxはPython>=3.11、NumPyは>=3.12。全体のPython>=3.10対応を単純な一括更新で破らない。

GitHub公式releaseの最新はMetal0.33.0、checkout7.0.1、setup-python7.1.0、upload-artifact7.0.2で、いずれもprerelease=false。情報源は公式 `https://pypi.org/pypi/<package>/json` と `https://api.github.com/repos/<owner>/<repo>/releases/latest`。これは候補確認のみで、互換性検証や採用ではない。

測定への干渉を避けてinstall・lint・compile・unit回帰・Swift build・packaging・doctor・モデル実行は開始していない。依存定義・lock・既存評価資料・Homebrew・backend venv・既定設定を保持。Swift／vendor／間接依存の詳細再棚卸しも今回は延期した。前回記録のTorch要求不一致、固定commit／Transformers上限制約、生成model不足、remote CI保留を解決済みとは扱わず、現在の実環境を新たに認定していない。ROADMAPの実装完了項目も追加しない。

🟠 [Next] 次回に実プロセスとcheckpointを再確認し、競合がなければRuff0.17.6を隔離検証してproject定義とdev lockを同期する。推論系は環境別の上流制約と実モデルqualificationを満たしたものだけ採用する。

## English

Dependency changes and validation were deferred because an active Strata comparison requires a quiet host (parent PID34953, confirmed twice). Ruff0.17.6 and other stable candidates were checked using official metadata only. No dependencies, environments or existing evidence were changed; no builds, tests or model workloads were started. Recheck contention before validating candidates on the next run. Previous compatibility concerns remain unverified and unresolved.

## 简体中文

Strata低负载对比试验正在运行，已两次确认parent PID34953，因此推迟依赖更新和验证。仅通过官方metadata检查了Ruff0.17.6等stable候选版本。没有修改依赖、环境或原有证据，没有启动构建、测试及模型负载。下次先检查是否仍有冲突，再隔离验证候选版本；此前兼容性问题未重新验证或解决。
