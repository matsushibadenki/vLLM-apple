# Stable dependency review — 2026-10-19

> 朝の保留記録です。ユーザー指示後の適用・検証結果は末尾の「ユーザー指示による再開」を参照。

## 日本語

🟢 [Done] 公式PyPI metadataとRuffの公式releaseを読み取り確認。
依存定義・lock・バックエンドvenvは今回は変更していない。
前回記録は[2026-10-12](dependency-update-2026-10-12.md)。
既存のStrata関連の未コミット変更・評価資料・ROADMAPを保持した。
リポジトリ内のAGENTS.md検索では追加ファイルが見つからず、チャットの指示を適用した。

### 競合確認

2026-10-19 08:29頃 JSTのprocess確認で、PID21146の
`scripts/compare_strata_experts.py --quiet-host --quiet-timeout 180 --cycles 512 --warmup-cycles 8 --runs 4 --memory-suite --resident-bytes 20000`
が稼働していた。出力予定は`docs/evaluation/strata-phased-partial-2026-10-19-081829-jst.json`。
再確認時も同じprocessが稼働。測定への干渉を避け、install・lint・build・回帰・GPU試験を開始していない。
P1 r11はstatus scriptでfailed / runner_alive=false。これは過去の試験結果であり依存更新の結果ではない。

⭕️ [Pending] 実行中のStrata比較と競合するため、更新・互換性試験は一時保留。
🟠 [Next] 比較process終了と新規runner不在を確認後、Ruffのpatch更新を隔離環境でlint検証し、pyprojectとdev lockを同時更新する。
🟠 [Next] 推論系は環境ごとの版制約・既存モデル・数値・品質・メモリを検証して採用する。最新版確認だけでは採用しない。

### 確認したstable候補

| Library | 既存定義／確認した環境 | 最新stable | 判断 |
| --- | --- | --- | --- |
| setuptools | 84.0.0 | 84.0.0 | 定義は最新 |
| Ruff | 0.17.0 | 0.17.1 | patch候補。Python>=3.7。py310・明示rule維持 |
| PyInstaller | 6.22.3 | 6.22.3 | 定義は最新 |
| MLX | MLX-Gen環境0.31.2 | 0.32.4 | 未採用、backend ABIを先に照合 |
| MLX-LM | 前回Homebrew確認0.32.0 | 0.32.2 | 未採用、最新版はPython>=3.11。今回はHomebrew installed版を再認定していない |
| MLX-Gen | 0.33.1 | 0.38.0 | 品質・メモリ試験待ち |
| PyTorch | 2.14.0（生成環境） | 2.14.1 | 生成backendとの組合せ試験待ち |
| Core ML tools | 8.1 | 9.0 | 変換artifact・数値試験待ち |
| Diffusers | 0.41.0.dev0 | 0.41.0 | 既存dev版のstable移行を優先比較 |
| mflux | 確認した上記venvには未導入 | 0.23.0 | 利用環境とadapterを確認してから判断 |
| NumPy | Core ML1.26.4、MLX-Gen2.2.6、Qwen Image2.5.3 | 2.5.4 | 最新版はPython>=3.12。環境ごとに検証 |

今回、Swift・CI Actionsの更新検証は実施していない。
ランタイムdependencies=[]と環境分離、Homebrew管理環境を維持した。
既存のvLLM-MetalによるMLX／MLX-LM厳密固定は前回記録を引き継ぎ、上位制約を破らない。

公式根拠: [Ruff0.17.1 release](https://github.com/astral-sh/ruff/releases/tag/0.17.1)。
Ruffのpatchにはdedent等の修正があり、既存設定でのlint確認を行ってから採用する。
各版・Python要件は公式PyPI JSON `https://pypi.org/pypi/<package>/json` を確認した。
対象はsetuptools、ruff、pyinstaller、mlx、mlx-lm、mlx-gen、torch、coremltools、diffusers、mflux、numpy。
現時点ではいずれの候補も今回の互換性検証済みとは扱わない。

## English

Stable candidates were checked against official PyPI metadata and Ruff release notes.
Ruff0.17.1, MLX0.32.4 and MLX-LM0.32.2 are newer candidates; setuptools84.0.0 and PyInstaller6.22.3 remain current.
An active Strata memory comparison (PID21146) requires a quiet host. No dependencies or environments were changed, and no additional tests were started.
After that process finishes, validate Ruff in isolation and qualify inference upgrades per environment. Existing user edits were preserved.

## 简体中文

已通过官方PyPI及Ruff release检查stable候选版本。
Ruff0.17.1、MLX0.32.4、MLX-LM0.32.2有更新；setuptools84.0.0及PyInstaller6.22.3仍为最新版。
Strata内存对比试验（PID21146）正在运行并要求低负载，因此没有修改依赖或环境，也没有启动额外测试。
试验结束后先隔离验证Ruff，再按推理环境分别验证升级。保留了用户已有修改。

## ユーザー指示による再開 / Resumed update / 恢复更新

以下は上記の朝の保留判断を更新する、実際の作業結果。

### 日本語

- 🟢 [Done] Ruffを0.17.0→0.17.1へ更新。`pyproject.toml`と`requirements-dev.lock`を整合させ、py310・既存ruleを維持。`tests/test_strata_runtime.py`の作業中の追加内容を保持し、importの改行だけ整形。
- 🟢 [Done] 新Ruffで`ruff check --no-cache vllm_apple tests`成功、一時venvの`pip check`成功。setuptools84.0.0／PyInstaller6.22.3は最新stableなので維持。
- 🟢 [Done] 公式vLLM core0.31.0+cpu／vLLM-Metal0.31.0を、Homebrewと独立した`.venv-vllm-metal-031`へ導入。system-site-packages=false。既存Homebrew0.30.0と生成系venvは変更していない。新venvの`pip check`成功、直接importでMetalPlatformを選択し`MetalPlatform.is_available()=True`を確認。
- 🟠 [Next] 新backendの既定採用は見送り。`qualification-preflight`は既存の5秒制限で`backend_environment_inspection_failed`。初回・warm・他試験終了後にも失敗。段階別の手動診断ではplatform importに9.823秒、Metal availability=true。制限を緩和せず、model inference／quality／memoryの検証開始前に停止した。原因を単にhost loadと断定しない。timeoutで版情報も取得できず、報告上のcandidate-version-mismatchは版の不一致が確定した意味ではない。
- ⭕️ [Pending] Core ML変換元と画像・動画生成モデル不足により、Core ML tools9.0、Diffusers0.41.0、MLX-Gen0.38.0、mflux0.23.0の実モデルparity・quality・memory検証は未実施。手元のmodelsはGemma2とCore ML artifact等であり、生成backendを認定しない。

vLLM/Metal0.31.0の公式wheel metadataは、MLX==0.32.1、MLX-LM固定commit `9e6acca691e64d6d8bb808c328fcdea459099cca`、Transformers>=5.10.4,<5.18.0、Torch==2.13.0、setuptools==77.0.3を要求する。したがってMLX0.32.4、MLX-LM0.32.2、Transformers5.20.1、Torch2.14.1をこのbackendへ単独適用しない。Transformersは制約内のstable5.17.0を新venvへ導入した。project build用setuptools84.0.0とも環境を分離した。

依存解決と主要wheelのURL・SHA256は[evidence](evaluation/vllm-metal-0.31.0-dependency-candidate-2026-10-19.json)、新venvのfreezeは[候補snapshot](evaluation/vllm-metal-0.31.0-candidate-freeze-2026-10-19.txt)。これらは未認定候補の再現用資料であり既定runtimeへの昇格ではない。upstreamが固定する間接依存（例: opentelemetry-semantic-conventionsのbeta版番号）も含むため、すべての間接依存がstableになったとは扱わない。公式stableの同名PyPI0.1.0を誤採用せず、公式releaseのwheelを指定した。

検証：

- MLXがない通常Python: Strata4 testsは全skip。MLXのある既存Homebrew環境で同じ4 testsを再実行し全成功。
- backend／compat／Metal integration／preflight／promotion関連50 tests成功。初回sandbox内のHTTP bind失敗は、ソケットが許可される実行で再検証した。
- Python3.10と3.12のcompileall成功。
- Swift SDK38 tests成功、Mac sample build成功（8.59秒）。今回Swiftやpackaging toolの変更はなく、unsigned配布packageの再作成はしていない。
- 全Python回帰初回1573 tests、51.009秒、1 error・46 skips。APIログ取得テストの`StopIteration`。該当APIテストは単独再実行で成功（0.596秒）。初回失敗は消さず、全回帰再実行結果を追記する。
- 新backendはpip導入時に通信切断で失敗し、再試行で成功。最初の未完了installを合格とは扱っていない。
- 新backendはpreflight不合格のため、実モデルsmoke／30分認定／数値・品質・メモリ認定は実施していない。既存backendの資格を新versionへ引き継がない。

上流根拠：[Ruff0.17.1](https://github.com/astral-sh/ruff/releases/tag/0.17.1)、[vLLM-Metal0.31.0](https://github.com/vllm-project/vllm-metal/releases/tag/v0.31.0)、[vLLM0.31.0](https://github.com/vllm-project/vllm/releases/tag/v0.31.0)。最新版条件は公式PyPI JSONでも再確認。CI Actionsのremote runner検証は引き続き⭕️ [Pending]。

### English

Updated Ruff from0.17.0 to0.17.1, preserving Python3.10 targeting and user edits. Lint, Python3.10/3.12 compilation, 4 native Strata tests, 50 backend contract tests, 38 Swift tests and the Mac sample build passed. The first full Python run had one API logging error out of1573 tests (46 skipped); its isolated retry passed.
Installed official vLLM/Metal0.31.0 into a separate candidate environment. Dependency checks and direct Metal availability passed, but the unchanged five-second preflight failed; platform initialization took9.823 seconds in a manual diagnostic. The candidate is not promoted and no model qualification is claimed. Retain upstream exact dependency constraints and existing Homebrew/runtime environments. Generation and conversion upgrades await the required model artifacts.

### 简体中文

Ruff从0.17.0更新为0.17.1，保留Python3.10目标和用户现有修改。Lint、Python3.10/3.12语法检查、4个native Strata测试、50个backend相关测试、38个Swift测试和Mac示例构建成功。完整Python回归首次运行1573个测试，出现1个API日志错误、46个跳过；该测试单独重跑成功。
在独立候选环境安装官方vLLM/Metal0.31.0。依赖检查和Metal直接可用性检查成功，但现有5秒preflight未通过；手动诊断中platform初始化耗时9.823秒。因此没有切换默认runtime，也没有认定模型质量及内存表现。保留上游精确依赖约束及原Homebrew环境，生成和转换升级等待所需模型artifact。

### 最終回帰結果 / Final regression / 最终回归

🟢 [Done] 全Python回帰の再実行は1573 tests、39.073秒、OK（46 skipped）。初回のAPIログ取得エラーの根因を修正・解決したとは扱わず、単独再試験と全回帰の再実行で再現しなかった事実を記録する。

English: The complete retry passed1573 tests in39.073 seconds with46 skips. The initial API logging error remains recorded; its root cause is not claimed resolved.

简体中文：完整重跑1573个测试成功，耗时39.073秒，46个跳过。保留首次API日志错误记录，不宣称已修复其根因。
