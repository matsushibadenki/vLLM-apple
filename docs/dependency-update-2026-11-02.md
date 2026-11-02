# Stable dependency maintenance — 2026-11-02

## 日本語

🟢 [Done] Ruffを0.17.2から最新版stable0.17.3へ更新。
`pyproject.toml`と`requirements-dev.lock`を整合させ、Python>=3.10、py310 target、明示lint ruleを維持。
setuptools84.2.1とPyInstaller6.22.3は公式stable最新版なので維持した。
ランタイムdependencies=[]、バックエンド別環境、Homebrew管理環境を変更していない。

### 既存作業と競合

対象は `/Users/Shared/Program/Tauri/Soliton/vLLM-apple`。
追加の適用対象AGENTS.mdは見つからず、チャットの指示に従った。
前回の[2026-10-26記録](dependency-update-2026-10-26.md)、依存定義・CI・ROADMAPを確認。
P1／Strata比較／推論／qualificationの関連実プロセスは、開始時と検証前の再確認で見つからなかった。
最新更新のP1 r11 checkpointはfailed・runner_alive=falseであり、過去の未認定結果として扱う。

既存のmanaged engine／service、P1 runner、Gemma batch qualification、P3 profiler、関連テストと未追跡のbenchmark GPU lock実装を保持。
検証開始時のコピーと変更済み・未追跡Python9ファイルのSHA256一致を確認した。
既存ROADMAPの変更も保持して新しい依存更新欄を追加した。これらの実装を今回の依存更新成果とは扱わない。

### 実際の検証

候補は `/private/tmp/vllm-apple-deps-20261102-env`、ソースは一時コピーで検証。

- 🟢 [Done] 新Ruff0.17.3・旧Ruff0.17.2ともlint0件。同じソースのfilename／code／行列／messageで比較し追加0・解消0。unsafe fixや一括整形は実施していない。
- 🟢 [Done] Python3.12の全unit回帰1608 tests、60.546秒、OK（46 skipped）。skipは検証成功に含めない。
- 🟢 [Done] Python3.10／3.12のcompileall、候補venvのpip check成功。
- 🟢 [Done] Swift SDK38 tests成功。Mac sample build成功（15.00秒）。Swift packagesは外部packageなし／ローカルSDK参照のまま。
- 🟢 [Done] wheel作成・install・CLI help起動成功。ビルドは維持したsetuptools84.2.1を使用。
- 前回のlint33件／起動timeoutは今回のソースで再現しなかった。旧Ruffでもlint0件なので、Ruff更新によって修正したとは扱わず、起動不安定性の根因解決も認定しない。
- PyInstaller自体は変更していない。今回のpackaging検証はwheelとCLIまでで、unsigned Mac配布ZIPの再作成はしていない。
- backendの依存定義・実行環境は変更していないため、実モデルのquality／numerical／memory認定やdoctorの再検証は今回行っていない。前回のdoctor失敗や既存モデルの資格を新しい認定結果に置き換えない。

### 棚卸しと見送り

| 対象 | 確認した既存版 | 最新stable | 判断 |
| --- | --- | --- | --- |
| setuptools | 84.2.1 | 84.2.1 | 維持 |
| Ruff | 0.17.2 | 0.17.3 | 適用 |
| PyInstaller | 6.22.3 | 6.22.3 | 維持 |
| vLLM／Metal | Homebrew0.32.0+cpu／0.32.0 | official Metal0.32.0 | 現行stableを維持、再認定なし |
| MLX／MLX-LM | Homebrew0.32.2／0.32.0固定commit | 0.32.4／0.32.2 | Metalの固定制約を維持 |
| Transformers／tokenizers | Homebrew5.19.0／0.23.3 | 5.22.1／0.23.3 | MetalのTransformers<5.20.0制約を守り維持。最新TransformersはPython>=3.11 |
| PyTorch／torchao | Homebrew2.13.0、生成環境2.14.0／0.18.0 | 2.15.0／0.18.0 | MetalのTorch==2.13.0を維持。最新TorchはPython>=3.11 |
| MLX-Gen | 0.33.1、MLX0.31.2 | 0.38.0 | ⭕️ [Pending] 対応生成modelのquality／memory検証待ち |
| Core ML tools | 8.1、NumPy1.26.4 | 9.0 | ⭕️ [Pending] 変換元modelとartifact parity検証待ち |
| Diffusers | 0.41.0.dev0、Torch2.14.0、Transformers5.17.0 | 0.42.1 | ⭕️ [Pending] 生成modelのAPI・数値検証待ち。最新はPython>=3.11。既存dev版をstableへ移行したとは扱わない |
| mflux | 棚卸ししたvenvには未導入 | 0.24.0 | 利用環境への不要な追加は行わない。最新はPython>=3.11 |
| NumPy | Homebrew2.3.5、Core ML1.26.4、MLX-Gen2.2.6、Qwen Image2.5.3 | 2.5.4 | 環境ごとの制約と数値検証を優先。最新はPython>=3.12 |
| CI checkout／setup-python／upload-artifact | @v6 | 7.0.1／7.1.0／7.0.2 | ⭕️ [Pending] 実GitHub runner未検証のため維持 |

Homebrew Metal0.32.0はMLX==0.32.2、MLX-LM commit
`9e6acca691e64d6d8bb808c328fcdea459099cca`、mlx-vlm==0.7.6、Transformers>=5.16.1,<5.20.0を指定。
core0.32.0はTorch==2.13.0、setuptools==78.1.1を指定するため、project build環境と分離する。
これらを破る単独更新を行っていない。vendorコード・上流固定commitも変更していない。
手元のmodelsはGemma2とCore ML artifact等であり、画像・動画生成modelやCore ML変換元の代替にはならない。

### 上流根拠と次の工程

公式PyPI JSON `https://pypi.org/pypi/<package>/json` で表の候補版・Python要件を確認し、
GitHub公式releaseのprerelease=falseを照合。beta／RC／nightlyを直接更新候補に含めていない。

- [Ruff0.17.3](https://github.com/astral-sh/ruff/releases/tag/0.17.3)：公式APIでrelease notesを確認。UP032修正・内部性能修正。preview ruleは有効化しない。
- [Metal0.32.0](https://github.com/vllm-project/vllm-metal/releases/tag/v0.32.0)
- [checkout7.0.1](https://github.com/actions/checkout/releases/tag/v7.0.1)、[setup-python7.1.0](https://github.com/actions/setup-python/releases/tag/v7.1.0)、[upload-artifact7.0.2](https://github.com/actions/upload-artifact/releases/tag/v7.0.2)

🟠 [Next] 現在のHomebrew backendのdoctor／起動検査を、長時間試験と競合しない工程で再検証する。
⭕️ [Pending] 生成・変換modelを必要とするqualification、CI Python3.13／macOS15 runnerとActions最新版の実環境検証。

## English

Updated Ruff0.17.2→0.17.3 in both dependency definitions. Old and new Ruff each report zero diagnostics on the current source.
The complete Python3.12 suite passed1608 tests with46 skips. Python3.10/3.12 compilation, pip check, wheel build/install, CLI startup,38 Swift SDK tests and the Mac sample build passed.
Existing user changes were preserved. Earlier failures were not reproduced, but their root causes are not claimed fixed by this dependency update.
Setuptools and PyInstaller remain current. Inference environments retain their exact upstream constraints; model-dependent qualification and remote CI remain pending. No new backend qualification is claimed.

## 简体中文

已在两个依赖定义中将Ruff0.17.2更新为0.17.3。对当前代码，新旧Ruff均为零诊断。
完整Python3.12回归1608个测试成功、46个跳过；Python3.10/3.12语法检查、pip check、wheel构建安装、CLI启动、38个Swift SDK测试及Mac示例构建成功。
保留用户已有修改。此前错误本次未复现，但不宣称依赖更新已修复其根因。
Setuptools和PyInstaller仍为最新版。推理环境保留上游精确约束；需要模型的qualification及remote CI继续Pending，不新增后端认证。
