# Stable dependency maintenance — 2026-10-12

## 日本語

互換性を維持するため、開発・配布ツールと推論バックエンドを別単位で確認した。
既存の未コミット変更（Strata、expert residency、ROADMAP、評価資料）は保持した。
追加のAGENTS.mdはリポジトリ内のファイル検索では見つからず、チャットで指定された指示に従った。
前回の上流判断は `docs/upstream-review-2026-10-05.md` を参照した。
開始時の関連process確認では推論・qualification・P1 runnerはなく、
`p1_stability_status.py docs/evaluation/p1-stability-m4-2026-10-08-r11` はfailed・runner_alive=false。
過去P1の失敗は今回の更新結果ではない。

🟢 [Done] setuptools84.0.0、Ruff0.17.0、PyInstaller6.22.3の依存定義と既存配布テストを更新。推論backendは変更しない。

### 候補と判断

| 対象 | 既存 | 最新stable候補 | 判断 |
| --- | --- | --- | --- |
| setuptools | 80.9.0 | 84.0.0 | Python>=3.10。wheel作成・インストールを隔離環境で検証 |
| Ruff | 0.16.0 | 0.17.0 | dev定義とlockを同時更新。target-version=py310と明示rule選択を維持 |
| PyInstaller | 6.16.0 | 6.22.3 | Python>=3.8,<3.16。CIと同じPython3.12で検証。既存版固定テストも更新 |
| vLLM core / Metal | 0.30.0+cpu / 0.30.0 | core 0.31.0 / official Metal 0.30.0 | 維持。Metalの公式stableはcore0.30.0対応。PyPI同名0.1.0は採用しない |
| Homebrew MLX / MLX-LM | 0.32.1 / 0.32.0 | 0.32.3 / 0.32.0 | 維持。MetalはMLX==0.32.1とMLX-LM git commit 9e6acca691e64d6d8bb808c328fcdea459099ccaを指定 |
| Transformers / tokenizers | 5.19.0 / 0.23.3 | 5.19.0 / 0.23.3 | 更新なし |
| Homebrew PyTorch | 2.13.0 | 2.14.1 | 維持。core/Metal組合せの独立qualificationが必要 |
| Core ML tools / NumPy | 8.1 / 1.26.4 | 9.0 / 2.5.4 | 維持。変換artifactと数値parityを先に検証。NumPy最新はPython>=3.12 |
| MLX-Gen環境 | mlx-gen0.33.1、mlx0.31.2、torch2.14.0、numpy2.2.6 | mlx-gen0.38.0、mlx0.32.3、torch2.14.1、numpy2.5.4 | 維持。生成品質・メモリqualification未実施 |
| Qwen Image環境 | diffusers0.41.0.dev0、torch2.14.0、torchao0.18.0、numpy2.5.3 | diffusers0.41.0、torch2.14.1、torchao0.18.0、numpy2.5.4 | 今回は維持。既存dev版のstable移行を優先候補にする。新しいdev版は導入しない |
| mflux | 確認した環境では未導入 | 0.22.0 | インストールを追加しない。利用環境と既存adapterのAPI比較が必要 |
| Swift packages | SDK・サンプルとも外部packageなし | — | ローカルSDK参照を維持 |
| CI Actions | checkout/setup-python/upload-artifact @v6 | v7.0.1 / v7.0.0 / v7.0.2 | ⭕️ [Pending] 実際のGitHub runnerで検証できていないため維持 |

runtimeのdependencies=[]、バックエンド別venv、Homebrew管理環境を維持。
最新stable候補はPyPI JSONと公式GitHub releaseから確認し、pre-releaseは更新候補に含めなかった。
直接取り込むvendorコードの変更はない。依存する上流固定commitを勝手に最新commitへ置換しない。

### 検証と制約

検証は作業ツリーのコピー `/private/tmp/vllm-apple-deps-20261012` と
Python3.12.15の一時venvで実施。推論環境へのpip上書きは行っていない。

- Ruff0.17.0: `ruff check vllm_apple tests` 成功。
- Python3.10 / 3.12: `compileall -q vllm_apple tests` 成功。
- setuptools84.0.0: `pip wheel --no-deps --no-build-isolation .` とprojectインストール成功。
- 候補ツールvenv: `pip check` 成功。間接依存はaltgraph0.17.5、macholib1.16.4、packaging26.3、pyinstaller-hooks-contrib2026.8。wheel0.48.0は検証用。
- Swift SDK: 38 tests成功。
- SwiftUI Mac sample: `swift build --package-path samples/VLLMAppleChat` 成功。
- 全Python回帰: 1553 tests、285.162秒、3 failures・1 error・40 skips。全合格とは扱わない。
- PyInstaller旧版を文字列固定した配布contractテストは候補版へ修正し、`tests.test_xcode_sample` 9 tests成功。
- daemon再起動のhealth待機失敗と子process起動timeoutは、変更前の作業ツリー・Python3.12でも再現した。根因や性能改善は未認定。
- Metal shape試験は全回帰で失敗したが、変更前・候補の個別再実行では成功。環境負荷等を原因と断定しない。
- 最初のコピーではexperimentsが不足し、全回帰は中断した。補完後に上記1553 testsを完走。初回を合格証拠には使わない。
- Python3.12の既存コード個別試験を15秒で打ち切った例もあり、timeoutだけでdeadlockと判断していない。
- PyInstaller6.22.3: standalone daemon生成・`--help`起動、Macアプリへの組み込み、unsigned arm64 ZIP作成・SHA256検証成功。
- Python3.13 CI、macOS15 runner、推論backendの実model quality/numerical/memory試験は今回未実施。backend最新版を認定しない。

### 次の工程

🟠 [Next] daemon／子process起動失敗の根因を別工程で調査し、全回帰を再度合格させる。試験のdeadlineやquality gateを緩和しない。
🟠 [Next] Qwen ImageのDiffusers stable移行を隔離環境で比較し、既存モデル・API・数値・メモリの結果が揃ってから採用する。
🟠 [Next] MLX-Gen／Core ML tools／PyTorchの更新は各環境のqualification単位で進める。
⭕️ [Pending] GitHub Actions v7、CI Python3.13、macOS15 runnerの検証。

### 上流根拠

- [setuptools84.0.0](https://github.com/pypa/setuptools/releases/tag/v84.0.0) / [PyPI metadata](https://pypi.org/pypi/setuptools/json)
- [Ruff0.17.0 release notes](https://github.com/astral-sh/ruff/releases/tag/0.17.0): Pythonの既定とrule変更があるため明示py310とrule設定を維持。
- [PyInstaller6.22.3](https://github.com/pyinstaller/pyinstaller/releases/tag/v6.22.3) / [PyPI metadata](https://pypi.org/pypi/pyinstaller/json)
- [official vLLM-Metal0.30.0](https://github.com/vllm-project/vllm-metal/releases/tag/v0.30.0)
- Backend候補の版・Python条件は `https://pypi.org/pypi/<package>/json` で確認。対象: numpy、vllm、mlx、mlx-lm、mlx-gen、transformers、tokenizers、torch、torchao、coremltools、diffusers、mflux。
- [checkout7.0.1](https://github.com/actions/checkout/releases/tag/v7.0.1)、[setup-python7.0.0](https://github.com/actions/setup-python/releases/tag/v7.0.0)、[upload-artifact7.0.2](https://github.com/actions/upload-artifact/releases/tag/v7.0.2)。

## English

Updated setuptools to84.0.0, Ruff to0.17.0 and PyInstaller to6.22.3. The standalone daemon and unsigned arm64 Mac package built successfully with a valid checksum. Development and packaging candidates were validated in a temporary Python3.12 environment.
Lint, Python3.10/3.12 compilation, wheel creation/install, 38 Swift SDK tests and the Mac sample build passed.
The complete Python suite ran 1553 tests with 3 failures, 1 error and 40 skips; it did not pass.
The packaging version assertion was updated and its 9 tests passed. Daemon and subprocess startup failures also reproduced before dependency changes; a Metal shape failure passed on isolated reruns.
Inference environments remain unchanged. Metal's exact MLX0.32.1 and MLX-LM commit requirements are retained.
Backend candidates need independent qualification. CI Actions v7 remain pending actual runner validation.

## 简体中文

已更新setuptools84.0.0、Ruff0.17.0和PyInstaller6.22.3。独立daemon及unsigned arm64 Mac打包成功，SHA256检查通过。在临时Python3.12环境中验证开发及打包工具候选版本。
Lint、Python3.10/3.12语法检查、wheel构建与安装、38个Swift SDK测试及Mac示例构建成功。
完整Python回归运行1553个测试，出现3个失败、1个错误、40个跳过，不能视为全部通过。
更新打包版本断言后，相关9个测试成功。daemon及子进程启动失败在更新前也复现；Metal shape测试单独重跑成功。
推理环境保持原样，保留Metal要求的MLX0.32.1及MLX-LM固定commit。
后端升级需独立qualification，CI Actions v7等待真实runner验证。
