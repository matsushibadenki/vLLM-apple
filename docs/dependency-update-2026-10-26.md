# Stable dependency maintenance — 2026-10-26

## 日本語

🟢 [Done] setuptools84.0.0→84.2.1、Ruff0.17.1→0.17.2を適用。
Python>=3.10、target-version=py310、明示lint rule、runtime dependencies=[]を維持する。
PyInstaller6.22.3は公式stable最新版のままなので変更しない。

### 移動先と既存作業

旧 `/Users/Shared/Program/vLLM-apple` は存在せず、Codexの同じproject idが指す
`/Users/Shared/Program/Tauri/Soliton/vLLM-apple` を確認して継続した。
定期実行のpromptもこの移動先へ更新し、週次の曜日・時刻・通知方針は維持した。
移動先で追加の適用対象AGENTS.mdは見つからず、チャットの指示に従った。
既存backend.pyのKV／allocator予算変更と、未追跡の関連2テスト・mlx_kv_budget.pyを保持。
テスト用コピーにも既存作業を含め、今回の更新は依存定義と記録に限定した。
前回の[2026-10-19記録](dependency-update-2026-10-19.md)を参照した。

process確認でStrata比較・P1・推論・qualification runnerを検出しなかった。
P1 r11 statusはfailed／runner_alive=falseであり、過去の未認定結果として保持。

### 検証

一時コピー `/private/tmp/vllm-apple-deps-20261026-source` と一時Python3.12環境を使用。
Homebrew管理環境や既存backend venvをpipで上書きしていない。

- setuptools84.2.1でwheel作成成功。wheelのインストールと`vllm-apple --help`も成功。
- Python3.10／3.12のcompileall成功。一時venvのpip check成功。
- Ruff0.17.2のlintは33件の既存違反で不合格。旧Ruff0.17.1も同じコピーに対し33件。filename・code・行列・messageを比較して追加0／解消0。既存作業を一括整形しない。
- 全Python回帰1588 tests、184.112秒、2 errors・46 skips。HTTP timeout回復とchild main-thread HTTP経路でsubprocess startup timeout。全合格とは扱わない。
- 更新前の元作業ツリー・通常Python3.12でも関連9 testsを検証し、同種の子process起動timeoutが4件発生。失敗した個別test名は異なるため、今回の2件を一件ずつ同一条件で再現したとは扱わない。根因や負荷との因果は未確定。
- Swift結果は下記最終検証欄に記載。PyInstallerとSwiftの依存定義は不変。setuptoolsの配布検証はwheel作成・install・CLI起動までで、unsigned Mac配布ZIPは再作成していない。

### 上流・バックエンドの棚卸し

| 対象 | 確認した既存版 | 最新stable候補 | 判断 |
| --- | --- | --- | --- |
| setuptools | 84.0.0 | 84.2.1 | 🟢 [Done] project build定義更新 |
| Ruff | 0.17.1 | 0.17.2 | 🟢 [Done] project定義とdev lockを同時更新。lint追加違反なし |
| PyInstaller | 6.22.3 | 6.22.3 | 維持 |
| vLLM／Metal | Homebrew0.32.0+cpu／0.32.0 | official Metal0.32.0 | 既に別作業で導入済み。今回変更・再認定しない |
| MLX／MLX-LM | Homebrew0.32.2／0.32.0固定commit | 0.32.4／0.32.2 | Metalの正確な制約を維持 |
| Transformers／tokenizers | Homebrew5.19.0／0.23.3 | 5.20.2／0.23.3 | MetalはTransformers<5.20.0のため単独更新しない |
| PyTorch／torchao | Homebrew2.13.0、生成2.14.0／0.18.0 | 2.14.1／0.18.0 | backendの版制約とqualificationを優先 |
| MLX-Gen | 0.33.1（MLX0.31.2、NumPy2.2.6） | 0.38.0 | ⭕️ [Pending] 対応生成モデルでquality／memory未検証 |
| Core ML tools | 8.1（NumPy1.26.4） | 9.0 | ⭕️ [Pending] 変換元modelとartifact parity未検証 |
| Diffusers | 0.41.0.dev0（Torch2.14.0、Transformers5.17.0） | 0.42.0 | ⭕️ [Pending] 最新はPython>=3.11、生成モデルでのAPI・数値検証待ち。既存dev版を今回stableへ移行したとは扱わない |
| mflux | 上記venvには未導入 | 0.23.0 | 不要な環境追加は行わない |
| NumPy | Homebrew2.3.5、Core ML1.26.4、Qwen Image2.5.3 | 2.5.4 | 最新はPython>=3.12。環境ごとの制約と数値検証を維持 |
| CI checkout／setup-python／upload-artifact | @v6 | 7.0.1／7.1.0／7.0.2 | ⭕️ [Pending] 実GitHub runnerで未検証のため維持 |
| Swift | 外部packageなし・local SDK参照 | — | 依存変更なし |

現在のMetal0.32.0 metadataはMLX==0.32.2、MLX-LM commit
`9e6acca691e64d6d8bb808c328fcdea459099cca`、mlx-vlm==0.7.6、
Transformers>=5.16.1,<5.20.0を指定。
core0.32.0はTorch==2.13.0、setuptools==78.1.1を指定するため、
project build用setuptools84.2.1とは分離する。
現在のdoctorもbackend_environment_inspection_failedで不合格。版取得まで到達していない。
以前の0.31候補や既存モデルの認定を0.32へ引き継がず、gateを緩めない。
今回、実モデル生成・数値・メモリ試験は行っていない。
vendorコード・上流固定commitの書き換えは行っていない。

### 根拠と次の工程

公式PyPI JSON `https://pypi.org/pypi/<package>/json` と公式releaseからstable版を確認。
beta／RC／nightlyを直接更新候補に含めていない。

- [setuptools84.2.1](https://github.com/pypa/setuptools/releases/tag/v84.2.1)
- [Ruff0.17.2](https://github.com/astral-sh/ruff/releases/tag/0.17.2)：bug fixes／内部性能修正。既存rule設定を維持しunsafe fixは実行しない。
- [Metal0.32.0](https://github.com/vllm-project/vllm-metal/releases/tag/v0.32.0)：core0.32.0対応。新しいMPS経路はexperimentalなので採用しない。

🟠 [Next] 既存lint33件、子process起動の不安定性、doctor失敗を別工程で修正・検証する。
⭕️ [Pending] 生成／変換モデルが必要なqualificationと、実remote CIの検証。

## English

Updated project build setuptools84.0.0→84.2.1 and Ruff0.17.1→0.17.2.
The project moved under Tauri/Soliton; the recurring prompt now follows that verified project path.
Wheel build/install, CLI startup, Python3.10/3.12 compilation and pip check passed.
Both old and new Ruff report the same33 existing diagnostics; no new diagnostic was introduced.
The full Python suite ran1588 tests with2 startup-timeout errors and46 skips. Related baseline tests also show startup instability, with different failing test names; the errors are not claimed fixed.
Homebrew already contains Metal/core0.32.0. Preserve its exact dependency constraints and existing environments; doctor still fails. Model-dependent inference upgrades and remote CI remain pending.

## 简体中文

project build的setuptools从84.0.0更新为84.2.1，Ruff从0.17.1更新为0.17.2。
确认项目已移动到Tauri/Soliton下，并同步修改定期执行的目标路径。
wheel构建安装、CLI启动、Python3.10/3.12语法检查及pip check成功。
新旧Ruff均报告相同的33个已有问题，没有新增诊断。
完整Python回归1588个测试出现2个启动timeout、46个跳过。旧环境的相关测试也出现启动不稳定，但失败test名称不同，不宣称已修复。
Homebrew已安装Metal/core0.32.0，保留其精确依赖约束；doctor仍未通过。需要模型的推理验证及remote CI继续保留为Pending。

## 最終検証 / Final verification / 最终验证

🟢 [Done] Swift SDK38 tests成功。Mac sample build成功。ユーザーのbackend.pyはsnapshotとのSHA256一致を確認し、既存変更を保持した。

English: All38 Swift SDK tests and the Mac sample build passed. The pre-existing backend.py content was preserved byte for byte.

简体中文：38个Swift SDK测试及Mac示例构建成功；通过SHA256确认保留了用户原有backend.py内容。
