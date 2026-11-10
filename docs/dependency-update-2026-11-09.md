# Stable dependency maintenance — 2026-11-09

## 日本語

🟢 [Done] Ruffを0.17.3→最新版stable0.17.4へ更新。project定義とdev lockを整合させ、Python>=3.10、py310 target、明示lint rule、runtime dependencies=[]を維持。
setuptools84.2.1／PyInstaller6.22.3は公式stable最新版なので変更しない。

### 既存作業と検証

対象は `/Users/Shared/Program/Tauri/Soliton/vLLM-apple`。追加の適用対象AGENTS.mdは見つからず、チャットの指示を適用。
前回の[2026-11-02記録](dependency-update-2026-11-02.md)、CI、ROADMAP、依存定義を確認した。
関連processを確認して推論／P1／Strata比較／qualificationの実行を検出しなかった。
最新更新のP1 r11 checkpointはfailed・runner_alive=false。過去の結果であり今回の依存更新の失敗ではない。

既存の`mlx_kv_budget.py`のreviewed source pair追加と、未追跡のhost preflight評価資料を保持。
ソース一時コピーと元`mlx_kv_budget.py`のSHA256一致を確認。既存実装の改善を今回の成果には含めない。
Homebrew環境・既存backend venv・vendor・上流固定commitは変更していない。

候補環境は `/private/tmp/vllm-apple-deps-20261109-env`、検証対象は既存作業を含む一時ソースコピー。

- 🟢 [Done] Ruff新0.17.4／旧0.17.3ともlint0件。診断のfilename・code・行列・message比較で追加0／解消0。unsafe fix／preview rule有効化／一括整形は行っていない。
- 🟢 [Done] Python3.12の全回帰1632 tests、75.584秒、OK（46 skipped）。skipは実機検証済みとは扱わない。
- 🟢 [Done] Python3.10／3.12 compileall、候補環境pip check成功。
- 🟢 [Done] Swift SDK38 tests、Mac sample build成功（6.06秒）。Swiftは外部packageなし／local SDK参照のまま。
- 🟢 [Done] setuptools84.2.1でwheel作成成功。wheel install・CLI help起動・install後pip checkも成功。
- PyInstallerは不変。packaging検証はwheelとCLIまでで、unsigned Mac配布ZIPは再作成していない。
- 今回はbackendを変更していない。実モデルquality／numerical／memory試験やdoctorの再検証は行わず、既存認定を新しいstackへ引き継がない。

### 重要な既存環境の変化

Homebrewには今回の処理開始前からMetal/core0.33.0が導入済み。公式releaseはstableだが、環境には
Torch **2.16.0.dev20261013** とMLX-LM **0.32.3.dev12+g4cfc64784** がある。
公式Metal0.33.0 release notesもMPS経路向けPyTorch nightly固定を説明しており、stable channel名だけでは間接依存の全stable性を保証できない。

読み取りのHomebrew `pip check` は不合格：core0.33.0 metadataのDarwin要件`torch==2.14.0`に対して2.16.0.dev20261013が導入されている。
これは今回のRuff更新前から存在し、今回の更新による導入・回帰ではない。
metadataの不一致を確認した事実であり、MLX経路の実際の推論失敗が確認された意味ではない。
Homebrewをpipで上書きしたり、Torch／MLX-LMだけをdowngradeしたりせず、現在のstackを今回認定しない。

Metal0.33.0 metadataはMLX==0.32.4、MLX-LM commit
`4cfc6478473b0668ede3eb0cc93f2a9181a15052`、mlx-vlm commit
`b7bec926d061b6fd8935b244eb3e76435aa04514`、Transformers>=5.16.1,<5.21.0を指定。
core0.33.0はTorch==2.14.0とsetuptools==78.1.1を指定する。
project build用setuptools84.2.1とは引き続き環境を分離する。

🟠 [Next] upstream installerのnightly固定とwheel metadataを照合し、stableのみで構成できる互換stackを隔離環境で検証する。MLX／MPS経路別のquality・数値・メモリ結果が揃う前に既定を置き換えない。

### 今回の棚卸し

| 対象 | 既存／棚卸しした環境 | 最新stable | 採用判断 |
| --- | --- | --- | --- |
| setuptools／PyInstaller | 84.2.1／6.22.3 | 同左 | 維持 |
| Ruff | 0.17.3 | 0.17.4 | 適用 |
| vLLM-Metal | Homebrew0.33.0 | 0.33.0 | 今回変更・再認定なし。上記の間接依存問題を記録 |
| MLX／MLX-LM | Homebrew0.32.4／0.32.3.dev12+g4cfc64784 | 0.32.4／0.32.2 | 固定commitを単独置換しない。全stable stackとして未認定 |
| Transformers／tokenizers | Homebrew5.20.2／0.23.3 | 5.22.2／0.23.3 | MetalのTransformers<5.21.0制約を維持。最新はPython>=3.11 |
| PyTorch／torchao | Homebrew上記nightly、生成2.14.0／0.18.0 | 2.15.0／0.18.0 | 上記のstack単位の再検証が必要 |
| MLX-Gen | 0.33.1、MLX0.31.2、NumPy2.2.6 | 0.38.0 | ⭕️ [Pending] 対応生成modelでquality／memory未検証 |
| Core ML tools | 8.1、NumPy1.26.4 | 9.0 | ⭕️ [Pending] 変換元modelとartifact parity未検証 |
| Diffusers | 0.41.0.dev0、Torch2.14.0、Transformers5.17.0 | 0.42.1 | ⭕️ [Pending] 対応生成modelのAPI・数値検証待ち。既存dev版をstableへ移行したとは扱わない |
| mflux | 棚卸ししたvenvには未導入 | 0.24.0 | 不要な環境追加なし。最新はPython>=3.11 |
| NumPy | Homebrew2.3.5、Core ML1.26.4、MLX-Gen2.2.6、Qwen Image2.5.3 | 2.5.4 | 環境別の制約と数値検証を優先。最新はPython>=3.12 |
| CI checkout／setup-python／upload-artifact | @v6 | 7.0.1／7.1.0／7.0.2 | ⭕️ [Pending] 実remote runner検証待ち |

手元のmodelsはGemma2とCore ML artifact等であり、画像・動画生成modelやCore ML変換元の代替にはならない。
生成系venvの古いTransformers／tokenizersもstack単位の検証まで維持する。

### 上流根拠

公式PyPI JSON `https://pypi.org/pypi/<package>/json` で候補版・Python条件を確認。
GitHub公式releaseのprerelease=falseを照合し、beta／RC／nightlyを今回の直接更新候補に含めていない。
既存の間接dev版を今回導入したり、stableとして再分類したりしていない。

- [Ruff0.17.4](https://github.com/astral-sh/ruff/releases/tag/0.17.4)：公式release APIでnotes確認。構文判定／診断描画等の修正、内部性能修正。既存設定で比較した。
- [Metal0.33.0](https://github.com/vllm-project/vllm-metal/releases/tag/v0.33.0)：MPS経路のnightly固定を含む公式説明。
- [checkout7.0.1](https://github.com/actions/checkout/releases/tag/v7.0.1)、[setup-python7.1.0](https://github.com/actions/setup-python/releases/tag/v7.1.0)、[upload-artifact7.0.2](https://github.com/actions/upload-artifact/releases/tag/v7.0.2)

⭕️ [Pending] 実remote CIのPython3.13／macOS15とActions最新版、生成・変換modelを必要とするqualification。

## English

Updated Ruff0.17.3→0.17.4. Old and new Ruff both report zero diagnostics.
All1632 Python tests passed (46 skipped); Python3.10/3.12 compilation, pip check, wheel build/install, CLI startup,38 Swift tests and the Mac sample build passed. User edits were preserved.
Homebrew had independently moved to Metal/core0.33.0 and contains development versions of Torch and MLX-LM. Its pip check fails because the installed Torch nightly differs from core's2.14.0 metadata pin; official Metal notes describe the nightly installer pin. This is pre-existing and does not by itself prove an MLX inference failure. No inference environment was modified or newly qualified. Validate a stable-only compatible stack in isolation before promotion; model-dependent generation and remote CI remain pending.

## 简体中文

已将Ruff0.17.3更新为0.17.4，新旧Ruff均为零诊断。
1632个Python测试成功（46个跳过）；Python3.10/3.12语法检查、pip check、wheel构建安装、CLI启动、38个Swift测试及Mac示例构建成功，保留了用户修改。
Homebrew已由其他工作更新为Metal/core0.33.0，并包含Torch及MLX-LM开发版。其pip check因Torch nightly与core要求的2.14.0不一致而失败；官方说明也提到installer固定nightly。这是已有环境状态，不能仅据此断言MLX推理失败。此次没有修改或认证推理环境；应先隔离验证全stable兼容stack。需要模型的生成验证及remote CI继续Pending。
