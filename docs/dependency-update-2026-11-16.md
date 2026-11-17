# Stable dependency maintenance — 2026-11-16

## 日本語

🟢 [Done] Ruff 0.17.4 → 0.17.5。開発依存とlockを同期し、Python 3.10ターゲットとlint設定を維持。[公式リリース](https://github.com/astral-sh/ruff/releases/tag/0.17.5)は11月12日公開の安定版。プレビュー規則は有効化していない。setuptools 84.2.1、PyInstaller 6.22.3は最新のため維持。

隔離コピーでlint、Python 3.10／3.12 compileall、pip check、wheel作成・インストール・CLI起動、Swift SDK 38件、Macサンプルビルドが成功。Python回帰は1,633件実行し失敗なし（46件skip）。PyInstaller配布ZIPは再作成していない。推論・P1・Strata試験の競合は確認されず、既存の未追跡評価資料はそのまま保持。

⭕️ [Pending] Homebrew Metal 0.33.0の既存Torch不一致（要求2.14.0、実際2.16.0.dev20261013）は継続。MLX 0.32.4は最新だが、MLX-LMは固定コミット。Transformers 5.23.0はMetalの上限（5.21未満）外。Torch 2.15.0、NumPy 2.5.5、MLX-Gen 0.38.0、Core ML tools 9.0、Diffusers 0.43.0、mflux 0.24.0は対象環境の実モデル検証まで保留。tokenizers 0.23.3とtorchao 0.18.0は上流最新版。実行環境と既定設定は変更せず、doctor・実モデル認定は今回未実施。Swiftの外部依存なし。CI Actions v7候補は実ランナー未検証のため維持。[前回記録](dependency-update-2026-11-09.md)の制約を引き継ぐ。

## English

Updated Ruff to 0.17.5. Lint, compilation, wheel/CLI, 38 Swift tests and the Mac sample build passed. Python ran 1,633 tests with no failures and 46 skips. Inference environments remain unchanged pending qualification; the existing Homebrew Torch pin mismatch persists.

## 简体中文

Ruff 已更新至 0.17.5。Lint、编译、wheel／CLI、38 项 Swift 测试及 Mac 示例构建均通过。Python 运行 1,633 项测试，无失败，跳过 46 项。推理环境保持不变，等待验证；Homebrew 的 Torch 版本约束仍不一致。
