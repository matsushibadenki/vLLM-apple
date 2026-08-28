# Development and CI

## Reproducible setup

control planeのruntimeはdependency-freeである。build backendと開発toolを完全version固定し、既定
Pythonは`.python-version`で3.12とする。test runnerはstandard libraryの`unittest`を使用する。

```bash
make bootstrap
make check
```

`make check`はPython 189 test、compile、Ruff、Swift SDK test、SwiftUI Mac sample buildを順に実行する。
MLX、vLLM、vLLM-Metalを必要とする実device試験は通常unit CIへ混ぜず、version matrixとpromotion gateを
通したApple Silicon環境で実行する。

## Continuous integration

GitHub Actionsは次をfail-fast無効・job timeout付きで実行する。

- Ubuntu 24.04：Python 3.10、3.12、3.13の全unit/integration test、compile、Ruff
- macOS 15：Swift SDK test、SwiftUI Mac sample build

Ruffは初期CIではsyntax、undefined name、基本的なimport/runtime errorに範囲を固定する。既存全fileの
style rewriteを機能変更へ混ぜず、追加ruleはcodebaseを段階的に整備してから有効化する。

workflow権限はrepository contentsのread-onlyとし、同一branchの古いrunはconcurrency groupで中止する。
Metal device、実model、30分soak、memory pressure試験はhardware qualification trackで扱う。

## Apple Silicon qualification runner

`.github/workflows/metal-qualification.yml`は手動実行専用で、次のlabelを持つself-hosted runnerだけを
対象にする。

```text
self-hosted, macOS, ARM64, vllm-metal
```

runnerにはnative arm64 Python、確認対象のvLLM-Metal environment、ローカルmodelを事前配置する。
workflowは1800〜21600秒の範囲だけを許可し、Apple Silicon、GPU core取得、memory pressure、version
matrix、実際のMetal platform選択をpreflightで確認してからsampling/streaming probeとsoakを行う。
認定後は`VLLMAppleQualificationCheck`が同じreportをSwift SDKのbounded readerで読み戻す。
この検証が失敗した場合、Macアプリで履歴表示できない成果物としてworkflowを失敗させる。

PRやpushからは起動せず、同時に複数の実modelを走らせない。report directoryは0700、reportは必要な
場合だけ明示的な`upload-report`入力で14日間保存する。reportには生成本文を含めないが、model識別子を
含むため公開可能性を確認してからuploadする。
