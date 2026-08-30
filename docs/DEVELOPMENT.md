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

runnerにはnative arm64 Python、確認対象のvLLM-MetalまたはMLX LM environment、ローカルmodelを事前配置する。
workflowは1800〜21600秒の範囲だけを許可し、Apple Silicon、GPU core取得、memory pressure、version
matrixをpreflightで確認してからsampling/streaming probeとsoakを行う。`backend-kind=vllm_metal`では実際の
Metal platform選択も必須とし、`backend-kind=mlx_lm`ではMLX LM versionと明示architecture feature集合を検査する。
qualification reportにはsoakのthroughput/RSS安定性に加え、streaming phase probeのTTFT、TPOT、tokens/sec、
peak RSSをconstant-memory集計して含める。既定は3 sample、32 output tokensで、生成本文とraw timing列は保存しない。
認定後は`VLLMAppleQualificationCheck`が同じreportをSwift SDKのbounded readerで読み戻す。
workflowでは`--require-phase-profile`と`--require-memory-fit`を常に指定する。sample数、token数、
TTFT/TPOT、tokens/sec、peak RSS、resident estimateとhard ceilingの整合性が欠けるreportは、soakの`passed`がtrueでも拒否する。
さらに英語、日本語、简体中文の固定応答課題をstreamingで実行し、生成本文を保存せず言語別一致結果だけを残す。
`--require-quality-smoke`は3言語すべての成功と`stores_generated_text=false`をSwift側でも再検証する。
現在のself-hosted Qwen認定はtext-onlyを対象とし、`--require-text-only`でreportの`requested_modes=["text"]`を必須にする。
この検証が失敗した場合、Macアプリで履歴表示できない成果物としてworkflowを失敗させる。

大容量modelをrunnerへ配置する前に、artifactのdownloadサイズ、想定resident size、配置先を指定して
pre-download admissionを実行する。

```bash
vllm-apple artifact-admission \
  --model Qwen/Qwen3.8-Flash-Next \
  --artifact-gib 105 \
  --resident-gib 105 \
  --target models
```

判定は現在availableなUnified Memoryから、総memoryの8%または1 GiBの大きい方を緊急余白として除き、
配置先diskには既定でartifact sizeの105%をstaging領域として要求する。`eligible=false`は終了code 1、
入力またはfilesystem検査の失敗は終了code 2となる。このcommandはdownloadやmodel loadを行わない。
2026-08-30時点の32 GiB開発Macでは、105 GiB resident候補はmemory gateで拒否されるため、Qwenの
実model認定は十分なUnified Memoryを持つself-hosted runnerで継続する。

手動qualification workflowでも`artifact-size-gib`と`estimated-resident-gib`を対で指定できる。
指定時はbackend preflightとmodel loadより前に同じadmissionを実行し、結果をprivateな
`artifact-admission.json`として保存する。片方だけの指定、非有限値、16,384 GiBを超える値、または
memory/disk不適合はfail closedとする。`artifact-target`には実際にdownloadを配置するvolume上のpathを指定する。
入力を指定したrunではSwift checkerにも`--require-artifact-admission`を渡し、64 KiB以下のregular fileを
型付きdecodeする。Swift側でも各fit flagと`eligible`をbyte値から再計算するため、単なるtrue値の改変を
認定証拠として受け入れない。symlink、oversize、schema不一致、再計算不一致はevidence missingとして拒否する。
admissionの`model`はqualification reportの`model`と完全一致しなければならず、別候補で得た容量判定の
再利用を認定証拠として受け入れない。識別子は4 KiB以下の表示可能文字列に限定する。

PRやpushからは起動せず、同時に複数の実modelを走らせない。report directoryは0700、reportは必要な
場合だけ明示的な`upload-report`入力で14日間保存する。reportには生成本文を含めないが、model識別子を
含むため公開可能性を確認してからuploadする。
