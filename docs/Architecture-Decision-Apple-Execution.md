# Architecture Decision: Apple Execution Plane

Status: Accepted — 2026-08-27

## Context

現在のコードベースは、vLLM-Metalを安全に運用するcontrol plane、memory policy、API、Swift統合を
主に実装している。これは必要な基盤だが、tokens/secの上限はexecution backendにも依存する。

## Decision

1. Control Plane、Execution Plane、Apple Runtimeを独立した責務として扱う。
2. `AppleExecutionPlanner`がhardware、model architecture、workload phase、memory pressure、thermalを
   入力に、versioned execution planを生成する。
3. memory計算をKV専用型から`StateMemorySpec`へ一般化し、SWA、MLA、recurrent、hybrid modelを扱う。
4. prefillとdecodeを別phaseとしてprofileし、batch、precision、kernel、device assignmentを分ける。
5. 初期実行backendはvLLM-MetalとMLXを優先し、process isolationを維持する。
6. Paged Attentionを含む既存機能は再実装せず、計測済みの不足にだけnative Metalを追加する。
7. quantizationはMLX互換形式を先に利用し、次にdequantizeとGEMM/GEMVのfusionを検証する。
8. CPU/GPU/ANE異種実行、adaptive context compression、speculative executionは、品質と
   end-to-end性能のgateを通過した場合だけplannerが選択する。

## Non-goals

- NVIDIA向けvLLM構造をApple Silicon上にそのまま再現すること
- vLLM-Metalを早期にforkして独自Paged Attentionを重複実装すること
- staticなSoC世代名だけで最適kernelやdevice placementを決めること
- control planeの改善をkernel throughputの改善として報告すること

## Staged acceptance

1. Planner schemaとdry-runが同一入力に決定論的なplanと理由を返す。
2. `StateMemorySpec`がstandard Transformerと少なくとも一つの非標準state architectureを表現する。
3. prefill/decode別profileがTTFT、TPOT、peak memoryを記録する。
4. plannerを既存context、scheduler、elastic memory policyへ接続する。
5. native kernelまたは異種実行は、correctness gateと代表workloadのend-to-end比較を通過する。

詳細仕様と進捗は[Design-Specifications.md](Design-Specifications.md)と[ROADMAP.md](ROADMAP.md)を参照する。
