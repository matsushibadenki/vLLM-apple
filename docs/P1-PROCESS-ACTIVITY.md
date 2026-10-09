# P1 process activity diagnostics — 2026-10-08

## 日本語

[Done] RSSだけでは分からないprocess全体のpage fault、actual page-in、copy-on-write fault、context switch、thread数／running thread数をP1 resources snapshotに追加。Darwin PROC_PIDTASKINFOの公開96-byte ABIを使用。既存RSS取得のABI／単位／fallbackは維持し、追加診断はprofile時のみ。OS API未対応・失敗・負のcounterはavailable=falseと理由を返し、欠測をゼロで補わない。

sourceの構造定義は[Apple公式proc_info.h](https://raw.githubusercontent.com/apple-oss-distributions/xnu/main/bsd/sys/proc_info.h)で確認。これらはprocess全体の累積counterであり、GPU時間、圧縮memory量、host全体のmemory pressure、遅延原因を表す値ではない。比較は同PID／同epochの有効な前後sampleだけを使い、counter減少やPID変更を跨いで差分を認定しない。

M4／Gemma-2-2b-it-4bit／compact／prefill512の短時間実機試験はwarmup3/3・長文12/12・短文30/30の品質／SLO合格。正常停止、実行中runtime/model identity不変。sampleはRSS 2,214,166,528 bytes、fault225,057、page-in79、context switch65,547、thread24／running1。累積値1点なのでr11の原因やresource plateauを認定しない。

Problem: RSS・allocatorとthread CPUだけではprocessのfault／scheduling活動が欠測だった。
Root cause: P1遅延とRSS未達の原因は未確定。
Evidence: raw実機reportとOS APIの合成負荷測定。
Changed files: process_memory.py、mlx_gemma2_compat.py、test_process_memory.py。
Change: 既存公開ABIのcounterを明示し、profile snapshotへ追加。
Why: 推測によるruntime最適化を避け、同epochの原因比較に必要な観測を揃える。
Before: RSSのみ。After: RSSに加えて累積fault／page-in／context switch。
CPU impact: sampler自身の1,200回測定、中央値1.792 µs、p95 2.000 µs、p99 2.167 µs、最大67.334 µs。P1 resources取得時のみ追加。通常経路にOS call追加なし。
GPU impact: 呼出・同期変更なし。
Memory impact: 固定サイズのOS structと数値snapshot。RSS改善は未認定。
I/O impact: subprocess起動ゼロ、本文／tensor転送なし。
Energy impact: 未測定。
Correctness: named ABIと96-byte size、native RSS互換・別buffer、portable未対応／partial read欠測を検証。対象8 tests、全回帰1480 tests（11 skip）、Ruff・diff check成功。
Regression risk: OS ABI依存はDarwin限定、失敗時は追加counterの欠測を明示。既存RSSの失敗判定・gateを緩めない。
Keep / Revert: 診断としてKeep。速度・RSS・長時間認定として扱わない。

[Next] 同PID／同workloadの前後counter差分とeval遅延を照合し、RSSの再増加を観察。長時間campaignは09:00 JST開始。
[Pending] 電力samplerまたは外部meterによるjoule/request。

## English

[Done] P1 resource snapshots now include optional process-wide fault, actual page-in, copy-on-write, context-switch and thread counters from Darwin's public 96-byte task-info ABI. Existing RSS semantics and fallback are preserved. Unsupported, failed or negative counters remain explicitly unavailable. Standard operation adds no OS sampling.

Real M4/Gemma 2B compact/prefill512 checks passed 45/45 quality/SLO, unchanged run identities and clean shutdown. All 1480 regression tests passed, 11 skipped. Native diagnostic sampling cost had median 1.792 µs, p95 2.000 µs and p99 2.167 µs over 1200 reads, with no subprocesses. A single cumulative snapshot cannot establish r11 causality, compression, system memory pressure or RSS stability. Compare valid samples only within the same PID/epoch; resets are not negative activity. [Next] Correlate same-workload counter deltas and eval waits; long campaigns start at 09:00 JST. [Pending] Electrical energy measurement.

## 简体中文

[Done] P1 resources增加Darwin公开96-byte task-info ABI的process累计fault、实际page-in、copy-on-write、context switch及thread counter。保持现有RSS单位、fallback和gate。未支持、失败或负counter明确标记不可用，不填零；标准路径不增加OS采样。

实际M4/Gemma 2B、compact/prefill512测试45/45质量及SLO通过，identity不变、正常停止。全回归1480 tests通过，11 skip。1200次native读取，中位耗时1.792 µs、p95 2.000 µs、p99 2.167 µs，无subprocess。单个累计snapshot不能认证r11原因、memory压缩、系统memory pressure或RSS稳定性。只比较相同PID／epoch内有效sample，不能跨reset计算负活动。[Next] 对照同workload的counter差分与eval等待，长时间测试09:00 JST开始。[Pending] 实际电能测量。

## Evidence

- [Real-model report](evaluation/p1-process-activity-m4-2026-10-08.json)
- [Native sampler measurement](evaluation/p1-process-activity-overhead-2026-10-08.json)
- [Validation receipt](evaluation/p1-process-activity-validation-2026-10-08.json)
