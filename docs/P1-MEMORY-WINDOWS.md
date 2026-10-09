# P1 memory-window audit — 2026-10-09

## 日本語

🟢 [Done] 診断windowにRSS変化とallocatorのactive／cache変化を独立した数値として保存する。既存snapshotを再利用し、追加HTTP・OS取得・GPU同期なし。既存の64件上限、PID一致・counter reset拒否・欠測明示を維持。allocator欠測でも有効なprocess差分は保持し、allocatorだけavailable=falseとする。

既存2試験のfirst／last windowをofflineで監査した。保持された4 windowは全部page-in増分0。最初の短文windowのRSSはそれぞれ+9,977,856／+12,992,512 bytesだが、allocator activeは+30,113,792／+851,968 bytes、cacheは+33,275,216／+58,377,660 bytes。同じworkloadでもこれらは同一量ではない。
一方の最後の長文windowではRSS −1,228,800 bytesに対しactive +342,163,054 bytes。RSSをactive＋cacheの和としたり、その差を「その他memory」としたりしない。非atomic snapshot、resident memoryとallocatorの契約差、sample時刻の違いを含む。allocator activeはKV専用ではない。これだけでleak、圧縮、再allocation、r11根因を認定しない。

Problem: RSS増加とallocator保持を直接比較する診断が不足。
Root cause: RSSとallocatorは異なる観測量。P1未達の原因自体は未確定。
Evidence: [2試験のoffline監査](evaluation/p1-memory-window-audit-2026-10-09.json)。元JSONは保持し、SHA256で出典を記録。
Changed files: process_activity_delta.py、qualify_gemma2_batch_mask.py、audit_memory_windows.py、対象tests。
Change: independent memory deltaと再現可能なoffline監査。
Before: process活動差分のみ。After: allocator active／cache差分を併記。
CPU impact: 定数個のcounter比較・辞書作成。速度改善率は未測定。
GPU impact: 追加call・同期なし。今回GPU試験を追加実行していない。
Memory impact: 各windowに固定数の数値追加、保持上限64。RSS削減未認定。
I/O impact: 追加HTTP／OS callなし、既存reportへ数値を追加。
Energy impact: 未測定。
Correctness verification: RSS減少とactive増加の同時保持、allocator欠測をゼロにしない、既存PID／reset検証。
Regression risk: collector追加fieldのみ。既存品質／資源gateを緩めず、古いreportの資格を変えない。
Keep / Revert: 診断としてKeep。速度・メモリ改善の認定ではない。

🟠 [Next] 実KV保持量も同workloadで照合し、解放候補を実測で選ぶ。長時間試験は09:00 JST開始。定期更新は停止したまま。

## English

🟢 [Done] Independently retain RSS, allocator-active and allocator-cache changes in existing bounded workload windows, with no extra HTTP, OS or GPU calls. Missing allocator values remain explicit while valid process deltas survive. An offline auditor preserves source hashes and inspects retained first/last windows from two existing trials. RSS and allocator changes disagree, including RSS decreasing while active memory grows. They are non-atomic measurements with different meanings; do not add them as RSS categories or infer KV/leaks from their difference. No GPU rerun, speed, RSS savings or r11 causality claim. 🟠 [Next] Compare actual KV retention and test one measured release candidate. Long campaigns start at 09:00 JST; recurring updates remain stopped.

## 简体中文

🟢 [Done] 在现有有界window中独立保存RSS、allocator active及cache差分，不增加HTTP、OS或GPU调用。allocator缺失明确标记，仍保留有效process差分。离线工具审核2个旧试验的first／last window并记录SHA。RSS及allocator差分并不一致，也出现RSS下降而active上升。它们是不同含义的非atomic测量，不能相加作为RSS分类，也不能由差值推断KV／leak。本次没有追加GPU试验，不认证速度、RSS节省或r11原因。🟠 [Next] 对照实际KV保持量，实测选择一个释放候选。长时间测试09:00 JST，定期更新保持停止。

```sh
PYTHONPATH=. .venv/bin/python scripts/audit_memory_windows.py REPORT.json
```

検証 / Validation / 验证：対象3 tests、全回帰1483 tests（11 skip）、Ruff・diff check成功。[Receipt](evaluation/p1-memory-windows-validation-2026-10-09.json). 新collector fieldの実GPU確認は次の短時間試験で行う。Offline差分は既存raw snapshotで検証済み。
