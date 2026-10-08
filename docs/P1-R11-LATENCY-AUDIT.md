# P1 r11 latency audit — 2026-10-08

## 日本語

[Done] `scripts/audit_p1_latency.py`で失敗要求の開始→初回token時刻と、前後snapshotの遅いscheduler stepを照合。
同じstepの重複を除き、要求時間との交差部分だけを集計した。
[監査結果](evaluation/p1-r11-latency-overlap-2026-10-08.json)にsource／入力hashを保存。
17件すべての失敗sampleを取得。対象2 testsとRuffは成功。

17件中15件は19〜22 tokenの短文で、18〜21 tokenがcache再利用されていた。
残る2件は2,073 tokenの長文でcache再利用4 token。
15件で遅いstepとの時間重複を観測したが、初回応答約8.9秒の2件では保持sampleとの重複はなかった。
長文2件では重複部分が約9.60秒／9.87秒で、初回応答約11.84秒／10.32秒。
短文の初回応答約20秒の4件では保持stepとの重複は約1.76〜3.08秒にとどまる。
初回失敗window前にはload average 8.74／logical CPU 10、thermal fair、電源ACを観測。

これらは同時刻の観測であり、要求へのphase帰属や遅延の因果証明ではない。
bounded履歴に残っていないstep、250 ms未満のstep、queue・tokenize・HTTP書込等の時間は補完しない。
CPU wall差をGPU時間、sample不在をidle、累積queue最大値を当該要求のqueue時間として扱わない。
clientとbackendのwall clockが安定していることを前提とした診断で、時刻同期の認定はしない。

[Next] requestごとのqueue／tokenize／初回出力／HTTP書込の計測を追加して、短文で未説明の遅延を分離する。
その後に支配時間を確認して最小修正し、同条件の短時間比較を行う。性能改善・根因解消は未認定。
長時間campaignは09:00 JST開始を維持する。

```sh
python3 scripts/audit_p1_latency.py docs/evaluation/p1-stability-m4-2026-10-08-r11/30min.json --output /tmp/p1-latency-new.json
python3 -m unittest tests.test_p1_latency_audit -v
```

出力は未作成pathを指定する。runtime・依存・GPU処理は変更していない。

## English

[Done] Added an offline overlap audit for all 17 retained failures, deduplicating shared step samples and clipping time intersections.
Fifteen failures used short, nearly fully cached prompts; two used long prompts. Fifteen failures overlapped retained slow steps,
but two roughly 8.9-second TTFT failures did not. Missing retained samples do not imply idle time.
Shared step overlap is not request attribution; CPU/wall differences are not GPU timing. Two tests and Ruff passed.
[Next] Measure per-request queue, tokenization, first output and HTTP writes before changing the runtime. No root-cause or speed qualification yet.

## 简体中文

[Done] 为全部17个保留失败sample增加离线时间重叠审核，去重step并仅累计交叉部分。
15次失败是几乎全部cache复用的短文，2次是长文。15次与保留的slow step重叠，2次约8.9秒TTFT没有重叠。
sample缺失不等于idle；共享step重叠不代表请求归属，CPU／wall差值不是GPU时间。2项测试及Ruff通过。
[Next] 先测量每个请求的queue、tokenize、首个输出及HTTP写入，再修改runtime。根因和速度提升仍未认证。
