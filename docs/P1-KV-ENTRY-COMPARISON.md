# P1 KV entry comparison — 2026-10-09

## 日本語

🟢 [Done] qualifierに`--prompt-cache-entries {1,2,3,4}`を追加し、既存MLX-LMのLRU entry上限を比較できるようにした。既定は4、byte上限256 MiB、モデル・kernel・prefill512・compactを維持。これは検証用の明示候補で、標準への自動昇格はしない。

比較は各設定で新worker、長文12・短文30・warmup3。bounded runnerを使用、異なる設定を同時に実行しない。runtime sources・runner・model identityの一致を確認する。単回比較なので速度優位・一般品質・P1長時間資格を認定しない。

Problem: byte上限削減は未完了だったため、LRU entry数による保持量／再利用のtradeoffを検証する必要があった。
Root cause: 保存entry数を減らすと以前の短文cacheが失われ、再利用率が落ちる。r11遅延／RSS未達の根因は未確定。
Evidence: 各設定のraw reportと同identity比較。
Changed files: qualify_gemma2_batch_mask.py、docs／証跡。
Change: 上流が既に対応しているentry上限を検証CLIで明示選択する。
Before: 検証CLI固定4。After: 1–4を選択、既定4維持。
CPU impact: kernel／実行処理の変更なし。総CPU削減未認定。
GPU impact: 小さいcacheでprefill再計算が増える可能性。GPU単独時間未測定。
Memory impact: LRU数値会計とRSSを別記。保持KV削減をRSS削減と同一視しない。
I/O impact: 既存reportと設定の記録のみ。
Energy impact: 未測定。
Correctness verification: 実モデルの品質・SLO・再利用率・保持量・identity・正常停止、既存回帰。
Regression risk: 過小cacheで応答時間が悪化するため、既定を変更しない。
Keep / Revert: 比較CLIをKeep。1 entryの標準採用はReject。

## English

🟢 [Done] The qualifier now explicitly selects 1–4 retained prompt-cache entries through existing MLX-LM functionality. Default remains four, with the 256 MiB byte limit and unchanged model/kernel. Fresh workers compare quality/SLO, reuse, retained KV accounting and RSS using the bounded runner. One trial per setting cannot qualify performance, general quality or P1 stability. The one-entry setting reduces retained KV accounting but loses short-prompt reuse; it is not promoted. Keep the diagnostic CLI, not a new production default.

## 简体中文

🟢 [Done] qualifier通过现有MLX-LM功能显式选择1–4个保持prompt-cache entry。默认仍为4、byte上限256 MiB，model／kernel不变。采用全新worker和bounded runner比较质量／SLO、复用、保持KV会计及RSS。每项单次试验不能认证性能、通用质量或P1稳定性。1 entry减少保持KV会计，但短文复用降低，不晋升默认配置。保留比较CLI，保持原默认。

## Results / 結果 / 结果

| entries | 最終LRU会計 MiB | 短文token reuse | 短文E2E mean / max ms | 長文E2E mean / max ms | 最終RSS bytes |
| --- | ---: | ---: | ---: | ---: | ---: |
| 4 | 48.344 | 87.38% | 424.798 / 484.070 | 600.792 / 3509.224 | 2,135,293,952 |
| 1 | 2.641 | 23.93% | 509.082 / 630.671 | 610.171 / 3516.547 | 2,210,398,208 |
| 3 | 33.109 | 87.38% | 477.797 / 713.808 | 630.125 / 3589.355 | 2,097,381,376 |

長文token reuseは全設定91.06%。3 entryはLRU会計を約31.5%削減し、観測token reuseを維持したが、短文mean／maxは悪化した。短文p95 bucket上限は4 entryで500 ms、1／3 entryで1000 ms。長文p95 bucket上限は全設定5000 ms。p99は各12／30件のnearest rankでmax bucketに入り、少数標本の参考値に限定。median／p95を実測の厳密quantileとして扱わない。rawにbucket分布を保存。

全設定の長文12・短文30で計126/126品質／SLO合格。正常停止・identity不変、全試験でruntime／runner／model一致。今回3 entryの最終RSSは低いが、単回・非atomicの最終sampleであり、RSS改善の効果やplateauと認定しない。1 entryはLRU会計を減らしても最終RSSが高く、KV会計だけで全体memory改善を判断しない。

🟠 [Next] 3／4 entryを順番を替えて独立反復し、tail latencyとRSS trendを比較する。既定4 entryを維持。3 entryは実測の保持量削減候補で、標準performance／P1資格ではない。長時間試験は09:00 JST、定期更新は停止したまま。

English: All 126 long/short quality/SLO checks passed, with unchanged and matching identities and clean shutdown. Three entries retained the observed reuse ratios while reducing final LRU accounting by about 31.5%, but short mean/max latency worsened; retain default four. Single-run RSS samples do not qualify memory savings. Short p95 histogram upper bounds were 500 ms for four entries and 1000 ms for one/three; long p95 was 5000 ms for all. Repeat alternating three/four workers before promotion.
简体中文：126件长／短文质量及SLO全部通过，identity不变且各试验一致、正常停止。3 entry保持观测复用率，最终LRU会计约减少31.5%，但短文mean／max恶化，保持默认4。单次RSS sample不认证memory节省。短文p95 histogram上限：4 entry为500 ms，1／3为1000 ms；长文均5000 ms。晋升前需交替独立重复3／4试验。

- [Raw comparison and hashes](evaluation/p1-kv-entries-comparison-2026-10-09.json)
- [4 entries](evaluation/p1-kv-entries-4-m4-2026-10-09.json), [1 entry](evaluation/p1-kv-entries-1-m4-2026-10-09.json), [3 entries](evaluation/p1-kv-entries-3-m4-2026-10-09.json)

Validation / 検証 / 验证：全回帰1486 tests（11 skip）、Ruff・diff check成功。今回CLI追加のみでruntime kernel変更なし。

Warmup：4 entry品質3/3・SLO2/3、1／3 entryは各品質3/3・SLO3/3。warmup未達を長文／短文126件の全件合格に含めて隠さない。English: Four-entry warmup missed one SLO; other warmups passed. 简体中文：4 entry warmup有1次SLO未通过，其余warmup通过。

Follow-up / 続報 / 后续：[反復campaign](P1-KV-ENTRY-REPEATS.md)では3-entry SLO未達。最終LRU snapshotはfault試験後のため、上表の差を固定正常負荷のmemory改善率として扱わない。
