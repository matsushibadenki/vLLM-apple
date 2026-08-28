# Optimizer Smoke Tests

## 2026-08-24 — GPT-2 MLX 4 bit

ローカルHugging Face cacheのGPT-2 FP32 safetensorsを、MLX 0.27.1 / MLX-LM 0.26.2、
Apple Silicon上で4 bit、group size 64へ変換した。

- source size: 525 MiB
- artifact size: 83,036,262 bytes、9 files
- elapsed: 3,490 ms
- peak child RSS: 1,034,600,448 bytes
- artifact tree SHA-256: `e6e8682763a8462621ecfe2a84f9359f5d286fc329e68b0cab9124d1d94b0ddc`
- result: MLX-LMによる再load成功、`Hello`から1 token生成成功
- resume: completed checkpointから変換を再実行せず、同一manifestを再利用

実行時にはoriginal snapshotを変更せず、失敗した試行ではartifactが公開されないことも確認した。
最初の失敗から、virtual environment launcherのsymlink保持と、MLX-LMが要求する非既存output
subdirectoryのworkspace正規化を実装へ反映した。

このsmoke testは変換とloadの互換性を確認するものであり、quantization前後の品質同等性を保証しない。
perplexityとdomain別生成品質の比較は次のquality gateで実施する。

## 2026-08-25 — GPT-2 multilingual perplexity gate

英語、日本語、简体中文のgeneral/science 6 slices、12 samples、420 scored tokensを使い、
baseline、4 bit、8 bitを別processで順次評価した。許容相対perplexity劣化は各slice 10%とした。

| Model | Artifact bytes | Aggregate perplexity | Peak evaluation RSS | Gate |
|---|---:|---:|---:|---|
| FP32 baseline | 548,105,171 | 34.9967 | 966,557,696 | reference |
| MLX 4 bit | 83,036,262 | 483.8420 | 567,033,856 | rejected |
| MLX 8 bit | 145,195,738 | 35.6934 | 625,639,424 | approved |

4 bitは全6 slicesで10%を超えたため拒否した。8 bitの最大劣化はgeneral/enの7.02%で、
全sliceがgateを通過した。この結果は小規模smoke datasetに限定され、generation quality、
long-context、code、mathematics、safety alignmentは未評価としてreportへ記録した。

## 2026-08-25 — GPT-2 deterministic generation gate

英語、日本語、简体中文のgeneral/mathematics 6 promptsを、固定seed、greedy、各16 tokensで
baselineと8 bit candidateへ別processで順次入力した。最低token一致率は各sample 70%、
期待条件scoreの許容悪化は0とした。

| Model | Elapsed | Peak evaluation RSS | Exact fingerprints |
|---|---:|---:|---:|
| FP32 baseline | 856 ms | 966,393,856 | reference |
| MLX 8 bit | 428 ms | 627,572,736 | 3 / 6 |

8 bitは英語2件が75%、日本語2件と中国語mathematicsが100%一致したが、中国語generalは
56.25%に留まり、全体gateはrejectedとなった。6件すべての期待条件scoreはbaseline/candidate
とも0であり、このdatasetは事実性や数学能力の保証には使えない。実行速度とpeak RSSの改善だけで
品質合格とはせず、より実用的な期待応答datasetとtask scoreを次の優先作業とする。

reportはpromptと生成文を保存せず、上限付きtoken ID、fingerprint、期待条件scoreのみを保持した。

## 2026-08-25 — GPT-2 selectable multilingual task suite

code、mathematics、retrieval smokeを英語、日本語、简体中文で各1件、計9件追加した。
短い正答は先頭一致で評価し、`8`が`18`へ含まれる場合を正答にしない。baselineと8 bitを
固定seed、greedy、各16 tokensで別process評価した。

| Model | Task score | Elapsed | Peak evaluation RSS | Gate |
|---|---:|---:|---:|---|
| FP32 baseline | 0 / 9 | 1,096 ms | 966,934,528 | reference |
| MLX 8 bit | 0 / 9 | 564 ms | 624,066,560 | rejected |

8 bitはtoken一致率70%基準で3件通過、6件不通過だった。両modelともtask scoreが0のため、
このGPT-2 smokeは実用能力の証明ではなく、非instruction modelを誤合格させないことの確認である。
未評価能力は実際のdomainから算出され、このrunでは`safety_alignment`のみとなった。

`--domain code`では3件だけを287 msで評価し、全suiteと異なるdataset fingerprintが生成された。
次は多言語instruction modelで正答baselineを確立し、実長contextへ段階的に拡張する。

## 2026-08-26 — Gemma 2 instruction model 8 bit gate

ローカルGemma 2 2B IT BF16へtokenizer固有chat templateを明示適用し、同じ9件をbaseline評価した。
その後、originalを変更せずMLX 8 bit、group size 64へ変換し、別processで同一条件を評価した。

| Model | Artifact bytes | Task score | Elapsed | Peak evaluation RSS | Gate |
|---|---:|---:|---:|---:|---|
| Gemma 2 2B IT BF16 | 約4.9 GiB | 3 / 9 | 10,172 ms | 3,065,266,176 | reference |
| Gemma 2 2B IT MLX 8 bit | 2,816,739,036 | 3 / 9 | 5,321 ms | 4,021,583,872 | approved |

8 bit candidateは9件すべてでbaselineとtoken IDが100%一致し、task scoreの悪化もなかった。
英語数学、中国語数学、英語retrievalが正答した。変換時peak child RSSは5,822,447,616 bytes、
所要時間は6,934 msだった。

評価速度とartifact sizeは改善したが、評価processのpeak RSSは約31%増加した。この小規模runだけで
メモリ効率改善を主張せず、長context時のKV cacheとload時peakを分離計測する必要がある。
template適用後の入力は20〜46 tokensで、4096 token budget内であることをreportへ記録した。
