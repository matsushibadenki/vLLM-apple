# Strata integration — 2026-10-10

## 日本語

🟢 [Done] 第11工程：`grouped=True`の明示候補でtop-kの全Leaseを保持し、
未集約Expert出力を1回のmx.evalで確定してから解放する。既定は逐次実行。
entry／byte予算不足なら逐次fallback。byte判定はfile header込みの保守的な見積り。
source形式はdense／affine 4／8-bit、Routerと最終集約を維持する。
grouped時telemetry sampleはtop-k group単位、selected_experts数は維持する。
native含む11 tests成功、dense／4／8-bit数値一致、entry／byte上限fallback、失敗時解放を確認。

[同一sourceの比較](evaluation/strata-grouped-analysis-cpu-2026-10-10.json)：
逐次とgroupedの全Expert事前常駐CPU比較18 trial全件数値一致。
8要求のExpert eval回数は16→8。
再利用中央値（μs）はLRU 160.000→152.500、cost 183.291→185.125。
速度改善は一貫せず、run変動5%超、baselineより遅いため採用しない。
全Expert常駐でevictionがないので、このLRU/cost差を退避方式の優劣とは扱わない。
各campaignは別時刻の取得であり、interleaved取得・実モデル・GPU・RSS・P3資格は未検証。

🟠 [Next] Router host取り出しとPython／MLX graph発行の固定費を調査し、
常駐Expertをまとめたgather演算への接続候補を検討する。同期削減だけで高速化したとはしない。

🟢 [Done] 第10工程：任意有効化の固定13 phaseカウンターを追加し、
checksum read/hash、file stat、MLX load/weight eval、Router eval/to_host、Expert build/eval、
acquire/release、出力assembly/evalを分離した。標準設定は計測無効。
[分解結果](evaluation/strata-phase-analysis-cpu-2026-10-10.json)：profile、計測off、
全Expert事前常駐profileを各9 fresh process、合計27 trial取得し全件数値一致。
再利用時の3 trial中央値（μs）は次の通り。

| 条件 | baseline | LRU | cost_frequency |
|---|---:|---:|---:|
| 計測off | 64.959 | 226.333 | 177.250 |
| 分解計測on | 80.250 | 634.708 | 415.292 |
| 全Expert事前常駐・計測on | 142.542 | 323.166 | 247.625 |

計測on/offは別時刻の取得でhost条件と混ざるため、この差を計測overheadの純粋な値とはしない。
各runの総時間変動は5%を超え、厳密な寄与率や方式優位は認定しない。
事前常駐では全candidate trialで要求中miss増分とchecksum/load/weight-evalのcall数が0。
それでもbaselineを上回る遅延が残るため、I/Oだけが遅さの原因ではない。
8要求合計のphase中央値（LRU／cost、μs）は次の通り。

| phase | 分解計測on | 全Expert事前常駐 |
|---|---:|---:|
| acquire（読み込み内包） | 2507.84 / 1452.12 | 57.96 / 44.96 |
| checksum read | 295.45 / 215.92 | 0 / 0 |
| checksum hash | 39.71 / 27.08 | 0 / 0 |
| MLX load | 680.37 / 213.79 | 0 / 0 |
| weight eval | 796.67 / 373.96 | 0 / 0 |
| Expert eval | 1536.46 / 677.91 | 874.04 / 586.96 |
| Router eval | 452.58 / 197.04 | 313.79 / 187.79 |
| Router to_host | 396.33 / 154.79 | 253.08 / 157.46 |

中央値はphaseごとに算出し、加算してrequest総時間にしない。acquire内のload等は重複。
Router evalはgate/top-k実行を含み、純粋な同期待ちではない。MLX内部I/Oはload/evalと分離できず、
checksum readはOS cacheを含むhost read。物理SSD cold I/OやGPU counterは測定外。

現時点の原因判断：hash計算より読み込み・評価が大きく、I/Oを除いてもExpert単位の
逐次evalとRouter host取り出しの負担が残る。これは有力な構造要因であり寄与率の確定ではない。
🟠 [Next] top-k全体のLeaseを保持して評価境界をまとめる候補を、同じ数値suiteと計測で比較する。
小容量fallbackと大容量出力上限を維持し、改善が測定ノイズを超えるまで既定採用しない。
関連17 tests・native含む10 tests・Ruff成功。実Qwen品質／GPU／chat E2Eは未認定。

🟢 [Done] 第9工程：`scripts/compare_strata_experts.py`でbaseline／LRU／cost_frequencyを
各3 fresh process、順序をrotationして比較する合成CPU runnerを実装。
[測定記録](evaluation/strata-synthetic-qwen3-cpu-2026-10-10.json)は9 trialすべて数値一致。
初回の3 trial中央値はbaseline 0.194 ms／LRU 1.266 ms／cost 0.703 ms。
後続7呼び出しの中央値を各trialで取り、その3 trial中央値は0.135／0.336／0.213 ms。
hit/miss、常駐量、close後の0 byte、source SHAを保存する。
export・初期化は計測外、OS file cacheはresetしない。baselineは事前に評価済みsource重み、
候補は空の常駐cacheから開始する。絶対値は小さい合成blockのCPU時間で、モデルtokens/sではない。
cost方式もbaselineより遅いため採用せず、P3資格・自動適用はfalse。
量子化品質suite、RSS、GPU、chat E2E、電力、安定性は測定していない。

```sh
/path/to/mlx-python scripts/compare_strata_experts.py --output /tmp/new-strata-report.json
```

🟠 [Next] I/O・checksum・load・host Router同期を個別計測し、実互換weightを使った
メモリ適合性と性能のtradeoffを比較する。合成比較の改善率を実モデルへ外挿しない。

🟢 [Done] 第8工程：`install_qwen3_moe_residency`でMLX-LMの標準
Qwen3MoeSparseMoeBlockのswitch_mlpのみを明示置換する参照hookを実装。
gate、softmax、top-k、norm_topk_prob、最終集約は既存blockに任せる。
hookは未集約Expert出力を元の形状で返す。各Expertを逐次実行するためtop-k全体を
同時常駐させず、1 entry予算でも実行できる。Router結果のhost同期を伴い、速度向け設計ではない。
manifest全Expertの存在とsource量子化形式を確認してから置換する。
hookは元SwitchGLUを保持しない。別のcallerがsource参照を保持している場合や
allocator内部cacheについては解放を保証しない。baseline復帰は元モデル再読み込みで行う。
要求開始前にinstallし、phaseはsafe pointで明示指定する。自動phase推定やdaemon既定採用は行わない。
最大4096 token、未集約出力64 MiB上限。telemetryはExpert演算ごとのsample。

実MLXを含む12 tests成功。合成Qwen3ブロックのdense／4／8-bitとtop-k正規化
あり／なしの6構成で元blockとの誤差1e-5未満、gate identity維持、元SwitchGLU
weakref解放、1 entry動作、phase記録、重複Router拒否を確認。
これはQwen3の実装構造への接続検証でありQwen3.8/Qwen4Exp対応の証明ではない。

```python
hook = install_qwen3_moe_residency(
    sparse_moe_block, backend, executor, layer=layer_index, phase="prefill",
)
# Requests must be drained before changing execution phase.
hook.set_phase("decode")
```

🟠 [Next] 互換モデルの実weightでbaseline／LRU／cost_frequencyを比較し、
ファイルI/O・host同期・decode latencyを測定する。標準採用は既存P3判定の後。
⭕️ [Pending] 現在metadataのみのQwen3.8/Qwen4Exp実weightと専用architecture対応。

🟢 [Done] 第7工程：標準MLX-LM SwitchGLU/SwiGLUから独立Expertを書き出す
`export_switch_experts`を実装。denseまたは同一設定のaffine 4／8-bitに限定し、
Linear bias・独自activation・混在形式を拒否する。layer番号から全Expertを列挙し、
新規directoryへ保存して最後に完成manifestを公開する。既存directoryは拒否する。
失敗時は部分artifactを残し、完成manifestは作らない。途中結果はfrom_manifestで利用できない。
exportはoffline処理でありsource重みを保持する。runtimeで元SwitchGLUを残したまま
置換してもメモリ削減にならないため、runtime所有権の切り替えは別工程で扱う。
tensor／file／合計file容量を検査するが、file容量の判定は保存後で、source／allocator overheadは上限外。
実MLXを含む11 tests成功。合成dense／4／8-bit SwitchGLUからexport→manifest検証→
token別実行を行い、元SwitchGLUのRouter集約出力との差1e-5未満を確認した。
これは実APIでの合成構造検証であり対象Qwenの出力品質・速度の認定ではない。

```python
export_switch_experts(
    {layer_index: switch_glu}, new_directory, model_sha256=model_digest,
    maximum_expert_bytes=expert_limit, maximum_total_bytes=artifact_limit,
)
```

🟠 [Next] architectureを限定したモデル接続と、source重みのruntime所有権を移す手順。
⭕️ [Pending] Qwen対象weight取得後のE2E・品質・LRU比較。

🟢 [Done] 第6工程：`expert-manifest.json`の検証と`from_manifest`を実装。
schema_version=1、model_sha256、quantization_bits、group_size、expertsを要求する。
expertsはlayer・expert・size_bytes・sha256の配列。ファイル名は既存規則から生成し、
manifestは4 MiB／65,536 entry以内、重複field／entryを拒否する。
呼び出し側のexpected_model_sha256と一致させ、cache miss時の読み込み前にサイズと
SHA256を1 MiB単位で検証する。量子化設定はmanifestから取得する。
明示設定の旧constructorは合成試験用途として残す。実接続ではfrom_manifestを使う。
実MLX演算を含む10 tests・関連15 tests・Ruff成功。
チェックサムは変更検出であり署名や品質認定ではない。model_sha256の算出規則は
生成側と利用側で同一に固定する必要がある。trusted offline artifactが前提で、
検証とmx.loadの間の敵対的変更や既に常駐したentryの再検証は扱わない。

```python
backend = MLXFileExpertBackend.from_manifest(
    expert_root, expected_model_sha256=model_digest, maximum_file_bytes=file_limit,
)
```

🟠 [Next] 実モデルから独立Expertを生成するexportとarchitecture別Router接続。
⭕️ [Pending] 対象Qwenの実weightによる品質／E2E比較。

🟢 [Done] 第5工程：独立Expertファイルの4-bit／8-bit affine量子化を追加。
`quantization_bits`と`group_size`を明示し、各gate/up/downにweight(uint32)、
scales、biasesを要求する。量子化biasesはaffine補正値であり、Linear biasではない。
packed容量、scale形状、入出力次元を検査し、`mx.quantized_matmul`で重みを展開せず演算する。
resident bytesにはpacked weight・scales・biasesを含む。
実MLX CPUテスト6件成功。4／8-bit × group32／64の合成重みについて、
同じ量子化重みのdequantize参照との誤差1e-6未満と不一致groupの拒否を確認。
この一致は元の非量子化モデルに対する品質保証ではない。
group128はAPI候補として受理するが今回の数値検証範囲外。
関連17テスト中13件成功、native必須4件は実MLX環境で成功。Ruff成功。
🟠 [Next] bits／group／model identityを結ぶartifact manifestと実Router接続。
ファイル単体にはbits／groupの自己記述がないため、信頼できる生成手順と明示設定を前提とする。
速度、量子化による品質変化、実Qwen／GPU推論は未認定。

🟢 [Done] 第4工程：`execute_routed`でtoken別の複数Expert選択と重み付き集約を実装。
最大4096行、各行のtop-kを逐次実行し、全Router行を読み込み前に検証する。
選択順と未正規化の重みをそのまま使い、top-kの再計算はしない。
実MLX CPUテスト4件成功：異なるtoken選択、2 Expert集約、退避、数値誤差1e-6未満、
不正Routerの事前拒否、入力次元不一致時のLease解放を確認。関連CPUテストは12件成功、
native必須3件は別の実MLX環境で成功。Ruff成功。
本経路は参照用逐次実行であり、実モデルRouterへの自動接続や量子化、速度改善は未実装／未認定。
🟠 [Next] 量子化Expertの明示形式と数値比較。実モデル接続は対応architectureを限定する。

🟢 [Done] 第3工程：`MLXFileExpertBackend`を実装。Expertごとの独立safetensorsから
浮動小数点・biasなしSwiGLUのgate/up/downを読み込み、形状検査とmx.eval後に常駐管理へ渡す。
解放時はtensor参照を破棄する。allocatorがOSへ直ちにメモリを返すことは保証しない。
共有SwitchGLU配列のsliceをcacheしても元配列が残るため、その方式は採用しない。
各token groupで共通のRouter選択を保持して重み付き集約を行い、結果のmx.evalをLease解放前に完了する。
入力のRouter選択自体はbackend callerが渡す。モデルへの自動hookは未実装。

実MLX CPU配列で小さな独立Expertファイルを生成し、退避／再読み込み後の参照計算との
誤差1e-6未満・Lease解放・byte計上を検証した。これは合成重みの演算検証でありモデル品質や速度認定ではない。
sandboxではMetal device unavailableとなり、Metalにアクセスできる環境で成功した。
ローカルの`qwen3.8-flash-next-metadata`はconfigのみで、実モデルのExpert重みは未確認。
⭕️ [Pending] この対象モデルのE2E比較には実weightと量子化／architecture adapterが必要。
🟠 [Next] 複数Expert・量子化形式・実Routerの接続を個別に検証する。
独立ファイルは信頼できるoffline生成物を前提とし、読み込み前のfileサイズ上限は
MLX allocatorの厳密なpeak上限を保証しない。大容量sourceをそのまま読み込まない。

🟢 [Done] 第2工程：`ResidentExpertExecutor`でRouter選択と常駐管理を接続。
選択順・重みを保持し、全ExpertのLeaseを取得してからbackendの同期consumerを呼ぶ。
consumerはGPU結果をmaterializeしてから返す契約。lazy結果の完了は本接続部で保証しない。
途中の取得失敗・演算失敗では取得済みLeaseを解放し、成功時だけphase別telemetryへ記録。
cache hitは各acquireで返したLeaseから取得するため、共有カウンターの差分に依存しない。
関連22 tests・Ruff成功。テストbackendでの接続検証であり実MoE推論や高速化の証明ではない。
既存runtime内で実モデル用load_expert／release_expert実装は確認できていない。

🟢 [Done] 最初の工程として既存ExpertResidencyManagerへ明示選択の
`eviction_policy="cost_frequency"`を追加した。既定は`lru`。
保持スコアはアクセス回数 × 同期load_expert呼び出し時間 ÷ resident bytes ÷ 最終使用からの論理経過時間。
低いスコアから退避し、同点は最終使用時刻・ExpertKeyで決定する。
統計は常駐entry内だけに保持し、回数は100万で飽和する。退避後の履歴は保持しない。
Lease中のentryは候補から除外し、既存のbyte／entry上限と遅延resizeを共有する。
Router選択やモデルの重みには手を加えない。

load時間はホスト側呼び出し時間であり、MLXの遅延GPU実行やSSD missの実時間を
保証しない。backendは実際に利用可能なresourceを返す必要がある。
このスコアは比較候補で、最適性や速度改善の証明ではない。

🟢 [Done] 関連31テスト成功。高コストentryの保持、Lease保護、遅延予算変更、
未知policy拒否、既存LRUと投機実行・P3判定を検証した。
この工程では本番設定変更・GPU負荷・依存更新を行っていない。

🟠 [Next] 実MoE backendのRouter観測とExpert resource操作を確認し、
prefill／decode別の測定を収集する。LRUと同一モデル・精度・workloadで独立3回以上比較し、
既存P3判定へ接続する。品質、memory、TTFT／TPOT、P1／対象P2前提を満たすまで標準採用しない。

🔴 [Later] コストに基づく先読み制御、互換draft／Prompt Lookup、KV圧縮と融合演算。
⭕️ [Pending] 対応artifactと実機で検証できないMTP／ANE経路。

参照設計：https://github.com/Niko1221/Strata/blob/main/docs/HOW_IT_WORKS.md
直接のソース移植は行っていない。将来移植時は参照commitとlicenseを固定する。

## English

🟢 [Done] Opt-in grouped top-k pins all leases through one expert-result eval;
insufficient entry/byte budgets retain sequential fallback. Eleven native-related
tests pass including dense/4/8-bit parity and cleanup. Eighteen preloaded CPU
trials halve eval calls (16→8) but show inconsistent latency change:
LRU 160→152.5 us; cost 183.291→185.125 us. Spread exceeds 5%; no adoption.
These all-resident trials do not compare eviction policies. 🟠 [Next] Host routing
and Python/graph overhead, then bounded gather-based candidates.

🟢 [Done] Opt-in bounded host phase counters and 27 isolated synthetic trials separate
checksum I/O/hash, MLX load/eval, routing and expert execution. Preloading eliminates
all request-time miss/load/checksum calls but leaves candidates slower than baseline.
Per-expert synchronization and router host materialization remain substantial.
Run spread exceeds 5%; exact contributions and policy superiority remain unqualified.
Nested spans must not be summed; router eval includes gate/top-k, MLX load/eval may
include internal I/O. Separate-time controls do not isolate instrumentation overhead.
🟠 [Next] Evaluate grouped top-k completion with pinned leases and bounded fallback.

🟢 [Done] Nine fresh-process synthetic CPU trials compare baseline/LRU/cost policies
in rotated order; all pass parity. Median subsequent-call times: 0.135/0.336/0.213 ms.
Cost remains slower than baseline and is not adopted. Export/startup are excluded,
OS file cache is not reset, and no real-model/GPU/RSS/P3 qualification is granted.
🟠 [Next] Separate I/O/checksum/load/router-sync measurements and real-weight tradeoffs.

🟢 [Done] Explicit standard Qwen3 MoE switch hook preserves the original gate,
top-k normalization and aggregation. It returns unaggregated expert outputs and
retains no original switch bank. Install/change phase only at idle safe points;
baseline restoration reloads the source model. Twelve tests including native MLX
pass, with six synthetic dense/4/8-bit × normalized/unnormalized block comparisons
within 1e-5 and original-switch weakref release. This serial host-synchronized
reference path is not a speedup or Qwen4Exp qualification.
🟠 [Next] Real compatible weights, latency/I/O measurements and P3 comparison.

🟢 [Done] Offline standard SwitchGLU export supports uniform dense/affine 4/8-bit
experts in a new directory. Unsupported activations/bias/mixed formats are rejected;
the manifest is published last. Partial artifacts remain after failure without a
completion manifest. Eleven tests including native MLX pass; synthetic exports
match original SwitchGLU routed outputs within 1e-5. Source weights remain owned
by the offline caller; no runtime replacement or model performance is certified.
🟠 [Next] Architecture-specific hooks and runtime ownership transition.

🟢 [Done] A bounded manifest binds expected model SHA256, quantization and per-expert
size/checksum. Duplicate fields/entries and drift are rejected before loading on
cache misses. Ten tests including native MLX pass; 15 related tests pass.
Use `from_manifest` for integration. Trusted immutable offline artifacts are required;
checksums are not signatures, and concurrent mutation/cached-entry revalidation are
outside this contract. 🟠 [Next] Model export and architecture-specific routing hooks.

🟢 [Done] Explicit affine 4/8-bit expert support uses quantized matmul without dense
weight expansion. Packed weights/scales/affine biases are validated and counted.
Six native CPU tests pass; tested 4/8-bit × group32/64 agree with dequantized
quantized-weight references within 1e-6. Group128 is accepted but untested here.
This does not establish original-model quality or speed. 🟠 [Next] Bind bits/group/model
identity in an artifact manifest and connect real routing.

🟢 [Done] Per-token authoritative routing supports multiple experts and preserves
unnormalized weights and order. All rows are validated before loading; up to 4096
rows run sequentially. Four native MLX CPU tests pass, including numerical parity,
eviction and failure cleanup. This reference path does not certify throughput or
install real-model router hooks. 🟠 [Next] Explicit quantized-expert format and parity.

🟢 [Done] Independent-file MLX dense bias-free SwiGLU adapter validates tensors,
materializes weights/results and drops tensor references on eviction. Native CPU
array tests verify reference parity through eviction/reload; synthetic weights do
not certify model quality, GPU performance or OS memory reclamation. No model hook
is installed. Trusted offline files are required; file bounds do not cap allocator peaks.
⭕️ [Pending] Target-model E2E requires actual weights and architecture/quantization support.
🟠 [Next] Multi-expert, quantization and real-router integration validation.

🟢 [Done] A synchronous resident-expert execution bridge preserves router order and
weights, retains all leases until backend completion, cleans up failed acquisitions
and execution, and records bounded phase-specific hit telemetry. Twenty-two related
tests pass. The consumer must materialize GPU results before returning. Tested with
a test backend; real-model resource adapters and speedup remain unverified.

🟢 [Done] Opt-in cost/frequency eviction reuses the existing residency manager;
LRU remains the default. Scores combine capped resident access counts, synchronous
load time, bytes and logical age. Leased entries remain pinned; bounds and deferred
resize are shared. Metadata is discarded on eviction. Related tests: 31 passed.
Host load time does not prove GPU readiness or actual miss cost. No speedup is claimed.
🟠 [Next] Connect real MoE resource operations and phase measurements, compare against
LRU and use the existing P3 gate before adoption.
🔴 [Later] Prefetch, compatible speculation, KV compression and fusion.
⭕️ [Pending] MTP/ANE routes lacking testable artifacts or hardware.

## 简体中文

🟢 [Done] 可选top-k合并求值在完成前保持全部Lease，entry／byte不足时逐次fallback。
包含native的11项测试通过，dense/4/8-bit一致及失败回收验证。
18次预常驻CPU试验将eval调用16→8，但延迟变化不一致：LRU 160→152.5 us，
cost 183.291→185.125 us，波动超5%，不采用。全常驻试验不能比较退避策略。
🟠 [Next] host路由及Python／graph固定开销，之后考虑有界gather候选。

🟢 [Done] 可选有界host分阶段计测及27次独立合成试验，分开checksum I/O/hash、
MLX加载／求值、路由及Expert执行。预常驻使请求内miss/load/checksum调用全为0，
候选仍慢于baseline，逐Expert同步及Router host取值仍有明显开销。
run波动超过5%，不能认证精确贡献或策略优势。嵌套时间不可相加，Router eval包含
gate/top-k，MLX load/eval可能包含内部I/O，异时对照不能单独确定计测overhead。
🟠 [Next] 验证持有top-k Lease后合并求值的候选及有界fallback。

🟢 [Done] 轮换顺序、各3个独立进程的合成CPU比较共9次全部数值一致。
后续调用中位数baseline/LRU/cost为0.135/0.336/0.213 ms，cost仍慢于baseline，不采用。
不计export/startup、不清OS file cache，未认证实际模型/GPU/RSS/P3。
🟠 [Next] 分开测量I/O/checksum/load/router同步及实际权重tradeoff。

🟢 [Done] 标准Qwen3 MoE显式switch hook保留原gate、top-k正规化及集约，
返回未集约Expert结果且不保留原switch bank。只在空闲safe point安装及切换phase，
恢复baseline需重载模型。包含实际MLX的12项测试通过；dense/4/8-bit及正规化开关
共6种合成block比较误差低于1e-5，原switch weakref已释放。
逐次host同步参考路径不代表加速或Qwen4Exp认证。
🟠 [Next] 实际兼容权重、延迟／I/O测量及P3比较。

🟢 [Done] 标准SwitchGLU离线导出支持一致的dense／affine 4/8-bit格式，仅写入新目录。
拒绝不支持的activation、bias及混合格式，最后发布完成manifest。
失败保留部分文件但不发布manifest。包含实际MLX的11项测试通过，合成导出与原
SwitchGLU路由结果误差低于1e-5。离线caller仍持有source权重，未认证runtime替换或模型性能。
🟠 [Next] architecture专用连接及runtime权重所有权切换。

🟢 [Done] 有界manifest绑定model SHA256、量化配置及各Expert大小／校验值，
cache miss加载前拒绝重复项与内容变化。包含实际MLX的10项及相关15项测试通过。
接入时使用from_manifest，前提是可信且不变的offline文件；checksum不是签名，
并发修改及已常驻entry重验证不在本契约内。
🟠 [Next] 模型导出及architecture专用Router接入。

🟢 [Done] 加入显式affine 4/8-bit Expert支持，不展开dense权重，验证并计入packed
weight／scale／affine bias。实际CPU测试6项通过；4/8-bit × group32/64与同一量化权重
的展开参考误差低于1e-6。group128可配置但未在本次验证。
不代表原模型质量或速度认证。🟠 [Next] 用manifest绑定bits／group／model identity并接入实际路由。

🟢 [Done] 实现逐token多Expert路由，保留选择顺序及未归一化权重，加载前验证全部行。
最多4096行逐次执行。实际MLX CPU测试4项通过，包含数值一致、退避和失败回收。
这是参考路径，不代表吞吐认证，也未自动接入模型Router。
🟠 [Next] 明确量化Expert格式及数值比较。

🟢 [Done] 独立文件MLX浮点无bias SwiGLU适配器验证tensor并完成权重及结果求值，
退避时释放tensor引用。实际CPU数组验证了退避及重载后的数值一致；合成权重不代表
模型质量、GPU性能或OS内存回收认证。尚未安装模型hook，文件必须是可信offline产物。
⭕️ [Pending] 目标模型E2E需要实际权重及architecture／量化适配。
🟠 [Next] 多Expert、量化及实际Router接入验证。

🟢 [Done] 同步Expert执行连接保留Router顺序及权重，在backend完成前保持Lease，
失败时回收Lease，并记录有界的分阶段hit统计。相关22项测试通过。
consumer返回前必须完成GPU结果求值。仅验证测试backend，实际模型适配及加速尚未验证。

🟢 [Done] 在已有常驻管理器中加入可选cost_frequency策略，默认仍为LRU。
评分结合常驻访问次数、同步加载时间、内存大小和逻辑时间衰减；Lease保护、
容量上限及延迟resize保持共用。移除entry时丢弃统计。相关31项测试通过。
host加载时间不能证明GPU就绪或实际miss成本，尚未证明速度改善。
🟠 [Next] 接入实际MoE资源操作及分阶段测量，与LRU比较，通过已有P3判定后再采用。
🔴 [Later] 预取、兼容推测解码、KV压缩和融合。
⭕️ [Pending] 缺少可测试artifact或硬件的MTP／ANE路径。
