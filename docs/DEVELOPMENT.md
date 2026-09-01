# Development and CI

## Reproducible setup

control planeのruntimeはdependency-freeである。build backendと開発toolを完全version固定し、既定
Pythonは`.python-version`で3.12とする。test runnerはstandard libraryの`unittest`を使用する。

```bash
make bootstrap
make check
```

`make check`は全Python test、compile、Ruff、Swift SDK test、SwiftUI Mac sample buildを順に実行する。
MLX、vLLM、vLLM-Metalを必要とする実device試験は通常unit CIへ混ぜず、version matrixとpromotion gateを
通したApple Silicon環境で実行する。
`vLLM candidate stack promotion` workflowも認定modelのstreaming SHA-256を試験前後で検証する。
署名済みmanifestを使う場合はQwen認定と同じ4つのintegrity入力をすべて指定し、CA chainとsigner identityを
含むprovenance検証を有効にする。private manifestは昇格report artifactへ含めない。
成功した実機workflowは`promotion-bundle.json`を生成する。bundleはmodel identifierをSHA-256化し、
preflight、qualification、任意artifact admissionの各raw report digestを結合する。取得後は
`vllm-apple qualification-bundle-verify --reports <directory> --bundle <bundle>`で、元reportの
差し替え、欠落、model/memory bindingの不一致をfail-closedで再検証できる。
release authorityはprivate keyをrunnerへ置かず、取得したbundleを隔離環境で
`vllm-apple qualification-bundle-sign --reports <directory>`により元証跡を再検証してからdetached
CMS署名できる。利用側はverify commandへ
`--signature`、`--trusted-ca`、`--expected-signer-sha256`をまとめて渡し、署名者と元reportを同時に検証する。
Macアプリでは`QualificationPromotionBundleStore`へ`promotion-bundle.json`のURLを渡し、同じdirectoryから
decodeした`QualificationReport`とともに`loadValidated`を呼ぶ。SDKはCryptoKitでpreflight、qualification、
任意artifact admission、model identifier、canonical bundle bodyを再hashし、Python生成時の`bundle_id`と
一致したbundleだけを返す。readerはbundleを64 KiB、各evidenceを1 MiBに制限し、symlinkを受け入れない。
`QualificationPromotionBundleImporter`はdirectory pickerで得たsecurity-scoped URLを取り込む用途に使う。
source検証後、Application Supportのcurrent-user所有directoryへ0700 stagingを作り、bundleが列挙した固定fileだけを
0600でコピーする。コピー後に全digestとbundle IDを再検証し、成功時だけbundle ID directoryへatomic moveする。
同じbundle IDの再import、unsafe source、copy中の変更は既存の認定を上書きせず拒否する。
署名をMac内でも必須化する場合は、app environmentへ`VLLM_APPLE_PROMOTION_TRUSTED_CA`と
`VLLM_APPLE_PROMOTION_SIGNER_SHA256`を対で設定し、import directoryへ`promotion-bundle.cms`を含める。
SDKはmacOS Security frameworkのCMSDecoderとSecTrustを使用し、detached content、custom CA anchor、
signer certificate SHA-256をnative検証する。片方だけのtrust設定はhash-onlyへdowngradeせずimportを拒否する。
UIはhash-only検証とtrusted signature検証を別の三言語statusとして表示する。

runtime profile loaderは1 MiB以下、current-user所有、0600のregular fileだけを受け付ける。field集合、
memory/context境界、capability数を再検証し、symlink、future version、未知fieldをfail closedする。
旧`profile_version=0`は既知の最小field集合に完全一致する場合だけv1へ移行し、暗黙のbest-effort migrationは行わない。
model profile cacheはhardware、OS、総Unified Memory、GPU core数、model IDのcanonical fingerprintを
24桁のfile名に使用し、model名をpathへ露出しない。0700 directoryと0600 profileを使い、最大128件、
最大512 entryのbounded走査で正規のcurrent-user cache fileだけを古い順に整理する。

scheduler queueは既定最大1,024件で、REALTIME、INTERACTIVE、NORMAL、BACKGROUNDの順にdispatchし、
同じpriorityでは到着順を保つ。queue tokenはsnapshotへ公開せず、queued、dispatching、activeの件数だけを
保持する。cancelは待機中requestを除去し、dispatchとの競合中またはadmit済みならreservation登録境界で
必ずmemory accountingをreleaseする。同じtokenの再cancelは副作用なくfalseを返す。

operator runtime fallbackはdispatcherがprobe合格として返したselected backendとfallback chainだけを、
重複なし・最大backend数まで順番に実行する。retryableな`BackendExecutionError`だけを次候補へ送り、
invalid requestなどnon-retryable error、通常のprogramming exceptionはfallbackで隠さない。attempt evidenceは
backend、成功/失敗、bounded error codeだけを保持し、例外detailや入力本文を保存しない。

global submission schedulerは1〜8本（既定1本）の専用daemon workerでbackend commandを実行し、
application threadへbounded handleを返す。commandは既存priority queueとmemory reservationを通り、成功、
backend例外、admission失敗、pending cancelの全経路でqueueから除去する。running commandのcancelはmemoryを
早期解放せず、cooperative cancellation eventをoperationへ渡し、workerの`finally`でだけreservationを解放する。

`vllm-apple inspect-model <model>`はmodelをloadせず、bounded metadata inspection、architecture mode、
state memory内訳、現在のMacに対するsafe/balanced/aggressive context、balanced tierのresident estimate、
backend capability fitをschema v1 JSONで返す。`--backend`、繰り返し可能な`--feature`と`--mode`で認定予定の
構成を事前確認できる。raw config本文やweight内容はreportへ含めない。

runtime startup progressはschema version、stage、completed/total units、整数percent、128文字以下の
localization message keyだけをsnapshotと`runtime.startup_progress` SSE eventへ出す。model path、prompt、
backend logは含めない。Swift SDKがtyped decodeし、Mac sampleは英語、日本語、简体中文の文言とProgressViewで表示する。

optimizer candidate比較は最大64候補に制限し、個別quality gateを通過した候補だけを選択対象にする。
順位はtask score降順、artifact bytes昇順、throughput降順、peak RSS昇順、candidate ID昇順で固定する。
品質未達候補は指標が優れていても選ばず、理由を`quality_gate_failed`としてreportへ残す。promptや生成本文は保存しない。

model inspectionは標準MHA/GQAに加え、MLAでは各Attention層の圧縮latentとRoPE keyだけをtoken依存stateとして
計上する。state-space/MambaではSSM stateとconvolution stateを固定recurrent stateとして計上し、純粋な
recurrent modelのKV bytesは0とする。hybrid modelはboundedな`layer_types`を検証し、Attention層数とrecurrent
層数を分離する。不完全なMLA metadata、未知のlayer type、context上限のない純recurrent modelはfail closedする。

未cache modelは`vllm-apple fetch-model-metadata org/model --revision main`でweightを取得せずに確認できる。
取得先はHTTPSの`huggingface.co`に固定し、model ID、revision、redirect先、1 MiB以下の応答、config構造、
`X-Repo-Commit`を検証する。reportはconfig SHA-256とresolved commitをbindingし、`weights_downloaded=false`と
`memory_fit_evaluated=false`を明示する。artifact byteが不明なmetadataだけで実行可能性を判定してはならない。

`MLXPromptCacheStateAdapter`はMLX wrapperが生成したimmutable prompt-cache snapshotをopaque objectとして所有し、
random handleだけをsemantic cacheへ渡す。array storageは重複objectを除外するbounded traversalで数え、entry/state byte
上限を超えるsnapshotは即時解放する。restoreとreleaseは同一lockで直列化し、release callbackが失敗した場合は所有権を
adapterへ戻すため、coordinatorのbounded retry queueから安全に再試行できる。prompt、token ID、tensor本文はmetricsや
anchor metadataへ保存しない。vLLM-Metalには安定したsnapshot ABIがないため、未定義HTTP endpointは追加しない。

`samples/VLLMAppleChatXcode`はXcodeGenで再生成できるmacOS application targetである。既存Swift source、
VLLMAppleKit local package、英語、日本語、简体中文resourceを共有し、検証済み`VLLM_APPLE_DAEMON_SOURCE`を
`Contents/MacOS/vllm-appled`へ埋め込む。build phaseは相対path、symlink、非実行fileを拒否し、codesign可能な場合は
hardened runtime optionでdaemonも署名する。profile、log、socket、token、modelはapp bundle外に保持する。

実modelのSwift lifecycle E2Eは`VLLM_APPLE_E2E_MODEL`を設定して`scripts/run_swift_model_e2e.sh`を実行する。
専用Swift executableがManagedRuntimeからdaemonとMLX backendを起動し、inference readiness、bounded streaming、
正常shutdownとprocess終了を検証する。生成本文は出力・保存せず、SSE event数と`generated_text_stored=false`だけを
成功summaryへ残す。通常CIにはmodel loadを混ぜず、Apple Siliconの手動qualificationで実行する。

model配布時は`model-integrity-create`のmanifestを`model-integrity-sign`でCMS detached署名する。verify側は
`--signature`、`--trusted-ca`、`--expected-signer-sha256`を必ず組で指定し、CA chain、CMS署名、signer certificate
fingerprint、manifest内容、model treeを順に検証する。daemonにも対応する4つのintegrity引数を渡すとbackend起動や
weight走査より前に同じ検証を行う。private keyは0600のregular fileだけを許可し、daemonやapp bundleへ同梱しない。

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

手動qualification workflowでは`artifact-size-bytes`と`estimated-resident-bytes`を対で指定できる。
指定時はbackend preflightとmodel loadより前に同じadmissionを実行し、結果をprivateな
`artifact-admission.json`として保存する。片方だけの指定、非有限値、16,384 GiBを超える値、または
memory/disk不適合はfail closedとする。`artifact-target`には実際にdownloadを配置するvolume上のpathを指定する。
入力を指定したrunではSwift checkerにも`--require-artifact-admission`を渡し、64 KiB以下のregular fileを
型付きdecodeする。Swift側でも各fit flagと`eligible`をbyte値から再計算するため、単なるtrue値の改変を
認定証拠として受け入れない。symlink、oversize、schema不一致、再計算不一致はevidence missingとして拒否する。
admissionの`model`はqualification reportの`model`と完全一致しなければならず、別候補で得た容量判定の
再利用を認定証拠として受け入れない。識別子は4 KiB以下の表示可能文字列に限定する。
さらにadmissionの`artifact_bytes`と`estimated_resident_bytes`をqualification reportの
`model_memory_fit`と完全一致させる。同じmodel IDでも量子化方式やrevisionが異なるartifactの証拠は流用できない。
workflowは丸め差を避けるためexact byteを受け取る。対話的な事前確認では`--artifact-gib`と
`--resident-gib`も利用できるが、認定証拠には`--artifact-bytes`と`--resident-bytes`を使用する。

PRやpushからは起動せず、同時に複数の実modelを走らせない。report directoryは0700、reportは必要な
場合だけ明示的な`upload-report`入力で14日間保存する。reportには生成本文を含めないが、model識別子を
含むため公開可能性を確認してからuploadする。
# Qwen3.8-Flash-Next text-only qualification

実weightを配置する前は、取得済みの`config.json`だけを使ってbackendのarchitecture featureを
fail-closedで照合できる。metadata-only判定はartifact容量とresident memoryを評価済みとは扱わない。

```bash
python3 -m vllm_apple qualification-preflight \
  --backend-executable /path/to/venv/bin/mlx_lm.server \
  --backend-kind mlx_lm \
  --model-metadata models/qwen3.8-flash-next-metadata \
  --mode text
```

`backend:missing_model_features`が残る間はweightをdownloadしない。実weight配置後の認定では従来どおり
`--model`を使用し、memory fitとartifact admissionを必須にする。`--model`と`--model-metadata`は同時に
指定できない。

更新したMLX LMにQwen4実装候補が含まれるかは、backendをimportせずMetal deviceやmodel bufferを確保しない
静的readiness監査で確認する。

```bash
python3 -m vllm_apple mlx-qwen4-readiness \
  --backend-executable /path/to/venv/bin/mlx_lm.server
```

この監査は再利用可能なGated DeltaNet、MoE、N-gram componentと、不足するQSA、Gated Residual、Qwen4 model
登録を分離して報告する。componentが見つかってもbackend capabilityを自動表明せず、`ready=true`だけでも
実weight qualificationの代用にはしない。

Gated ResidualとQSA block selectorの移植では、Transformers 5.16.1の公式Qwen4-Exp実装を参照基準とする。
`vllm_apple.qwen4_reference`は小型float64 fixtureだけを処理する依存なしCPU oracleであり、production inference
には使用しない。Gated Residualはbranch単位RMSNorm、low-rank SiLU/sigmoid mixing、2倍sigmoid injectionを、
QSAはzero-weight key RMSNorm・identity RoPE fixtureでcomplete blockのReLU head-score top-kと未完tail保持を
検証する。入力幅とtoken数を制限し、巨大model
allocationなしで後続MLX adapterとのcorrectness比較に使用する。

参照実装：
[Transformers 5.16.1 Qwen4-Exp](https://github.com/huggingface/transformers/blob/v5.16.1/src/transformers/models/qwen4_exp/modeling_qwen4_exp.py)

MLX 0.31系の小型primitive fixtureは通常のmacOS terminalで実行する。workerは128 byteの入力tensorから
Gated ResidualとQSA selectorを計算し、controllerがCPU oracleとの差の最大値だけを保持する。model weight、
生成本文、tensor値はreportへ保存しない。このfixtureはallocatorのpeak memoryを測定せず、実model RSS gateの
代用にはしない。

```bash
python3 -m vllm_apple qwen4-mlx-fixture \
  --python-executable /path/to/mlx-0.31-venv/bin/python
```

Metal deviceを取得できないsandboxでは終了code 2でfail closedする。通常terminalで`passed=true`になっても、
Qwen4 model登録、weight mapping、cache、実model semantic qualificationが完了するまではbackend capabilityを
表明しない。

公式safetensors indexはweightを開かず、text、MTP、Visionのmode別mappingを検証できる。indexは4 MiB、
16,384 entries、tensor名1 KiBに制限し、shard名のpath traversalを拒否する。

```bash
python3 -m vllm_apple qwen4-weight-map-inspect \
  --model-metadata /path/to/Qwen3.8-Flash-Next \
  --index /path/to/Qwen3.8-Flash-Next/model.safetensors.index.json \
  --mode text --mode mtp --mode vision
```

公式BF16 indexでは1,658 required entries、131 shards、359,999,963,128 artifact bytesが完全一致する。
reportは不足名を最大32件に制限し、tensorをloadしない。変換済みMLX artifactは同じcommandで変換後indexを検査し、
mappingが異なる場合は専用schemaを追加してから認定する。

cache/state契約はGDN recurrent/conv、QSA KV/indexer/block tail、PLE n-gram/conv tailをconfig fingerprintへ
結び付ける。固定9-token fixtureを一括prefill、4+2+3 segmented prefill、1-token decodeで処理し、token chain、
tail、block境界が完全一致することだけを保存する。token IDとtensor値は保存しない。

```bash
python3 -m vllm_apple qwen4-cache-fixture \
  --model-metadata /path/to/Qwen3.8-Flash-Next
```

Qwen専用workflowはmodel load前にweight map、conversion plan、cache fixtureの3つを実行する。これらはconfig
fingerprintとindex digestの一致をpromotion bundleで再検証し、一部だけの証跡や改変されたcache証跡を拒否する。

変換前には同じindexからcomponent分類とstreaming scheduleを生成する。公式indexの全1,658 tensorをGDN、QSA、
Gated Residual、MoE、PLE、Vision、MTP、embedding/headへ分類し、未分類tensorが1件でもあれば停止する。

```bash
python3 -m vllm_apple qwen4-conversion-plan \
  --model-metadata /path/to/Qwen3.8-Flash-Next \
  --index /path/to/Qwen3.8-Flash-Next/model.safetensors.index.json \
  --mode text
```

planはsource/destination shardを各1つだけ開き、MoEをon-demand expert、PLEをpartitioned lookup、Vision/MTPを
mode別optional componentとして扱う。現段階ではtensor dataをload・変換せず、source tensor名を保持したMLX adapter
向けscheduleだけを生成する。plan ID、config fingerprint、index SHA-256はweight/cache証跡とともにpromotion bundleへ
bindingされる。

実weightを扱う前処理には、tensorをdecode・量子化せずsource artifactを安全な作業領域へ複製するbounded shard
stagerを利用できる。8 MiB固定bufferでsource/destination shardを各1つだけ開き、temporary fileへ書き切って
`fsync`した後にatomic置換する。出力先はprivate directoryでなければならず、sourceと親子関係にできない。

```bash
python3 -m vllm_apple qwen4-shard-stage \
  --source /path/to/Qwen3.8-Flash-Next \
  --output /path/to/private-qwen4-stage \
  --maximum-output-bytes 370000000000 \
  --mode text \
  --execute
```

中断後は同じ引数へ`--resume`を追加する。checkpointのplan ID、file size、SHA-256が一致したshardだけを再利用する。
`--execute`を省略すると何も書き込まない。このstagerはtensor名とshard内容を保持するidentity-preserving層であり、
production MLX adapterまたは量子化器はmanifest検証済みartifactを別工程で入力にする。

staging後はadapterを構築する直前に全shardを再検証する。manifestにないfile、欠落file、config/indexとplanの不一致、
shardのsizeまたはSHA-256不一致はすべて拒否される。

```bash
python3 -m vllm_apple qwen4-stage-verify \
  --stage /path/to/private-qwen4-stage \
  --maximum-artifact-bytes 370000000000 \
  --mode text

python3 -m vllm_apple qwen4-adapter-contract \
  --stage /path/to/private-qwen4-stage \
  --maximum-artifact-bytes 370000000000 \
  --mode text
```

adapter contractは検証済みplan/config/indexへbindingされ、component policyとshardごとのactive entry数だけを出力する。
text-onlyではVision/MTP entryを`skipped_optional_entries`へ分離する。tensor dataのloadやMetal allocationは行わず、
後続production adapterが処理してよいshard/component境界を固定する。

次にsafetensors headerだけを読み、indexとのtensor名完全一致、dtype、shape、data offsetを検証する。

```bash
python3 -m vllm_apple qwen4-adapter-headers \
  --stage /path/to/private-qwen4-stage \
  --maximum-artifact-bytes 370000000000 \
  --mode text
```

headerは既定64 MiB、tensor rank 16、shardあたり16,384 tensorに制限する。重複JSON key、未知dtype、shapeとbyte長の
不一致、重複range、未割当gap、shard外offsetを拒否する。各shardを1つずつ開いてheader範囲だけを読み、weight dataや
MLX/Metal allocatorには触れない。reportにはdtype/rank別件数だけを保存し、tensor名やshape一覧は出力しない。

大容量Apple Silicon runnerには`self-hosted`、`macOS`、`ARM64`、`vllm-metal`に加えて
`large-memory` labelを設定する。GitHub Actionsの
`Qwen Flash Next text-only qualification`を手動実行し、runner上のlocal model path、backend
executable、artifactの正確なbyte数、load前resident推定値を指定する。

workflowはmodel loadより先にUnified Memoryとdisk stagingを判定し、text-only capability、
三言語quality smoke、TTFT、TPOT、tokens/sec、peak RSS、30分memory stabilityを検証する。
reportにはmodel identifierが含まれるため、artifact uploadは既定で無効である。
model treeはload前に8 MiB chunkのstreaming SHA-256 manifestをprivate directoryへ生成し、
qualification終了後に再検証する。試験中に同じpathのweightやconfigが差し替わった場合は認定を失敗させる。
manifestは相対file名を含むため、GitHub Actions artifactにはuploadしない。
配布元の署名済みmanifestがある場合は、`integrity-manifest`、`integrity-signature`、
`integrity-trusted-ca`、`integrity-signer-sha256`をすべて指定する。このmodeでは一時manifestを
信頼の根拠にせず、load前とqualification後の両方でCA chain、signer certificate identity、署名、
model treeを再検証する。4入力の一部だけを指定したrunはmodelをloadする前に拒否する。
