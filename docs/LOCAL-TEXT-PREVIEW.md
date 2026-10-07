# Local text preview 0.1.0

## 日本語

本プロジェクトを、現在のM4／32 GiBで使えるローカルtext previewとして仕上げた。
対象は検証済みGemma 2 2B 4-bit artifactのみ。一般のモデル対応・最速・本番安定性を認定した版ではない。

### 起動

配置済みモデルと検証済みMLX Pythonを使う場合、プロジェクトrootで実行する。

```sh
/opt/homebrew/opt/vllm-metal/libexec/bin/python -m vllm_apple.local_text \
  --model models/gemma-2-2b-it-4bit --check --language ja
/opt/homebrew/opt/vllm-metal/libexec/bin/python -m vllm_apple.local_text \
  --model models/gemma-2-2b-it-4bit --language ja
```

`--check`はGPUを起動せず、hardware・空きmemory・MLX 0.32.1／MLX-LM 0.32.0／Transformers 5.17.0、
review済み依存sourceとmodel／index／tokenizerのhashを確認する。未知version、改変artifact、追加weight、
未対象hardwareは理由を表示して拒否する。eligibleはpreviewの起動候補であり、GPU・性能認定ではない。
実起動時にはP1 profileのGPU・source gateも適用する。`--port`の既定は8000、接続は127.0.0.1のみ。

```sh
curl http://127.0.0.1:8000/v1/models
curl http://127.0.0.1:8000/v1/chat/completions \
  -H 'Content-Type: application/json' \
  -d '{"model":"default_model","messages":[{"role":"user","content":"1+1は？数字だけ答えてください。"}],"max_tokens":16,"stream":true}'
```

model aliasは`default_model`。このGemmaはuser roleの例で利用し、system role・tools等を対応済みと扱わない。
Ctrl-Cで停止する。launcherはworkerへprocessを置き換えるため、別の隠れたworkerを常駐させない。
HTTP contextはprompt＋最大出力4096 tokens、出力512 tokens、concurrency 1、allocator 8 GiB、
allocator cache 256 MiB、connection 16の固定上限。過大requestや未知modelを拒否する。

### wheel

`dist/local-preview/vllm_apple-0.1.0-py3-none-any.whl`を作成済み。
MLX依存は同梱せず、検証済み依存を持つ環境へインストールする。

```sh
# pythonは上記の検証済みMLX依存を持つ、選択した環境のPython。
python -m pip install --no-index --no-deps /absolute/path/to/vllm_apple-0.1.0-py3-none-any.whl
vllm-apple-local-text --model /absolute/path/to/gemma-2-2b-it-4bit --check --language ja
vllm-apple-local-text --model /absolute/path/to/gemma-2-2b-it-4bit --language ja
```

[最終wheel試験](evaluation/local-text-installed-wheel-m4-2026-10-07-final.json)はcheckoutをimportせず、
別venvのインストール済みpackageから起動。依存は既存のreview済みMLX環境を参照しているため、
独立clean-machine認定ではない。三言語SSE、未知model／context／output拒否、資源上限、正常終了を確認する。
checksumと検証範囲は[配布manifest](evaluation/local-text-preview-manifest-2026-10-07.json)へ保存。
Python全回帰1451 tests成功（11 skip）、Swift SDK tests・Mac sample build成功。
一時venvからのアンインストール後にpackage・metadata・起動entryの削除も確認した。

### 完了範囲

- [Done] 環境・artifact事前確認、一つの起動コマンド、loopback OpenAI text API、固定資源上限、停止。
- [Done] wheel作成・別venvからの実機確認、Python回帰、Swift SDK tests、既存Mac sampleのbuild。
- [Next] P1長時間SLO・資源gate。直近r8は品質2958/2958だがSLO2943/2958で不合格、8時間未開始。
- [Next] P2数値／goodput、P3実選択、P4の24時間・目的別profile、R0一般品質・LoRA。
- [pending] 他Mac／別SoC、実署名・notarization、独立clean-machine install。必要環境が整った時に再開。

Swift SDK／Mac sampleのbuild成功は、このpreview入口とMac UIの接続認定ではない。
本previewはOpenAI text API入口であり、daemonのhealth／profile／event APIを提供する入口とは区別する。
websiteは変更していない。既存の本番release認定gateを迂回して公開・昇格しない。

## English

The minimum usable milestone is a **local text preview**, limited to the measured M4/32 GiB
and pinned Gemma 2 2B 4-bit artifacts. Use the commands above with `--language en`.
`--check` verifies local artifacts, reviewed dependency sources, memory, hardware and exact
MLX 0.32.1, MLX-LM 0.32.0 and Transformers 5.17.0 versions without starting GPU work.
Unknown configurations and extra weight files are rejected. Eligibility is not certification.

The launch command replaces its process with the bounded worker: loopback only, model alias
`default_model`, concurrency one, 4,096 total context tokens, 512 output tokens, 8 GiB allocator,
256 MiB allocator cache and 16 connections. Ctrl-C stops it. The demonstrated Gemma requests
use user messages; system messages, tools and general quality are not certified.

A wheel is available in `dist/local-preview`. Install it into a selected environment containing
the reviewed dependencies, then use `vllm-apple-local-text`. The installed-wheel test imports
from a separate venv rather than the checkout, but reuses existing reviewed MLX dependencies:
it is not independent clean-machine certification. Three-language SSE, rejected oversized and
unknown-model requests, resource limits and clean exit are tested. The manifest contains the checksum. Python regression passed 1,451 tests (11 skipped); Swift SDK tests and the Mac sample build passed. Temporary-venv package removal was verified.

[Done] Local launcher, checks, bounded text API, package and smoke validation.
[Next] Production stability/performance, P1–P4 qualification, general RAG and LoRA.
[pending] Other hardware, signing/notarization and an independent installation environment.
Swift SDK tests and the Mac sample build do not certify UI integration with this preview entry point.
The preview offers OpenAI text routes, not daemon health/profile/event routes. Website unchanged.
Production release gates remain intact; this artifact is not promoted as a certified release.

## 简体中文

本次完成范围是**本地text preview**，仅限已测M4／32 GiB及固定Gemma 2 2B 4-bit artifact。
使用上面的命令并选择`--language zh`。`--check`不启动GPU，检查hardware、可用memory、artifact、
review过的依赖source及固定MLX 0.32.1／MLX-LM 0.32.0／Transformers 5.17.0。
未知配置、改动artifact、额外weight文件会被拒绝。eligible不代表认证。

启动命令用worker替换自身process：仅127.0.0.1，model alias为`default_model`，concurrency 1、
prompt加最大输出4096 tokens、输出512 tokens、allocator 8 GiB、cache 256 MiB、connection 16。
Ctrl-C停止。Gemma示例使用user消息，不认证system消息、tools或一般质量。

wheel位于`dist/local-preview`，安装到已有review依赖的所选环境后运行`vllm-apple-local-text`。
安装测试从独立venv导入package而非checkout，但复用已有MLX依赖，因此不是独立clean-machine认证。
已验证三语言SSE、过大request及未知model拒绝、资源上限和正常退出，checksum记录在manifest。Python全回归1451项通过（11项跳过），Swift SDK测试及Mac sample build通过，也确认了临时venv中package卸载。

[Done] 本地入口、检查、bounded text API、package及smoke验证。
[Next] 生产稳定性／性能、P1〜P4认证、一般RAG及LoRA。
[pending] 其他hardware、签名／notarization及独立安装环境。
Swift SDK测试及Mac sample的build不认证该preview入口与UI的连接。
该入口提供OpenAI text routes，不提供daemon的health／profile／event routes。website保持不变。
原生产release gate保留，不将此artifact提升为已认证release。
