# vLLM-Apple Runtime

An AI runtime control plane designed for Apple Silicon with a strong emphasis on memory stability.

Currently in Phase 1, it implements hardware/memory detection, safe context calculation, runtime profiling, a memory-reservation scheduler, a headless daemon, and a versioned local API.

## Core Principles for Memory Stability

- Context calculation adopts the smaller value between physical memory and currently available memory.
- Excludes OS reserves, workspace allocations, and emergency headroom from the inference budget.
- Limits HTTP request bodies to 4 MiB.
- Caps server worker threads at a default of 32, rejecting overloaded requests early with HTTP 503.
- Retains load metrics using fixed buckets and capped error classifications to prevent telemetry itself from increasing memory footprint.
- Writes profiles to temporary files and atomically replaces them after `fsync`.

The Swift Package for Mac applications is located in `sdk/swift`. It has no dependencies on SwiftUI or AppKit, exposing Foundation, `async/await`, and `AsyncThrowingStream` as public interfaces.

```bash
python3 -m vllm_apple hardware
python3 -m vllm_apple doctor
python3 -m vllm_apple context --model-memory-gb 8 --kv-bytes-per-token 524288
python3 -m vllm_apple serve
python3 -m vllm_apple serve /absolute/path/to/model --max-model-len 8192
python3 -m vllm_apple serve /absolute/path/to/MLX-model \
  --backend-kind mlx_lm --backend-executable /absolute/path/to/mlx_lm.server
python3 -m vllm_apple serve --socket-path /tmp/vllm-apple.sock \
  --session-token-file /tmp/vllm-apple.token
```

Unless explicitly specified, the server binds to `127.0.0.1` only. Even when no inference backend is configured, the health, hardware, and runtime profile APIs remain available.

When a model is specified, it launches the selected vLLM-Metal or MLX LM server as a separate process on the loopback interface and proxies OpenAI API requests through the control daemon. SSE transfers responses in small chunks without buffering full responses. If `doctor` fails to verify a compatible environment, startup is refused. To explicitly use a different environment, specify `--backend-executable /path/to/vllm`.

If the model exists in a local directory or Hugging Face cache, it calculates standard Transformer/GQA KV memory requirements from weight shards and `config.json`, automatically applying BALANCED context limits. Models that cannot be safely inspected are capped at 4096 tokens unless `--max-model-len` is explicitly provided.

A Unix Domain Socket (UDS) can be created for local connections from Mac apps. To avoid exposing session tokens on the command line, using `--session-token-file` is recommended. Token files and sockets are created with `0600` permissions, and runtime state/failure events can be subscribed to via SSE from `/v1/events`.

If the daemon is forcefully terminated and a stale UDS entry remains, subsequent startups verify ownership and socket type before replacing the stale entry. Session token files are reused, and standard SIGTERM shutdowns delete the UDS.

The v1 schemas for health, runtime snapshots, and runtime events are locked in `schemas/`, and live HTTP/SSE responses are validated with a dependency-free validator. If the validator detects unsupported schema keywords, tests fail to ensure schema constraints are never silently ignored.

`VLLMAppleKit`'s `UnixSocketRuntimeClient` supports health, profile, chat, and SSE over POSIX UDS. Passing socket paths and token files to `ManagedRuntime` automatically selects the UDS client, providing bounded stdout/stderr logging, readiness monitoring, and capped restart policies upon failure.

`RuntimeResourceResolver` detects `vllm-appled` inside the app bundle and resolves Application Support paths for profiles, logs, and tokens, as well as `/tmp` paths for UDS to satisfy Darwin path length limits. Directories are created with `0700` permissions, and session tokens with `0600`.

A minimal macOS SwiftUI chat sample supporting English, Japanese, and Simplified Chinese is located in `samples/VLLMAppleChat`. If no daemon is found in the app bundle, it connects to `127.0.0.1:8000`. Environment variables can be used to explicitly specify a development daemon executable.

```bash
cd samples/VLLMAppleChat
swift run
VLLM_APPLE_DAEMON_PATH=/path/to/vllm-appled swift run
```

### Trying Chat with Real Models

Models launched with MLX LM's `--model` flag are referenced as `default_model` in the chat API. `/v1/models` also returns only this managed alias. The send button in the Mac app is disabled when no model is loaded. When using an existing daemon, run the following in a separate terminal before running `swift run`:

```bash
python3 -m vllm_apple serve /absolute/path/to/MLX-model \
  --backend-kind mlx_lm \
  --backend-executable /absolute/path/to/mlx_lm.server \
  --max-model-len 4096
curl http://127.0.0.1:8000/health
python3 -m vllm_apple.soak --url http://127.0.0.1:8000 \
  --mode chat-mixed --model default_model --duration 5 --concurrency 1
cd samples/VLLMAppleChat
VLLM_APPLE_CHAT_MODEL_ID=default_model swift run
```

To have the Mac application launch the daemon automatically, set `VLLM_APPLE_DAEMON_PATH` to an executable `vllm-appled`, `VLLM_APPLE_MODEL_PATH` to the absolute path of the model, `VLLM_APPLE_BACKEND_KIND=mlx_lm`, and `VLLM_APPLE_BACKEND_EXECUTABLE` to the absolute path of `mlx_lm.server`. In this scenario, the application's model ID defaults to `default_model`, allowing prompt transmission once model loading completes.

### Verifying Homebrew vLLM-Metal Candidates

The Homebrew build is treated as a separate candidate from the certified stack with fixed revisions. First, check whether Metal is available and whether vLLM actually selected the Metal platform:

```bash
python3 -m vllm_apple doctor
```

If `vllm_metal_available_but_not_selected` or `vllm_metal_platform_not_selected` is displayed, do not promote the Homebrew build to the active execution path for vLLM-Metal; instead, explicitly specify the `mlx_lm` backend for chat. Only candidates with confirmed Metal platform selection are verified in order: non-stream, SSE, memory, and qualification.

On September 19, 2026, dependency checks and Metal platform initialization were verified with Homebrew vLLM / vLLM-Metal 0.29.0, Transformers 5.17.0, and tokenizers 0.23.2. Note that the `+cpu` version suffix does not indicate that Metal is unavailable. If CPU is selected, check GPU access privileges and plugin exceptions via `VLLM_LOGGING_LEVEL=DEBUG vllm --version`. Downgrading Transformers alone is not recommended as it breaks vision and audio backend dependencies.

If a dedicated environment created via the official install script exists separately from Homebrew, it can be explicitly specified as a candidate. Because Metal initialization fails in headless or sandboxed environments where the GPU is not visible, run diagnostics within a physical machine login session.

```bash
python3 -m vllm_apple doctor \
  --backend-executable "$HOME/.venv-vllm-metal/bin/vllm"
```

## Concurrency and Memory Stability Verification

For a running vision-capable server, you can execute a vision input smoke test via `python3 -m vllm_apple vision-smoke --url http://127.0.0.1:8000 --model MODEL_ID`. This sends 32x32 red and blue PNG images with identical prompts and cross-checks 6 responses across English, Japanese, and Simplified Chinese after trimming only leading/trailing whitespace. Image data and response bodies are excluded from saved reports. This serves as a smoke test for the vision input path and is not a certification of long-term stability or general vision understanding capability.

`vllm-apple-soak` does not maintain unlimited history samples; instead, it outputs JSON containing throughput, failure counts, and RSS growth using 12 fixed latency buckets and up to 17 error keys. By default, non-loopback connections are rejected.

```bash
python3 -m vllm_apple.soak --duration 300 --warmup 5 --concurrency 8
python3 -m vllm_apple.soak --mode chat --model your-model \
  --duration 1800 --concurrency 4 --pid 12345 --max-rss-growth-mib 256 \
  --session-token-file /path/to/session.token
python3 -m vllm_apple.soak --mode chat-mixed --model your-model \
  --duration 1800 --concurrency 4 --pid BACKEND_PID --max-rss-growth-mib 256 \
  --require-30-minute-window --session-token-file /path/to/session.token
```

It returns exit code 1 on request failures or RSS threshold breaches, and exit code 2 on configuration or connection preparation errors. For long-duration evaluations on real models, pass the post-load daemon PID to `--pid`. `chat-mixed` alternates between non-streaming JSON and streaming SSE requests, verifying response structure and successful completion up to `[DONE]`. The 30-minute qualification mode strictly requires $\ge 1800$ seconds, PID monitoring, and RSS growth caps, failing if the backend terminates early.

To verify everything from backend startup to shutdown in one execution, use `qualify-model`. It inspects vLLM/vLLM-Metal compatibility prior to launch and automatically attaches the post-load PID to RSS monitoring.

```bash
python3 -m vllm_apple qualify-model /path/to/model \
  --backend-executable /path/to/vllm \
  --max-model-len 16384 --duration 1800 \
  --concurrency 4 --max-rss-growth-mib 256
```

For quick wiring checks only, `--allow-short-run --duration 60` can be passed. These results do not qualify for 30-minute certification.

During daemon SIGINT/SIGTERM shutdown, new requests are rejected with `server_draining`, while existing HTTP/UDS requests are allowed up to a default 30-second grace period before stopping the backend. The grace period can be customized using `serve --shutdown-grace-period SECONDS`.

In schedulers with the operator dispatcher enabled, MLX/Metal accelerators are selected only if kernel probes bound to current hardware, OS, toolchain, and backend versions succeed. Unprobed kernels, correctness mismatches, performance regressions, or native crashes fail closed into quarantine, falling back to available MLX or CPU paths. MLX probes run in short-lived isolated subprocesses to protect the daemon from native crashes. Native Metal probes compile, dispatch, and read back fixed small shaders in Swift subprocesses using a dedicated temporary module cache. Anomalies in Metal devices, toolchains, or command queues do not crash the daemon; they are treated as operator quarantines.

`RuntimeProbeCoordinator` unifies MLX/Metal results into the same environment fingerprint and applies the dispatcher at scheduler safe points. Active requests are held as pending and switched after the final reservation release, sharing boundaries with execution plans or elastic memory policies.

When `serve <model>` runs with compatibility checks enabled, it detects Metal toolchains, MLX packages, and backend versions without importing MLX itself, automatically executing the probe coordinator at startup. Probe completion count and quarantine count are reported via bounded runtime events. `--skip-runtime-probes` can be passed to disable auto-probing strictly during troubleshooting.

Probe results are stored atomically in a private cache per hardware/environment fingerprint. Exact cache matches skip native probing and reuse quarantine states. If the OS, toolchain, MLX, backend, or probe suite version changes, a new cache entry is created and re-measured safely. Cache entries expire after 7 days by default, and future creation timestamps are rejected.

The MLX suite tests vector addition, 16x16 matmul, bounded KV copy, and scaled dot-product attention with sequence lengths 8/32 and head dimension 8. Attention outputs are normalized to 5 decimal places, tolerating harmless floating-point noise while quarantining the entire operator if either shape fails. The same capability covers causal prefill, single-token decode, and GQA with 4 query heads / 2 KV heads, enabling `attention` only when all scenarios match. Caches from different suite versions are not migrated; correctness is re-measured.

The Paged Attention probe reconstructs KV from non-contiguous block tables, verifying decode tiers for 14, 256, and 1024 tokens. The Native Metal version independently verifies compile, dispatch, and readback of a fixed 14-token shader as a distinct capability. The MLA probe expands 16 token $\times$ 4-dimensional compressed latents to 8 dimensions using separate key/value projections and compares attention outputs. Starting from probe suite version 4, each operator is quarantined independently; current suite version 6 re-measures with a new probe contract featuring bounded numerical tolerances.

Bounded shape profiles can also be generated from a model's `config.json`, extracting query/KV heads, head dimension, context, block count, and KV working-set without loading weight tensors.

```bash
python3 -m vllm_apple kernel-shape-profile /path/to/model \
  --contexts 128,1024,4096,16384 --block-tokens 16
```

The Native Metal adapter consumes this profile in batches of up to 4 shapes at a time, profiling each shape under a distinct probe identity. Representative buffers enforce a 64 MiB hard limit, rejecting shape allocations before allocation if they exceed this limit. Benchmark reports can be saved privately and atomically, bound to model profile, hardware, and environment fingerprints. On load, permissions, file size, all identities, and result provenance IDs are re-verified. For long shapes, bounded numerical vector comparison within a maximum absolute error of $10^{-5}$ is used; other probes require exact SHA-256 digest matching as before. Current suite version 7 splits shape kernels into three stages—score, softmax, and output—dispatching score along the context axis and output along the head dimension axis. Softmax performs max/sum reduction in fixed 256-thread scratch memory with no context-length-dependent threadgroup memory. Intermediate score buffers count toward the 64 MiB total allocation ceiling.

The shape autotuner evaluates up to 4 candidates (32, 64, 128, 256 threads) against the same CPU baseline, comparing Metal median runtimes only for candidates passing correctness checks. Candidates within 2% of the fastest speed are treated as equal, deterministically selecting the configuration with the smaller total thread count. The winner and all candidates are saved privately and atomically as a versioned report per model, hardware, and environment fingerprint. Erroneous fast kernels are never selected as winners. During reload, file ownership, permissions, size, and all fingerprints are checked, re-calculating the winner from saved candidates to reject any report inconsistent with the 2% tie-break policy.

```bash
python3 -m vllm_apple metal-shape-tune /path/to/model \
  --contexts 128,1024 --samples 3

python3 -m vllm_apple metal-shape-tune /path/to/model \
  --contexts 1024 --samples 3 --stdout
```

Installing a report into the daemon holds it as pending while active requests exist. Each reservation fixes its starting `tuning_id`, switching winners only at a scheduler safe point after the final active request completes, preventing mixed thread configurations within a single request.

During inference, the winner table is passed to the local backend as a versioned internal header of 4 KiB or less. OpenAI-compatible request JSON remains unmodified. The same reservation is retained until response completion for non-streaming, or upstream end / client disconnect for streaming, preventing mid-execution configuration switches. Engines lacking context-aware contract support safely fall back to legacy invocation.

When launching the daemon with a model, it automatically searches Application Support for saved reports matching the model shape, Apple hardware, and OS/Metal/MLX/backend versions exactly. Candidates in private directories are capped at 64, ignoring corrupted, permission-invalid, or identity-mismatched files, deterministically installing the latest verified report. Explicit report specification or disabling automatic adoption is supported.

```bash
python3 -m vllm_apple serve /path/to/model \
  --metal-tuning-report /path/to/tuning.json

python3 -m vllm_apple serve /path/to/model --disable-metal-tuning
```

An ASGI adapter without extra dependencies can be integrated into vLLM-Metal. Contexts are isolated per-request via `ContextVar`; if headers are corrupted or shapes do not match exactly, `None` is passed to fall back to the backend default configuration. Winners from separate requests are never mixed.

```python
from vllm_apple import BackendKernelTuningAdapter, KernelTuningASGIMiddleware

tuning = BackendKernelTuningAdapter()
app.add_middleware(KernelTuningASGIMiddleware, adapter=tuning)

# Bridge used at actual call site for Paged Attention
result = tuning.invoke_paged_attention(invoke_kernel, shape, pages, query)
```

`invoke_kernel` receives `(shape, configuration, ...)`. If `configuration` exists, `score_width`, `softmax_width`, and `output_width` are applied to each Metal dispatch. Metrics track accepted/rejected contexts and shape hits/misses.

During managed backend startup, `vllm serve --help` is inspected within a 10-second and 1 MiB limit; middleware is registered automatically only if both `--middleware` and `--disable-frontend-multiprocessing` are supported. Unsupported backends retain inference launch while disabling tuning integration. Accepted backends return a tuning ID in responses, and the control plane measures acknowledged, missing, and mismatched responses separately. Acknowledgements are recorded not at header parsing, but when a shape-matched winner is retrieved from kernel hooks.

vLLM-Metal source compatibility with the current tuning ABI can be inspected without running the source:

```bash
python3 -m vllm_apple vllm-metal-integration-inspect /path/to/vllm-metal
```

`doctor` verifies Python as well as verified matrices for vLLM, vLLM-Metal, and Transformers. Currently, vLLM 0.28.0 / Transformers 5.15.0 are rejected as unverified until vLLM-Metal compatibility verification is complete. See [VERSION-COMPATIBILITY.md](docs/VERSION-COMPATIBILITY.md) for details.

The current native v2 has differing thread configurations and shared memory conditions across NAX, tiled, per-token, and split/reduce, so 3-stage winners are not applied implicitly. See [VLLM-Metal-Integration.md](docs/VLLM-Metal-Integration.md) for details.

Native v2 tuning profiles treat NAX, tiled, per-token, and split-KV as distinct families. Applying medians over up to 9 samples, correctness and output digests, and 2% tie-breaking, profiles are pinned to hardware and upstream source fingerprints. Profiles are stored privately and atomically (up to 16 shapes, 512 KiB max), re-validating all policies including winners upon reload.

In vLLM-Metal source implementing the native measurement ABI, bounded decode/prefill shapes can be generated from model metadata to save real device profiles. Unsupported native extensions terminate safely.

```bash
python3 -m vllm_apple vllm-metal-v2-tune /path/to/model \
  --source-root /path/to/vllm-metal \
  --helper /path/to/vllm-apple-v2-measure \
  --contexts 1024,4096 --samples 3
```

Metal benchmarks for real model shapes can be run in a single command. By default, results are stored privately in Application Support per hardware, environment, and model profile, while `--stdout` returns versioned JSON easy for Mac apps to handle.

```bash
python3 -m vllm_apple metal-shape-benchmark /path/to/model \
  --contexts 128,1024 --block-tokens 16 --samples 1

python3 -m vllm_apple metal-shape-benchmark /path/to/model \
  --contexts 1024 --stdout
```

## Prefill / Decode Profile

When a streaming backend returns OpenAI-compatible stream usage, TTFT and TPOT can be measured separately. Prompts and generated text are not saved to profiles; only fixed-size buckets, token counts, and target PID peak RSS are retained.

```bash
python3 -m vllm_apple execution-profile --save
python3 -m vllm_apple phase-profile --model your-model --samples 5 \
  --pid BACKEND_PID --session-token-file /path/to/session.token
```

Backends that do not return usage fail as `usage_missing` without inferring token counts. Remote URLs are rejected by default, requiring explicit `--allow-remote`.

## Gradual Long-Context Evaluation

`long-context-evaluate` uses vLLM's `/tokenize` to adjust retrieval prompts to target token lengths, evaluating from short contexts (e.g., 1K, 4K, 16K) upward. Each stage separately logs retrieval success rate, TTFT, TPOT, tokens/sec, model load peak RSS, steady/request peak RSS, and state bytes.

```bash
python3 -m vllm_apple long-context-evaluate \
  --url http://127.0.0.1:8001 --model your-model \
  --stages 1024,4096,16384 --memory-ceiling-gb 24 \
  --state-bytes-per-token 131072 --pid BACKEND_PID
```

Connecting directly to `/tokenize` and `/v1/chat/completions`, the default URL is backend port 8001. If actual stream usage tokens deviate from target, retrieval keys mismatch, or memory ceilings are breached, the stage fails and longer stages are skipped. Prompts and generated text are excluded from reports.

## Semantic Anchor Cache

To reduce redundant re-prefilling in agentic contexts, `SemanticAnchorCache` holds backend state metadata at conversation turn, tool call, tool result, and thinking boundaries. Raw prompts and token sequences are not stored; only session/prefix SHA-256 hashes, token positions, opaque state handles, and accounted bytes are managed.

The cache is a thread-safe LRU with hard upper bounds on entry counts and state bytes. `put`, `resize`, and `clear` return evicted anchors, allowing the backend to reliably free corresponding KV/recurrent states. `deepest_reusable` returns only the deepest matching prefix anchor after context edits.

This is a backend-neutral standalone implementation inspired by FreeToken's semantic-aware caching, without dependencies on FreeToken, CUDA, or NVIDIA runtimes.

`SemanticStateCoordinator` bridges daemon `RuntimeService` with backend-owned state. Backends return opaque handles and accounted bytes on capture, and implement restore/release handlers. Stale handles are evicted from cache on restore failures, and release failures enter a retry queue capped at 1024 items. Runtime snapshots log capacity, resident bytes, captures, hits/misses, evictions, and restore/release failures. The current OpenAI HTTP proxy lacks KV handle APIs, so this is disabled by default.

`ElasticMemoryController` scales semantic cache baseline capacity down to 1/2 under Warning memory pressure and 1/8 under Critical pressure, restoring capacity when returning to Normal. Capacity changes are kept pending during active scheduler reservations and applied at safe points. Explicit pressure inputs are currently implemented, with continuous macOS memory pressure notification integration planned next.

## Model Optimization Compiler Dry-Run

O0 `vllm-apple-optimize plan` estimates output size, disk requirements, and peak memory for INT8/INT4 candidates without modifying the model. It detects paths overlapping with the original model, existing artifact paths, or resource budget overruns. In environments and models supported by MLX adapters, candidates are presented as runnable; otherwise, they are rejected with reasons.

Bounded I/O profiling is run explicitly only when estimating execution duration. Read/write samples are capped at 64 MiB each by default, and temporary files are removed after `fsync`. `plan` itself does not execute benchmarks implicitly.

O1 adapter capability detection inspects package metadata, platform, and model format/dtype without importing or executing external packages. The current MLX exporter requires Apple Silicon, safetensors, FP16/BF16/FP32, MLX/MLX-LM 0.26.x through 0.31.x, and 4/8-bit affine quantization to be deemed runnable. Unverified versions are not run speculatively, returning structured errors instead.

The conversion worker infrastructure provides a separate process, bounded stdout/stderr up to 64 KiB, process-group cancellation, and private sibling workspaces. Upon success, regular files, file counts, byte sizes, and directory depth are stream-validated, and all files/directories are `fsync`ed before atomically renaming the artifact directory. Outputs are not published on failure or cancellation. MLX exporters connect to workers via fixed argument vectors without shell execution, remote upload, or remote code trust. Source fingerprints stream SHA-256 in 8 MiB chunks across all regular files including weights, avoiding loading full models into memory. If safetensors dtypes are missing from config files, headers are inspected up to a 16 MiB limit without loading weights.

Persistent checkpoint v1 binds to plan, source fingerprint, output, command/environment fingerprint, and output byte budget. Checkpoint files convert plan IDs to SHA-256 filenames, saved atomically with `0600` permissions in private directories. Only `converted` states resume from workspace validation; earlier stages restart conversion from scratch. Resumptions are enabled explicitly to avoid accidental operation.

Workers acquire cross-process `flock` locks per checkpoint file. Duplicate runs of the same plan are rejected, and locks automatically release on abnormal process termination. With `resume=True`, bindings are re-validated, skipping command execution from `converted` workspaces directly to validation/promotion. Halts between promotion and completed checkpoint updates reconcile state by re-validating published artifacts. Upon success, a `0600` sidecar manifest (recording artifact tree hash, size, file count, elapsed milliseconds, peak child RSS) and a versioned export report JSON (formatted for easy Mac app parsing) are generated.

```bash
python3 -m vllm_apple.optimizer.cli capabilities /path/to/model

python3 -m vllm_apple.optimizer.cli profile /path/to/model \
  --workspace /path/to/temporary-workspace \
  --sample-mib 64 > optimizer-profile.json

python3 -m vllm_apple.optimizer.cli plan /path/to/model \
  --output /path/to/immutable-artifact \
  --objective memory \
  --max-memory-gb 16 \
  --max-disk-gb 40 \
  --max-duration-seconds 3600 \
  --performance-profile optimizer-profile.json \
  --license apache-2.0

# Dry-run: Displays invocation JSON without creating output or checkpoints
python3 -m vllm_apple.optimizer.cli export /path/to/model \
  --output /path/to/immutable-artifact \
  --checkpoint-root /path/to/existing-private-parent/checkpoints \
  --plan-id my-plan --bits 4 --group-size 64 --max-output-gb 20

# Explicit execution. Pass --resume for safe restart after interruption
python3 -m vllm_apple.optimizer.cli export /path/to/model \
  --output /path/to/immutable-artifact \
  --checkpoint-root /path/to/existing-private-parent/checkpoints \
  --plan-id my-plan --bits 4 --group-size 64 --max-output-gb 20 --execute

# Baseline and candidate are evaluated sequentially in separate processes; never loaded simultaneously
python3 -m vllm_apple.optimizer.cli evaluate /path/to/baseline \
  --dataset docs/evaluation/smoke-multilingual-v1.jsonl \
  --output /path/to/baseline-evaluation.json
python3 -m vllm_apple.optimizer.cli evaluate /path/to/candidate \
  --dataset docs/evaluation/smoke-multilingual-v1.jsonl \
  --output /path/to/candidate-evaluation.json
python3 -m vllm_apple.optimizer.cli quality-gate \
  --baseline /path/to/baseline-evaluation.json \
  --candidate /path/to/candidate-evaluation.json \
  --max-perplexity-regression 0.10 \
  --output /path/to/quality-gate.json

# Compare fixed-seed, greedy generation in separate processes
python3 -m vllm_apple.optimizer.cli generate-evaluate /path/to/baseline \
  --dataset docs/evaluation/generation-smoke-multilingual-v1.jsonl \
  --max-samples 6 --max-new-tokens 16 \
  --output /path/to/baseline-generation.json
python3 -m vllm_apple.optimizer.cli generate-evaluate /path/to/candidate \
  --dataset docs/evaluation/generation-smoke-multilingual-v1.jsonl \
  --max-samples 6 --max-new-tokens 16 \
  --output /path/to/candidate-generation.json
python3 -m vllm_apple.optimizer.cli generation-quality-gate \
  --baseline /path/to/baseline-generation.json \
  --candidate /path/to/candidate-generation.json \
  --min-token-agreement 0.70 --max-expectation-regression 0 \
  --output /path/to/generation-quality-gate.json

# Domains and languages can be specified repeatedly; selection criteria are included in fingerprints
python3 -m vllm_apple.optimizer.cli generate-evaluate /path/to/model \
  --dataset docs/evaluation/task-suite-multilingual-v1.jsonl \
  --chat-template --max-prompt-tokens 4096 \
  --domain code --domain mathematics --language ja \
  --output /path/to/ja-code-math.json
```

Duration returns `null` with a logged warning unless empirical profile measurements exist for the same hardware. CLI failures return JSON to stderr containing `code`, localizable `message_key`, and `recoverability`. The perplexity runner processes JSONL line-by-line, enforcing hard upper bounds of 16 MiB dataset size, 72 KiB per line, 4096 samples, 4096 tokens per sample, and 1,000,000 total tokens. Quality gates require identical dataset fingerprints, slices, and token counts, explicitly indicating un-evaluated generation, long-context, code, math, and safety in reports. The generation runner supports up to 64 prompts of 256 tokens each, excluding generated text and prompts from saved reports. Only bounded token IDs required for comparison, SHA-256 fingerprints, and expected string match scores are retained. Expectations explicitly specify `contains` or `prefix` on the dataset side. Short numerical answers use `prefix` to avoid false positives (e.g., `8` matching inside `18`). Instruction models require `--chat-template`, recording post-template input token counts for each sample. Prompt limits and model context limits are validated prior to generation; reports with mismatching prompt formats, token budgets, or actual input token counts between baseline and candidate are rejected for comparison.

## Development Verification

See [DEVELOPMENT.md](docs/DEVELOPMENT.md) for reproducible development environment setup and CI details.
See [RUNTIME-FAILURES.md](docs/RUNTIME-FAILURES.md) for runtime error recoverability and private crash diagnostics.
See [MEMORY-TELEMETRY.md](docs/MEMORY-TELEMETRY.md) for source-aware metrics across Unified Memory, allocators, and KV caches.

```bash
make bootstrap
make check
```

## Architecture Metadata Inspection

```bash
python3 -m vllm_apple inspect-architecture /path/to/model/config.json
```

Describes local architecture metadata without loading weights. The initial registry covers Llama, Qwen2, Qwen3, Mixtral, and Gemma2 structural recipes. Exit code 0 indicates metadata was successfully described; backend execution remains unverified.

See the [architecture support plan](docs/LLM-ARCHITECTURE-SUPPORT.md) for scope, limitations, and remaining runtime integration work.

`inspect-model` now emits recommendation schema v2. Feature declarations only establish `eligible_for_validation`; they do not certify backend compatibility or set `runnable=true`. Its metadata-only result exits with code 1 (input errors: 2).

Optional local evidence: `qualify-model --bind-architecture-evidence` produces an identity-bound 30-minute text report; `inspect-model` and `serve` accept it through `--architecture-evidence`. See the [protocol and limits](docs/LLM-ARCHITECTURE-SUPPORT.md#local-evidence-startup-gate-2026-09-26). Gemma 2 2B passed a bound 30-minute run on M4 with Homebrew MLX-LM 0.32.0 (6,491/6,491 requests). After the startup-gate change, a fresh run passed 5,995/5,995 requests, and managed `serve` passed multilingual/stream/shutdown checks without `--skip-backend-check`. Valid matching evidence permits that exact MLX candidate; the blanket version matrix remains unchanged.