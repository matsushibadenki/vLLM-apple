# Gemma 3 text support

The structural registry recognizes `gemma3_text`. This does not recognize the multimodal `gemma3` wrapper or certify arbitrary Gemma checkpoints.

The recipe preserves explicit head dimensions, Q/K normalization, embedding scaling, sandwich normalization, separate local/global RoPE bases and query scaling. A positive `sliding_window_pattern` defines one global layer at the end of each period; other layers retain windowed state. An explicit layer list must agree with that pattern. Nonpositive/bool values and periods exceeding the layer count are rejected.

The 1B fixture has 26 layers, a period of six and global layers 5, 11, 17 and 23 (zero-based). Structural descriptions remain unverified until identity-bound execution evidence is available. Evidence remains specific to the model artifact, backend build, wrapper sources, environment, Mac, context and concurrency.

Sources: [Hugging Face Gemma 3 documentation](https://huggingface.co/docs/transformers/model_doc/gemma3) and the installed MLX-LM `models/gemma3_text.py` implementation.

## Validation

- Structural fixtures and rejection tests: `python -m unittest tests.test_architecture_registry`.
- Evidence and qualification regression tests: `python -m unittest tests.test_architecture_evidence tests.test_model_recommendation tests.test_qualification`.
- Runtime qualification uses `qualify-model` with `--bind-architecture-evidence`, a full 1800-second soak, multilingual quality checks and phase samples. A candidate backend version is checked against its exact installed version; this does not globally expand the verified version matrix.

Runtime qualification results are recorded after the real run completes. Do not interpret structural recognition as execution approval.

## Real qualification, 2026-10-30

`google/gemma-3-1b-it` revision `dcc83ea841ab6100d6b47a070329e1ba4cf78752` passed with MLX-LM 0.32.0 on the development Apple M4. The bound scope is text, context 512, concurrency 1. English/Japanese/Simplified Chinese quality checks passed. The soak ran 1800.017 seconds with 6770 successes and zero errors; peak RSS growth was 12,189,696 bytes against a 268,435,456-byte limit, and the backend shut down cleanly.

Soliton's native GUI accepted the evidence, started the model and returned `2` to a Japanese arithmetic question with output limit 64 and temperature 0. A larger output limit still must fit alongside the prompt inside the qualified context; the evidence does not relax that admission check.

The owner-only qualification JSON stays in the local private runtime directory. It is not a portable certificate or a new global version-matrix entry. New environments, backend builds, wrapper source changes, model artifacts or expired evidence require qualification again. A sanitized GUI/qualification summary is stored in the Soliton repository at `docs/evaluation/gemma3-gui-20261030.json`.
