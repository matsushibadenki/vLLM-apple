# Gemma 2 batch mask compatibility

## 日本語

MLX-LM 0.32.0の確認済みGemma 2実装は、Grouped Query Attentionでscoreを
`[B, KV heads, repeats, Q, K]`へ分ける一方、batch用maskを`[B, 1, Q, K]`のまま適用する。
batch軸がhead軸へ誤って対応し、並列度2ではbroadcast例外で生成threadが終了した。

`mlx_gemma2_compat`は、共有headの4次元maskだけを`[B, 1, 1, Q, K]`へreshapeして
元のattentionへ渡す独立実装。bool／加算maskとも値を変えず、weight、RoPE、cache、
softcap、softmaxは元の演算を使う。2次元mask、非GQA、maskなしの経路は変更しない。
対応しないhead別4次元maskは明示拒否する。

適用条件はMLX-LM `0.32.0`と`models/gemma2.py`のSHA-256
`64b0935b06fe2c4d5d4ed23a9cf62deb6218c55a88b9403a657afe9e2be8f251`の一致。
一致しなければ起動を拒否し、新版への暗黙適用を防ぐ。インストール済みファイルは書き換えない。
通常の`serve`／`mlx_server`は変更せず、専用の実験用entry pointでだけ適用する。

repository rootから、MLX入りPythonで起動する：

```bash
PYTHONPATH=. /opt/homebrew/opt/vllm-metal/libexec/bin/python \
  -m vllm_apple.mlx_gemma2_compat \
  --model models/gemma-2-2b-it-4bit --host 127.0.0.1 --port 19096 \
  --decode-concurrency 2 --prompt-concurrency 2 --prompt-cache-size 4
```

引数解析とserver lifecycleはupstreamへ委譲する。[benchmark手順](TEXT-BENCHMARK.md)で
3件warmup後、並列度1／2で各30件を測定した。[実測report](evaluation/text-benchmark-m4-gemma2-mask-fix-2026-09-26.json)は
両方30/30正答・HTTP失敗0。試験processはSIGTERM後に回収済み。速度向上や自然shutdownの認定ではない。

GPU数値回帰はbatch 2／3 × query長1／4 × bool／加算maskの8条件で、各batch行の異なる
maskを使用し、未修正の逐次attentionと`atol=rtol=1e-5`で一致した。実行コマンド：

```bash
VLLM_APPLE_TEST_GEMMA2_MASK=1 /opt/homebrew/opt/vllm-metal/libexec/bin/python \
  -m unittest tests.test_mlx_gemma2_compat
```

数値回帰は小型F32 attention、HTTP試験は配置済み4-bit実モデルの短い算術promptに限定する。
長文・prefix編集・cancel・復旧・長時間並列負荷は未認定。既存の並列度1用証跡を、
この新しい実行経路の証跡として流用しない。rollbackは専用起動をやめ、既存の認定経路へ戻す。

## English

An explicit, version- and source-hash-bound launcher reshapes shared-head batched
Gemma 2 masks for grouped-query scores. It preserves upstream computation and never
modifies installed packages. M4 HTTP smoke passed 30/30 requests at concurrency 1
and 2; eight GPU cases matched unpatched per-row attention within 1e-5 tolerance.
Long-context, cancellation, recovery and sustained-load qualification remain pending
work on this M4. Managed serving is unchanged.

## 简体中文

专用启动入口仅对版本和源码哈希匹配的Gemma 2补齐批处理mask的分组维度，保留上游计算，
不修改已安装的软件包。M4上的并发度1和2各通过30/30次HTTP测试，8组GPU数值测试与未修改的
逐条attention在1e-5容差内一致。长上下文、取消、恢复和持续负载仍需在本机验证，
标准托管启动路径保持不变。
