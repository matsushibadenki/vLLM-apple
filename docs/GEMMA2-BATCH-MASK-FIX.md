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
既存の並列度1用証跡を、
この新しい実行経路の証跡として流用しない。rollbackは専用起動をやめ、既存の認定経路へ戻す。

追加の短時間qualificationは、約2K prompt tokensの同一prefix末尾を3種類に編集した
並列12件と、三言語算術の並列100件を実行した。いずれも完了・品質・設定SLOが全件合格し、
継続負荷のgoodputは19.283 output tokens/s、観測peak RSSは2,232,991,744 bytesだった。
長文側p95 TTFTはhistogram範囲外の`>5000 ms`、最大7,249.752 msであり、短い入力の
性能値として扱わない。SSEを最初のdata受信後にclientから閉じ、1秒後の正常応答も確認した。
ただしupstream serverからcancel完了通知は得られず、backend処理の停止・KV解放完了は未確認。
結果は[qualification report](evaluation/gemma2-batch-mask-m4-qualification-2026-09-26.json)に保存した。

再現コマンド：

```bash
.venv/bin/python scripts/qualify_gemma2_batch_mask.py \
  --python /opt/homebrew/opt/vllm-metal/libexec/bin/python \
  --model models/gemma-2-2b-it-4bit \
  --output docs/evaluation/gemma2-batch-mask-m4-qualification-2026-09-26.json \
  --port 19097 --sustained-requests 100 --long-requests 12
```

runnerは固定数workerを使い、reportをatomic保存する。仮想環境launcherのsymlinkを保持して
`sys.prefix`を変えず、local modelだけをofflineで起動する。これは約36秒の回帰試験であり、
30分認定、明示的cancel、遅いconsumer、一般的な長文品質は未認定。

## English

An explicit, version- and source-hash-bound launcher reshapes shared-head batched
Gemma 2 masks for grouped-query scores. It preserves upstream computation and never
modifies installed packages. M4 HTTP smoke passed 30/30 requests at concurrency 1
and 2; eight GPU cases matched unpatched per-row attention within 1e-5 tolerance.
A short M4 regression passed 12 shared-prefix edits and 100 sustained concurrent
requests, then recovered after a client-side stream disconnect. Backend cancellation
completion, slow consumers and a 30-minute mixed load remain unqualified. Managed
serving is unchanged.

## 简体中文

专用启动入口仅对版本和源码哈希匹配的Gemma 2补齐批处理mask的分组维度，保留上游计算，
不修改已安装的软件包。M4上的并发度1和2各通过30/30次HTTP测试，8组GPU数值测试与未修改的
逐条attention在1e-5容差内一致。追加测试通过12次共享前缀编辑、100次并发持续请求，
并在客户端中断stream后恢复。后端取消完成、慢速客户端及30分钟混合负载仍未认证，
标准托管启动路径保持不变。
