# Stream transport metrics

## 日本語

`run_phase_probe`と同じprofilerを使うqualification reportに、任意の`transport`フィールドを追加した。schema v1の既存必須項目とdecode計算は維持し、旧reportも更新後schemaで検証できる。旧schemaを固定したstrict consumerは、新しい任意フィールドの定義を取り込む必要がある。

- `end_to_end`：リクエスト開始からSSE `[DONE]`受信まで。
- `stream_tail`：最後の生成content受信から`[DONE]`受信まで。usage、終了通知、ネットワークやclient側待機を含み、serialization単独の時間ではない。
- `sample_count`／`unavailable_sample_count`：通信完了timestampがある／ない計測数。すべて未取得ならlatencyは`null`。欠測を0 msで補完しない。

mean、p50／p95のhistogram上限、maxを記録する。4系列・52 bucketの固定容量で、生sampleや生成本文を保存しない。`PhaseMeasurement.completed_ns`は互換性のため最後の生成content到着を表し、新しい`stream_done_ns`が通信完了を表す。途中切断やusage欠落は従来通り失敗する。

HTTPで観測するcontentにはreasoningも含む。SSE chunkはtokenと一対一ではない。既存TTFTは最初の生成content到着、既存TPOTはcontent到着区間とusage token数による集計で、backendのtoken単位timestampや純粋なprefill／decode時間を保証しない。queue、tokenize、model load、goodput、失敗を含む共通比較reportへの統合は未完了。今回の変更は計測機能であり、速度向上や実機認定を意味しない。

検証コマンド（loopback HTTP待受が可能な環境）：

```bash
.venv/bin/python -m unittest tests.test_phase_profile tests.test_phase_probe tests.test_schemas tests.test_qualification tests.test_qualification_bundle tests.test_qualification_preflight tests.test_phase_resource_profile tests.test_mlx_phase3_probe tests.test_mlx_candidate_qualification_cli
ruff check vllm_apple/phase_profile.py vllm_apple/phase_probe.py tests/test_phase_profile.py tests/test_phase_probe.py
```

## English

Phase reports now include optional transport metrics: request start to SSE `[DONE]`,
and last generated content to `[DONE]`. Missing observations remain `null` with a
separate unavailable count. Existing decode calculations are unchanged. Histograms
use constant memory; percentiles are bucket upper bounds. Content includes reasoning;
SSE chunks are not tokens, and these client measurements do not isolate backend
compute or serialization. No real-model performance improvement is claimed.

## 简体中文

阶段报告新增可选的通信指标：从请求开始到SSE `[DONE]`，以及从最后生成内容到
`[DONE]`的时间。缺失数据保持为`null`并单独计数，原有decode计算保持不变。
直方图使用固定内存，分位数表示桶的上界。生成内容包括推理文本；SSE块不等同于
token，客户端计时无法单独衡量后端计算或序列化。本次未进行真实模型的性能认证。
