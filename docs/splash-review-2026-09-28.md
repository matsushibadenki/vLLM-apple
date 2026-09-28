# Splash review — 2026-09-28

参照日時：2026-09-28（Asia/Tokyo）。公式GitHub repository、README、
`DEVELOPMENT.md`、license、remote refsを確認した。

| 項目 | 確認結果 |
| --- | --- |
| Repository | [incoai/splash](https://github.com/incoai/splash) |
| 最新release tag | `1.1.0`、commit `3e1f9ece3e2528f3eb46b82a05911591f34a4317` |
| 確認時HEAD | `c64a578fcfaa059023a2ca57230d4aead134b73b` |
| License | Apache-2.0。GGUF kernelにはllama.cpp由来のMIT materialがあるとupstreamが明記 |
| 主な要件 | Apple M3以降、macOS 26.4以降。主要4-bit例は36 GiB以上、48 GiB推奨。小型GGUFは24 GiBでも利用可能 |

## 採用判断

Splash engine、DFlash2、専用Metal kernelは導入しない。現在の認定対象であるM4／32 GiB／
Gemma 2 artifactと対応model familyが一致せず、別modelやdraftのdownload、独立した品質・性能
認定が必要になるためである。READMEの性能値はM5 Pro等の別条件であり、本プロジェクトの
性能値として扱わない。P0のbackend比較候補としては維持する。

採用した考え方は、request単体の上限とは別に、同時に保持するHTTP入力byteをserver全体で
予約すること。既存APIには4 MiBの単体上限とrequest数上限があったが、最大サイズに近いJSONが
並列到着した時の合計byte予算がなかった。Splashの実装コードは転載せず、次を独立実装した。

- default 16 MiB、設定可能範囲4 MiB〜1 GiBのaggregate入力byte予算
- `Content-Length`検証後、JSON read前に原子的に予約
- streamを含むrequest処理終了まで予約を維持し、失敗・timeoutでも`finally`で解放
- 予算超過は`503 request_bytes_exhausted`。単体4 MiB超過の`413 request_too_large`と区別
- `/v1/runtime.http`へ現在値、peak、上限、拒否件数を公開
- TCPとUnix domain socketで同じ予算を使用

model revisionをcommitへ解決してatomic assemblyとして公開する考え方、model geometry／
quantization／draft compatibilityを実行前に再検証する考え方も、このプロジェクトのidentity-bound
qualificationと整合する。今後のartifact管理を強化する際の設計参照にするが、今回の変更では
Splashのinstallerやweight形式を導入していない。

## English

Splash 1.1.0 and HEAD were reviewed from the official repository. Its complete engine,
DFlash2 path, and Metal kernels were not adopted because the supported model artifacts
do not match the currently qualified M4/32 GiB Gemma 2 route. Upstream benchmark claims
remain scoped to their published hardware and models.

One applicable design was implemented independently: a server-wide budget for request
body bytes held concurrently. The runtime now reserves declared bytes before reading
JSON, retains the reservation through generation, releases it on every exit path,
returns `503 request_bytes_exhausted` when capacity is unavailable, and reports current,
peak, limit, and rejection counters through `/v1/runtime.http`. No Splash source code
was copied.

## 简体中文

已检查Splash 1.1.0及官方仓库HEAD。由于其支持的模型artifact与当前已认证的M4／32 GiB／
Gemma 2路径不一致，本次没有引入完整engine、DFlash2或专用Metal kernel。上游性能数据仍只适用于
其公布的硬件和模型条件。

本次独立实现了一个适用的设计：server级并发request body字节预算。runtime在读取JSON前预留
声明的字节，在整个生成期间保持预留，并在所有退出路径释放。容量不足时返回
`503 request_bytes_exhausted`，`/v1/runtime.http`公开当前值、峰值、上限和拒绝次数。
没有复制Splash源码。
