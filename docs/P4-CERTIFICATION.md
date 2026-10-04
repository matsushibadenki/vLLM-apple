# P4 certification / 配布認定 / 发布认证

## 日本語

[Done] 各対応matrixセルの認定証拠を検証するrelease gateを実装。
`python3 -m vllm_apple.p4_certification BUNDLE --source-commit COMMIT --artifact-sha256 SHA256`
は不合格・欠落・改変時に非zeroで終了する。既存の署名archive checksum、manifest、
GitHub attestation、tag commit確認に続き、`promote_mac_release.sh`がこのgateを実行する。
P4証拠がない旧release bundleは昇格できない。

`mac-draft-release.yml`に必須入力`certification-run-id`を追加した。
同じrepositoryの確認済みrunから`P4-certification` artifactを取得する。
そのartifactのrootには`p4-certification-v1.json`と全参照ファイルを含める。
現時点では合格済みartifactを生成する実機qualification workflowは未整備。
署名workflowの成功だけでP4 artifactを作成してはならない。

bundleはschema_version=1、source_commit（40桁hex）、artifact_sha256（署名済みarchiveのhash）、
cells（1〜64）を持つ。各cellのidentityはhardware／os／model／backend／configurationの
SHA256を固定する。構成値のcanonical JSONとhashの元資料は収集側で保存する。
同じidentityの重複は禁止。cellのevidenceには以下の全roleが必要：
quality、performance、soak、recovery、rollback、installation、uninstallation、dependencies、licenses。
各referenceはbundleからの相対pathとファイルのSHA256。
外部path・symlink・4MiB超のJSONを拒否する。

各roleのenvelopeはschema_version=1、role、scope_sha256（identityのcanonical JSONのSHA256）、
source_commit、artifact_sha256、passed=true、raw_evidence（path／sha256）を持つ。
raw collector出力もhash検証する。performance以外はrawのpassed=trueも必要。
短い既存試験を加工してpassed=trueにしたり、scopeやartifactを付け替えてはならない。
hashは改変検出用であり、測定事実や署名・独立性を証明するものではない。
収集者・CI run・raw evidenceをreviewし、既存の`mac-release` environmentで配布を管理する。

追加条件：qualityのslicesはen／ja／zh／coding／tool／long_contextを全てtrueとする。
performanceはP3 selectorの実reportを参照し、scope一致、report hash、独立取得、
standard_adoption_eligible=true、baseline_retained=falseを検証する。
soakはduration_seconds>=86400、clean_shutdown=trueを要求する。
recoveryのcasesにはos_update／backend_update／startup_failure／profile_corruptionが必要。
rollbackはrestored_profile_sha256とinference_verified=trueが必要。
installation／uninstallationはindependent_clean_machine_verified=true、
dependenciesはlock_verified=true、licensesはsbom_license_review_verified=trueを要求する。
これらのboolは実試験結果を記録する契約であり、gateが実機試験自体を実行するものではない。

[Done] `profile_activation`は正確なidentityと認定reportのhashを確認し、熱／メモリ制約を
優先してfallbackを返す。診断codeと説明は英語・日本語・简体中文で共通化した。
実runtime／Swift SDK／UIへの接続は未完了。report hashは認証ではないので、
信頼できるrelease bundleの検証結果のみを渡す。

[Next] P1のSLO不合格とP2／P3の未達を解消した対象構成で、実collectorのenvelope出力、
独立3回の性能認定、実24時間soak、障害注入・rollback、目的別profile選択を完結する。
24時間試験はP1の短時間試験が不合格の現状では開始しない。

[Later] [pending] 別SoC・RAMの実機runner、Developer ID／notary資格情報、
独立clean-machine環境を用意し、実署名・配布・install／uninstallを検証する。
P4全体は未完了であり、現時点で認定済みmatrixセルはない。

2026-10-04検証：新規7テストと関連releaseテスト18件成功、全回帰1425件成功（11 skip）。
Ruffとshell構文検証成功。合格fixtureは合成データであり、実24時間認定ではない。

## English

[Done] A bounded evidence gate now blocks Mac draft-release promotion unless every advertised
hardware/OS/model/backend/configuration cell has matching, hashed quality, performance,
24-hour soak, recovery, rollback, installation, uninstallation, dependency and license evidence.
The gate follows existing notarization, archive, attestation and tag checks. Missing P4 evidence
blocks legacy bundles. The workflow requires a reviewed `certification-run-id` containing a
`P4-certification` artifact. Hashes establish integrity, not measurement authenticity.
Profile activation checks exact identity and gives thermal/memory limits priority, with common
English/Japanese/Simplified Chinese diagnostics; it is not yet wired into the runtime or SDK/UI.

[Next] Implement real collector envelopes and complete independent performance qualification,
objective-specific profile selection, 24-hour testing and fault/rollback verification after
the outstanding P1/P2/P3 gates pass. No certified matrix cell currently exists.
[Later] [pending] Other Macs, signing credentials and an independent clean-machine environment
are required for remaining distribution certification. P4 is not complete.

## 简体中文

[Done] Mac草稿发布提升现已要求每个hardware／OS／model／backend／configuration单元
具有相同配置且hash匹配的质量、性能、24小时稳定性、恢复、回滚、安装、卸载、依赖及许可证证据。
此gate在现有签名archive、notarization、attestation和tag检查之后执行，缺少P4证据的旧bundle
不能提升。workflow必须指定经review的certification-run-id，包含P4-certification artifact。
hash只验证完整性，不证明测量真实性。profile启用检查准确identity，并优先处理温度和内存限制，
提供英语、日语、简体中文诊断；尚未接入实际runtime或SDK／UI。

[Next] 先解决P1／P2／P3未通过项，再对接实际采集器，完成独立性能认证、按目标选择profile、
24小时测试及故障／回滚验证。目前没有已认证的matrix单元。
[Later] [pending] 其他Mac、签名资格信息、独立clean-machine环境仍需准备。P4整体尚未完成。
