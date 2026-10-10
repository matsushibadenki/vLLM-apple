# Hardware conditions and P4 collector connection — 2026-10-10

## 日本語

🟢 [Done] P3のhardware identityにGPUコア数を追加した。sysctlから取得できなければ既存AGXAcceleratorのioreg取得へfallbackする。未取得・不正値は実測設定の適用を拒否する。同じSoC名でもGPUコア数が異なる測定を流用しない。個体のserial numberは収集しない。

GPU profile、HTTP候補取得、設定適用に電源source／power mode／thermal観測を接続した。nominalかつ既知の電源条件のみ許可し、取得前後・各worker開始時の条件変更を拒否する。HTTP試験後の変更はidentity quality sliceを不合格にする。これはsample時点の観測で、観測間の短い変化や他アプリの干渉を全て検出した意味ではない。常時監視やworker途中停止は追加しない。

[実観測](evaluation/p3-operating-conditions-m4-2026-10-10.json)：M4の10 GPU cores、AC Power／automatic／nominal、前後一致を確認した。P1固定runtime／runnerは不変。旧P3証拠は新しい条件を持たず、変更後sourceとも一致しないため実測設定の適用に使えない。過去の結果を保存し、性能認定を遡及しない。

🟢 [Done] `scripts/collect_p4_evidence.py`で実collector結果を既存P4 gateへ接続した。入力は有界なregular JSON file、role重複は拒否し、output directoryは新規作成のみ。元collector内容・input SHA・保存内容SHAを保持する。欠けたroleは明示的な不合格raw evidenceとして渡し、全roleを既存gateが評価する。

P4用collectorは`p4_identity`の5 hash（hardware／os／model／backend／configuration）、role、source commit、artifact SHAの一致を要求する。性能reportは既存P3 policy scope／report_id／独立取得／標準採用条件もgateが検証する。未対応の旧形式は診断として保存し、内容を一般品質やP4資格へ昇格させない。P3 policy scopeをP4用に書き換えない。

[実collector接続結果](evaluation/p4-real-collector-m4-2026-10-10/certification.json)では、P1の112秒report、P2の限定品質判定、P3の不採用判定がgateへ届き、品質範囲不足・性能未認定・24時間不足・復旧等の証拠欠落により昇格を拒否した。接続確認用artifact SHAはcollector sourceのhashで、署名release artifactではない。releaseを認定した結果ではない。

```sh
PYTHONPATH=. .venv/bin/python scripts/collect_p4_evidence.py \
  --output-directory qualification-results/p4-new-run \
  --identity path/to/exact-five-hash-identity.json \
  --source-commit SOURCE_COMMIT_40_HEX \
  --artifact-sha256 ARTIFACT_SHA256_64_HEX \
  --evidence quality=path/to/quality.json \
  --evidence performance=path/to/p3-selection.json \
  --evidence soak=path/to/p1-final-report.json
```

不合格時のexit code 1は認定拒否を表し、`certification.json`へ理由を保存する。roleはquality／performance／soak／recovery／rollback／installation／uninstallation／dependencies／licenses。全条件が揃った合成fixtureの合格経路と、欠落・identity不一致・hash改変・短時間soak・未認定P3の拒否経路を試験した。合成fixtureは実機資格ではない。

最終検証：関連23 tests、全回帰1,513 tests（31 skip）、Ruff・diff検査成功。[全回帰log](evaluation/hardware-p4-final-regression-2026-10-10.log)。[旧証拠の実CLI起動拒否](evaluation/p3-old-evidence-rejection-2026-10-10.json)はworker開始前・receipt未作成。[P4実CLI](evaluation/p4-real-cli-validation-2026-10-10.json)も実reportを読み、認定拒否のexit 1と理由保存を確認した。

## English

🟢 [Done] P3 identity now includes GPU core count. Measured settings require known, matching power source/mode and nominal thermal state before and after acquisition/application; unknown, unsafe or changed observations reject application. Live M4 observation confirms 10 GPU cores and matching AC/automatic/nominal conditions. Sampling does not guarantee uninterrupted conditions or host exclusivity. Pinned P1 sources are unchanged; historical P3 evidence cannot qualify the new sources.

🟢 [Done] The P4 collector CLI preserves real input payloads and hashes, assembles all required roles and invokes the existing certification gate. Exact identity/source/artifact binding is required; missing and legacy unscoped results fail closed. Real P1/P2/P3 results reach the gate and correctly block promotion. The diagnostic artifact hash is a collector source hash, not a signed release. Complete synthetic success and negative cases are tested. This completes these two implementation tasks, not 24-hour certification or standard profile adoption.

Validation: 23 targeted tests and 1513 full regression tests pass (31 skips); Ruff and diff checks pass. Actual CLIs reject stale P3 evidence before worker launch and reject uncertified P4 inputs with persisted reasons.

## 简体中文

🟢 [Done] P3 identity已加入GPU核心数。应用测量配置要求已知且一致的电源source／mode及nominal thermal状态；未知、不安全或前后变化时拒绝应用。M4实测为10 GPU cores，AC／automatic／nominal前后一致。采样不保证整个运行期间条件恒定或host独占。P1固定source保持不变，旧P3证据不能认证新source。

🟢 [Done] P4 collector CLI保留真实输入及hash，组装全部role并调用现有认证gate。要求exact identity／source／artifact一致，缺失或旧无scope结果拒绝晋升。真实P1／P2／P3结果已到达gate并正确拒绝认证。诊断artifact hash来自collector source，不是已签名release。已验证完整合成成功路径及负面案例。本次完成两个实现任务，不等于24小时认证或标准profile采用。

验证：23项相关tests及1513项全量回归通过（31 skip），Ruff／diff通过。实际CLI在启动worker前拒绝旧P3证据，P4未认证输入被拒绝并保存原因。
