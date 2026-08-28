# Runtime Failures and Crash Diagnostics

runtime failureは自由文ではなく、versionedな次のfieldsでAPIとSSEへ公開する。

- `code`：機械判定用の安定identifier
- `message_key`：英語、日本語、简体中文UIでlocalizeするkey
- `recoverability`：`retryable` / `user_action_required` / `fatal`
- `detail_fingerprint`：同一原因を集計するSHA-256由来24桁fingerprint

raw exception、model path、command、backend logはAPIとSSEへ含めない。互換性維持用の`last_error`にも
自由文ではなく`message_key`だけを返す。Swift SDKは`RuntimeEvent.runtimeFailure`から型付きで復旧可能性を
取得できる。

backend launch failure時は`~/Library/Application Support/vllm-apple/diagnostics/`へ0600 JSONをatomic
保存する。diagnosticにはfailure、時刻、直近最大80行の件数とdigestだけを保存し、log本文は保存しない。
directoryは0700へ制限する。Macアプリは`diagnostic_id`をsupport情報として提示できるが、local pathや
機密情報を送信しない。

| Code | Recoverability |
| --- | --- |
| `backend_startup_failed` | retryable |
| `backend_exited` | user action required |
| `backend_readiness_timeout` | retryable |
| `backend_incompatible` | user action required |
| `memory_capacity_exceeded` | user action required |
| `execution_plan_rejected` | user action required |
| `internal_error` | fatal |
