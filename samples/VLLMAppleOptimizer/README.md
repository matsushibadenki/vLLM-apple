# VLLMAppleOptimizer

macOS 13以降で動作する、`vllm-apple-optimize`用のSwiftUI companion appです。
元モデルを変更せず、最初にdry-run計画だけを生成して候補、disk、peak memory、所要時間、
blocking reasonを表示します。

```bash
cd samples/VLLMAppleOptimizer
VLLM_APPLE_OPTIMIZER_EXECUTABLE="$(command -v vllm-apple-optimize)" swift run
```

App Sandbox client用の独立daemon transportを利用する場合は、daemonをloopbackへ限定し、modelと
outputの許可rootをそれぞれ明示します。片方だけの指定、remote bind、symlink／他ユーザー所有rootは
起動前に拒否されます。

```bash
vllm-apple serve \
  --optimizer-model-root /absolute/path/to/models \
  --optimizer-output-root /absolute/path/to/output
```

実行ファイルはshellを介さず直接起動します。通常file、非symlink、実行可能であることを検証し、
stdout/stderrを各1 MiB、計画生成を120秒に制限します。model/outputはユーザーが選択した
security-scoped URLとして処理中だけアクセスします。選択結果は最大1 MiBのsecurity-scoped bookmarkとして
保存し、再起動時はUIを出さずに復元します。stale bookmarkは更新し、壊れたbookmarkは破棄します。

budget内かつadapterが実行可能な候補は、確認dialog後にcheckpoint付きで変換できます。cancelは
CLIのsignal-safe cancellationへ接続し、同じplanからの明示resumeに対応します。完了後はartifact ID、
hash、size、peak RSS、license、品質評価の有無を表示します。stage別progress、checkpoint pause、
original/optimizedのperplexity品質比較は、同一のbounded JSONLとslice別quality gateで利用できます。
stage別progressとconverter process groupの安全なpause／continueにも対応します。generation responseは
同じbounded入力を用いるtoken-level side-by-side比較に対応します。
