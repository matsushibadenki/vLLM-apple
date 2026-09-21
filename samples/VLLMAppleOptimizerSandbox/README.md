# VLLMAppleOptimizerSandbox

App Sandbox内で動作し、独立管理されたloopback daemonへdry-run planだけを要求するmacOS clientです。
外部processの探索・起動・signal操作は含みません。

```bash
cd samples/VLLMAppleOptimizerSandbox
xcodegen generate
xcodebuild -project VLLMAppleOptimizerSandbox.xcodeproj \
  -scheme VLLMAppleOptimizerSandbox -configuration Debug CODE_SIGNING_ALLOWED=NO build
```

daemonは`--optimizer-model-root`と`--optimizer-output-root`の双方を指定して起動してください。
