# VLLMAppleChat Xcode app target

This sample generates a macOS Xcode application target that embeds `vllm-appled` as an auxiliary executable.
The Swift source and English, Japanese, and Simplified Chinese resources are shared with `samples/VLLMAppleChat`.

```bash
brew install xcodegen
cd samples/VLLMAppleChatXcode
xcodegen generate
VLLM_APPLE_DAEMON_SOURCE=/absolute/path/to/vllm-appled \
  xcodebuild -project VLLMAppleChat.xcodeproj -scheme VLLMAppleChat build
```

`VLLM_APPLE_DAEMON_SOURCE` must point to a standalone, executable, non-symlink artifact. The build phase copies it to
`VLLMAppleChat.app/Contents/MacOS/vllm-appled`, where `RuntimeResourceResolver` discovers it with
`Bundle.url(forAuxiliaryExecutable:)`. Mutable profiles, logs, sockets, tokens, and model files remain outside the signed
application bundle.

The app sandbox is disabled because the sample launches a separately signed local daemon and user-selected model backend.
Production distributions should sign the app and daemon with the same team, retain hardened runtime, and notarize the final
application. The later signed-daemon roadmap item covers creation of that standalone distribution artifact.
