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

## Unsigned release candidate

On an Apple Silicon Mac, install the locked packaging dependency and create a self-contained app archive:

```bash
python3 -m pip install . -r requirements-package.lock
scripts/build_mac_integration_package.sh /absolute/output/directory
```

The command produces `VLLMAppleChat-unsigned-arm64.zip` and a matching SHA-256 file. The app contains a standalone
`vllm-appled`; the target Mac does not need a separate Python installation. This artifact is deliberately unsigned and is
for integration testing only. The `Mac integration package` GitHub workflow builds the same candidate on `macos-15`.

## Signed and notarized release

The manually triggered `Mac notarized release` workflow imports a Developer ID Application certificate into an ephemeral
keychain, signs the embedded daemon before the containing app, submits the archive to Apple, requires an `Accepted` result,
staples the ticket, and runs both `codesign` and Gatekeeper verification. Signing material is removed in an `always()` step.

Configure the protected `mac-release` GitHub environment with these secrets:

- `APPLE_DEVELOPER_ID_CERTIFICATE_P12_BASE64`
- `APPLE_DEVELOPER_ID_CERTIFICATE_PASSWORD`
- `APPLE_DEVELOPER_ID_APPLICATION`
- `APPLE_BUILD_KEYCHAIN_PASSWORD`
- `APPLE_NOTARY_KEY_P8_BASE64`
- `APPLE_NOTARY_KEY_ID`
- `APPLE_NOTARY_ISSUER_ID`

The uploaded artifact contains the stapled `VLLMAppleChat-notarized-arm64.zip`, its SHA-256 file, Apple's bounded
notarization result, and `release-manifest-v1.json`. The manifest binds the archive and its two executables to the app
version, source commit, runner, and accepted notarization submission. GitHub also creates an OIDC build-provenance
attestation for the final ZIP. Keep environment approval enabled so release credentials are never available to ordinary CI
jobs.

Consumers can repeat the bounded integrity check without signing credentials:

```bash
vllm-apple-release-manifest verify VLLMAppleChat-notarized-arm64.zip \
  --notary-report notary-result.json \
  --manifest release-manifest-v1.json
```

## Draft release promotion

After the notarization workflow succeeds, create an exact version tag on the manifest's source commit and manually run
`Mac draft release promotion` with that workflow run ID and tag. The protected job downloads the existing artifact and
reverifies its SHA-256, bounded manifest, GitHub build attestation, and tag-to-commit binding. Only then does it create a
draft GitHub Release. Publication remains a separate human approval step.
