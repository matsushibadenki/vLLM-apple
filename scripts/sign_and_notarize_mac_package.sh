#!/bin/sh
set -eu

if [ "$#" -ne 2 ]; then
    echo "usage: $0 UNSIGNED_ZIP OUTPUT_DIRECTORY" >&2
    exit 64
fi

unsigned_zip=$1
output_directory=$2
: "${VLLM_APPLE_CODESIGN_IDENTITY:?VLLM_APPLE_CODESIGN_IDENTITY is required}"
: "${VLLM_APPLE_NOTARY_KEY:?VLLM_APPLE_NOTARY_KEY is required}"
: "${VLLM_APPLE_NOTARY_KEY_ID:?VLLM_APPLE_NOTARY_KEY_ID is required}"
: "${VLLM_APPLE_NOTARY_ISSUER:?VLLM_APPLE_NOTARY_ISSUER is required}"

if [ "$(uname -s)" != "Darwin" ]; then
    echo "error: macOS is required for signing and notarization." >&2
    exit 1
fi
if [ ! -f "${unsigned_zip}" ] || [ -L "${unsigned_zip}" ]; then
    echo "error: unsigned package must be a regular ZIP file, not a symlink." >&2
    exit 1
fi
case "${unsigned_zip}" in
    /*) ;;
    *)
        echo "error: unsigned package path must be absolute." >&2
        exit 1
        ;;
esac
if [ ! -f "${VLLM_APPLE_NOTARY_KEY}" ] || [ -L "${VLLM_APPLE_NOTARY_KEY}" ]; then
    echo "error: notarization key must be a regular file, not a symlink." >&2
    exit 1
fi
if [ -e "${output_directory}" ]; then
    echo "error: output path already exists: ${output_directory}" >&2
    exit 1
fi

work_directory=$(mktemp -d "${TMPDIR:-/tmp}/vllm-apple-notarize.XXXXXX")
trap 'rm -rf "${work_directory}"' EXIT HUP INT TERM
expanded_directory="${work_directory}/expanded"
mkdir -p "${expanded_directory}"
/usr/bin/ditto -x -k "${unsigned_zip}" "${expanded_directory}"

app_path="${expanded_directory}/VLLMAppleChat.app"
daemon_path="${app_path}/Contents/MacOS/vllm-appled"
test -d "${app_path}"
test -f "${daemon_path}" && test -x "${daemon_path}" && test ! -L "${daemon_path}"

/usr/bin/codesign --force --sign "${VLLM_APPLE_CODESIGN_IDENTITY}" \
    --options runtime --timestamp "${daemon_path}"
/usr/bin/codesign --force --sign "${VLLM_APPLE_CODESIGN_IDENTITY}" \
    --options runtime --timestamp \
    --entitlements "$(dirname -- "$0")/../samples/VLLMAppleChatXcode/VLLMAppleChat.entitlements" \
    "${app_path}"
/usr/bin/codesign --verify --deep --strict --verbose=2 "${app_path}"

submission_zip="${work_directory}/VLLMAppleChat-notarization-submission.zip"
/usr/bin/ditto -c -k --keepParent "${app_path}" "${submission_zip}"
notary_report="${work_directory}/notary-result.json"
xcrun notarytool submit "${submission_zip}" \
    --key "${VLLM_APPLE_NOTARY_KEY}" \
    --key-id "${VLLM_APPLE_NOTARY_KEY_ID}" \
    --issuer "${VLLM_APPLE_NOTARY_ISSUER}" \
    --wait --output-format json > "${notary_report}"

notary_status=$(/usr/bin/plutil -extract status raw -o - "${notary_report}")
if [ "${notary_status}" != "Accepted" ]; then
    echo "error: Apple notarization status was ${notary_status}." >&2
    exit 1
fi

xcrun stapler staple "${app_path}"
xcrun stapler validate "${app_path}"
/usr/bin/codesign --verify --deep --strict --verbose=2 "${app_path}"
/usr/sbin/spctl --assess --type execute --verbose=2 "${app_path}"

mkdir -p "${output_directory}"
cp "${notary_report}" "${output_directory}/notary-result.json"
/usr/bin/ditto -c -k --keepParent \
    "${app_path}" \
    "${output_directory}/VLLMAppleChat-notarized-arm64.zip"
(
    cd "${output_directory}"
    shasum -a 256 "VLLMAppleChat-notarized-arm64.zip" \
        > "VLLMAppleChat-notarized-arm64.zip.sha256"
)

echo "Created signed and notarized Mac package: ${output_directory}"
