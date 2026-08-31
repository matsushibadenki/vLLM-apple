#!/bin/sh
set -eu

repository_root=$(CDPATH= cd -- "$(dirname -- "$0")/.." && pwd)
output_directory=${1:-"${repository_root}/dist/mac-integration"}

if [ "$(uname -s)" != "Darwin" ]; then
    echo "error: macOS is required to build the Mac integration package." >&2
    exit 1
fi
if [ "$(uname -m)" != "arm64" ]; then
    echo "error: an Apple Silicon (arm64) host is required." >&2
    exit 1
fi
if [ -e "${output_directory}" ]; then
    echo "error: output path already exists: ${output_directory}" >&2
    exit 1
fi

work_directory=$(mktemp -d "${TMPDIR:-/tmp}/vllm-apple-package.XXXXXX")
trap 'rm -rf "${work_directory}"' EXIT HUP INT TERM

daemon_dist="${work_directory}/daemon-dist"
derived_data="${work_directory}/DerivedData"
bundle_root="${work_directory}/bundle"
mkdir -p "${daemon_dist}" "${bundle_root}"

PYINSTALLER_CONFIG_DIR="${work_directory}/pyinstaller-config" python3 -m PyInstaller \
    --clean \
    --noconfirm \
    --onefile \
    --name vllm-appled \
    --distpath "${daemon_dist}" \
    --workpath "${work_directory}/pyinstaller-work" \
    --specpath "${work_directory}" \
    "${repository_root}/packaging/vllm_appled_entry.py"

daemon_path="${daemon_dist}/vllm-appled"
test -f "${daemon_path}" && test -x "${daemon_path}" && test ! -L "${daemon_path}"

VLLM_APPLE_DAEMON_SOURCE="${daemon_path}" xcodebuild \
    -project "${repository_root}/samples/VLLMAppleChatXcode/VLLMAppleChat.xcodeproj" \
    -scheme VLLMAppleChat \
    -configuration Release \
    -derivedDataPath "${derived_data}" \
    CODE_SIGNING_ALLOWED=NO \
    build

app_source="${derived_data}/Build/Products/Release/VLLMAppleChat.app"
embedded_daemon="${app_source}/Contents/MacOS/vllm-appled"
test -d "${app_source}"
test -f "${embedded_daemon}" && test -x "${embedded_daemon}" && test ! -L "${embedded_daemon}"

cp -R "${app_source}" "${bundle_root}/VLLMAppleChat.app"
mkdir -p "${output_directory}"
/usr/bin/ditto -c -k --keepParent \
    "${bundle_root}/VLLMAppleChat.app" \
    "${output_directory}/VLLMAppleChat-unsigned-arm64.zip"

(
    cd "${output_directory}"
    shasum -a 256 "VLLMAppleChat-unsigned-arm64.zip" \
        > "VLLMAppleChat-unsigned-arm64.zip.sha256"
)

echo "Created unsigned Mac integration package: ${output_directory}"
