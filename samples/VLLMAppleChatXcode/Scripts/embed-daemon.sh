#!/bin/sh
set -eu

if [ -z "${VLLM_APPLE_DAEMON_SOURCE:-}" ]; then
    echo "error: Set VLLM_APPLE_DAEMON_SOURCE to a signed, standalone vllm-appled executable." >&2
    exit 1
fi

source_path="${VLLM_APPLE_DAEMON_SOURCE}"
if [ -L "${source_path}" ] || [ ! -f "${source_path}" ] || [ ! -x "${source_path}" ]; then
    echo "error: VLLM_APPLE_DAEMON_SOURCE must be a regular executable, not a symlink." >&2
    exit 1
fi

case "${source_path}" in
    /*) ;;
    *)
        echo "error: VLLM_APPLE_DAEMON_SOURCE must be an absolute path." >&2
        exit 1
        ;;
esac

: "${TARGET_BUILD_DIR:?TARGET_BUILD_DIR is required}"
: "${EXECUTABLE_FOLDER_PATH:?EXECUTABLE_FOLDER_PATH is required}"

destination_directory="${TARGET_BUILD_DIR}/${EXECUTABLE_FOLDER_PATH}"
destination="${destination_directory}/vllm-appled"
mkdir -p "${destination_directory}"
/usr/bin/install -m 0755 "${source_path}" "${destination}"

if [ -n "${EXPANDED_CODE_SIGN_IDENTITY:-}" ] && [ "${CODE_SIGNING_ALLOWED:-NO}" = "YES" ]; then
    /usr/bin/codesign --force --sign "${EXPANDED_CODE_SIGN_IDENTITY}" \
        --options runtime --timestamp=none "${destination}"
fi

test -f "${destination}" && test -x "${destination}" && test ! -L "${destination}"
