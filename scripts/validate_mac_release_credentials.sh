#!/bin/sh
set -eu

required_variables="
CERTIFICATE_P12_BASE64
CERTIFICATE_PASSWORD
CODESIGN_IDENTITY
NOTARY_KEY_BASE64
NOTARY_KEY_ID
NOTARY_ISSUER
KEYCHAIN_PASSWORD
"

for variable_name in ${required_variables}; do
    eval "variable_value=\${${variable_name}:-}"
    if [ -z "${variable_value}" ]; then
        echo "error: required release credential is missing: ${variable_name}" >&2
        exit 1
    fi
done

case "${CODESIGN_IDENTITY}" in
    "Developer ID Application: "*) ;;
    *)
        echo "error: CODESIGN_IDENTITY must be a Developer ID Application identity." >&2
        exit 1
        ;;
esac
if ! printf '%s\n' "${NOTARY_KEY_ID}" | grep -Eq '^[A-Z0-9]{10}$'; then
    echo "error: NOTARY_KEY_ID must contain 10 uppercase letters or digits." >&2
    exit 1
fi
if ! printf '%s\n' "${NOTARY_ISSUER}" | grep -Eiq \
    '^[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}$'; then
    echo "error: NOTARY_ISSUER must be a UUID." >&2
    exit 1
fi

credential_directory=$(mktemp -d "${RUNNER_TEMP:-${TMPDIR:-/tmp}}/vllm-apple-credentials.XXXXXX")
certificate_path="${credential_directory}/developer-id.p12"
notary_key_path="${credential_directory}/AuthKey.p8"
cleanup() {
    rm -f "${certificate_path}" "${notary_key_path}"
    rmdir "${credential_directory}" 2>/dev/null || true
}
trap cleanup EXIT HUP INT TERM
umask 077

if ! printf '%s' "${CERTIFICATE_P12_BASE64}" \
    | openssl base64 -d -A -out "${certificate_path}" 2>/dev/null; then
    echo "error: certificate is not valid Base64." >&2
    exit 1
fi
if [ ! -s "${certificate_path}" ]; then
    echo "error: decoded certificate is empty." >&2
    exit 1
fi
if ! CERTIFICATE_PASSWORD="${CERTIFICATE_PASSWORD}" openssl pkcs12 \
    -in "${certificate_path}" \
    -passin env:CERTIFICATE_PASSWORD \
    -noout >/dev/null 2>&1; then
    echo "error: certificate PKCS#12 data or password is invalid." >&2
    exit 1
fi

if ! printf '%s' "${NOTARY_KEY_BASE64}" \
    | openssl base64 -d -A -out "${notary_key_path}" 2>/dev/null; then
    echo "error: notarization key is not valid Base64." >&2
    exit 1
fi
if [ ! -s "${notary_key_path}" ]; then
    echo "error: decoded notarization key is empty." >&2
    exit 1
fi
if ! openssl pkey -in "${notary_key_path}" -check -noout >/dev/null 2>&1; then
    echo "error: notarization key is not a valid private key." >&2
    exit 1
fi

echo "Mac release credentials passed structural validation."
