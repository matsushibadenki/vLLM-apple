#!/bin/sh
set -eu

if [ "$#" -ne 3 ]; then
    echo "usage: $0 RELEASE_BUNDLE RELEASE_TAG GITHUB_REPOSITORY" >&2
    exit 64
fi

bundle_directory=$1
release_tag=$2
repository=$3
: "${GH_TOKEN:?GH_TOKEN is required}"

case "${bundle_directory}" in
    /*) ;;
    *)
        echo "error: release bundle path must be absolute." >&2
        exit 1
        ;;
esac
if [ ! -d "${bundle_directory}" ] || [ -L "${bundle_directory}" ]; then
    echo "error: release bundle must be a directory, not a symlink." >&2
    exit 1
fi
if ! printf '%s\n' "${release_tag}" | grep -Eq '^v[0-9]+\.[0-9]+\.[0-9]+$'; then
    echo "error: release tag must be an exact vMAJOR.MINOR.PATCH version." >&2
    exit 1
fi
if ! printf '%s\n' "${repository}" | grep -Eq '^[A-Za-z0-9_.-]+/[A-Za-z0-9_.-]+$'; then
    echo "error: GitHub repository must use owner/name form." >&2
    exit 1
fi

archive="${bundle_directory}/VLLMAppleChat-notarized-arm64.zip"
checksum="${bundle_directory}/VLLMAppleChat-notarized-arm64.zip.sha256"
notary_report="${bundle_directory}/notary-result.json"
manifest="${bundle_directory}/release-manifest-v1.json"
for evidence in "${archive}" "${checksum}" "${notary_report}" "${manifest}"; do
    if [ ! -f "${evidence}" ] || [ -L "${evidence}" ]; then
        echo "error: release evidence must be a regular file, not a symlink: ${evidence}" >&2
        exit 1
    fi
done

(
    cd "${bundle_directory}"
    shasum -a 256 -c "$(basename -- "${checksum}")"
)
vllm-apple-release-manifest verify "${archive}" \
    --notary-report "${notary_report}" \
    --manifest "${manifest}"
gh attestation verify "${archive}" --repo "${repository}"

manifest_commit=$(python3 -c \
    'import json,sys; print(json.load(open(sys.argv[1], encoding="utf-8"))["source"]["commit"])' \
    "${manifest}")
tag_commit=$(git rev-parse "${release_tag}^{commit}")
if [ "${manifest_commit}" != "${tag_commit}" ]; then
    echo "error: release tag does not point to the manifest source commit." >&2
    exit 1
fi

gh release create "${release_tag}" \
    "${archive}" \
    "${checksum}" \
    "${notary_report}" \
    "${manifest}" \
    --repo "${repository}" \
    --draft \
    --verify-tag \
    --generate-notes \
    --title "vLLM-Apple ${release_tag}"

echo "Created verified draft release ${release_tag}."
