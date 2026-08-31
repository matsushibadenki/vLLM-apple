import json
import plistlib
import stat
import tempfile
import unittest
import zipfile
from pathlib import Path

from vllm_apple.release_manifest import (
    INFO_PLIST,
    EXPECTED_COMPONENTS,
    ReleaseManifestError,
    build_release_manifest,
    save_release_manifest,
    verify_release_manifest,
)
from schema_validator import validate_instance


class ReleaseManifestTests(unittest.TestCase):
    def _fixture(self, root: Path) -> tuple[Path, Path]:
        archive = root / "VLLMAppleChat-notarized-arm64.zip"
        info = {
            "CFBundleIdentifier": "dev.vllm-apple.chat",
            "CFBundleVersion": "1",
            "CFBundleShortVersionString": "0.1.0",
            "LSMinimumSystemVersion": "13.0",
        }
        with zipfile.ZipFile(archive, "w") as output:
            for index, name in enumerate(EXPECTED_COMPONENTS):
                member = zipfile.ZipInfo(name)
                member.external_attr = (stat.S_IFREG | 0o755) << 16
                output.writestr(member, f"executable-{index}".encode())
            output.writestr(INFO_PLIST, plistlib.dumps(info))
        report = root / "notary-result.json"
        report.write_text(json.dumps({"id": "submission-1", "status": "Accepted"}))
        return archive, report

    def test_build_and_verify_binds_archive_components_and_provenance(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            archive, report = self._fixture(root)
            manifest = build_release_manifest(
                archive,
                report,
                source_commit="a" * 40,
                build_run_id="12345",
                runner_image="macos-15",
            )
            output = root / "release-manifest.json"
            save_release_manifest(manifest, output)
            verified = verify_release_manifest(archive, report, output)
            self.assertEqual(verified["notarization"]["status"], "Accepted")
            self.assertEqual(len(verified["artifact"]["components"]), 2)
            schema = json.loads(
                Path("schemas/runtime/mac-release-manifest-v1.schema.json").read_text()
            )
            validate_instance(verified, schema)

            with archive.open("ab") as stream:
                stream.write(b"tampered")
            with self.assertRaisesRegex(ReleaseManifestError, "does not match"):
                verify_release_manifest(archive, report, output)

    def test_rejects_unaccepted_notary_report_and_unsafe_zip_member(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            archive, report = self._fixture(root)
            report.write_text(json.dumps({"id": "submission-1", "status": "Invalid"}))
            with self.assertRaisesRegex(ReleaseManifestError, "Accepted"):
                build_release_manifest(
                    archive,
                    report,
                    source_commit="b" * 40,
                    build_run_id="1",
                    runner_image="macos-15",
                )

            unsafe = root / "unsafe.zip"
            with zipfile.ZipFile(unsafe, "w") as output:
                output.writestr("../escape", b"bad")
            report.write_text(json.dumps({"id": "submission-1", "status": "Accepted"}))
            with self.assertRaisesRegex(ReleaseManifestError, "unsafe path"):
                build_release_manifest(
                    unsafe,
                    report,
                    source_commit="b" * 40,
                    build_run_id="1",
                    runner_image="macos-15",
                )


if __name__ == "__main__":
    unittest.main()
