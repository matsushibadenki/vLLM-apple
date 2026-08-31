import os
import base64
import stat
import subprocess
import tempfile
import unittest
from pathlib import Path


SAMPLE = Path("samples/VLLMAppleChatXcode")
PACKAGE_SCRIPT = Path("scripts/build_mac_integration_package.sh")
PACKAGE_WORKFLOW = Path(".github/workflows/mac-integration-package.yml")
NOTARIZE_SCRIPT = Path("scripts/sign_and_notarize_mac_package.sh")
NOTARIZE_WORKFLOW = Path(".github/workflows/mac-notarized-release.yml")
PROMOTE_SCRIPT = Path("scripts/promote_mac_release.sh")
PROMOTE_WORKFLOW = Path(".github/workflows/mac-draft-release.yml")
CREDENTIAL_SCRIPT = Path("scripts/validate_mac_release_credentials.sh")


class XcodeSampleTests(unittest.TestCase):
    def test_release_credential_preflight_parses_real_private_material(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            key = root / "key.pem"
            certificate = root / "certificate.pem"
            archive = root / "certificate.p12"
            subprocess.run(
                [
                    "openssl",
                    "req",
                    "-x509",
                    "-newkey",
                    "rsa:2048",
                    "-nodes",
                    "-subj",
                    "/CN=vLLM Apple Test",
                    "-keyout",
                    str(key),
                    "-out",
                    str(certificate),
                    "-days",
                    "1",
                ],
                check=True,
                capture_output=True,
            )
            subprocess.run(
                [
                    "openssl",
                    "pkcs12",
                    "-export",
                    "-out",
                    str(archive),
                    "-inkey",
                    str(key),
                    "-in",
                    str(certificate),
                    "-passout",
                    "pass:test-password",
                ],
                check=True,
                capture_output=True,
            )
            environment = {
                **os.environ,
                "CERTIFICATE_P12_BASE64": base64.b64encode(archive.read_bytes()).decode(),
                "CERTIFICATE_PASSWORD": "test-password",
                "CODESIGN_IDENTITY": "Developer ID Application: Test (ABCDEFGHIJ)",
                "NOTARY_KEY_BASE64": base64.b64encode(key.read_bytes()).decode(),
                "NOTARY_KEY_ID": "ABCDEFGHIJ",
                "NOTARY_ISSUER": "01234567-89ab-cdef-0123-456789abcdef",
                "KEYCHAIN_PASSWORD": "ephemeral-test-password",
                "RUNNER_TEMP": str(root),
            }
            completed = subprocess.run(
                [str(CREDENTIAL_SCRIPT)],
                env=environment,
                check=False,
                capture_output=True,
                text=True,
            )
            self.assertEqual(completed.returncode, 0, completed.stderr)
            self.assertIn("passed structural validation", completed.stdout)

            environment["CERTIFICATE_PASSWORD"] = "secret-that-must-not-be-printed"
            rejected = subprocess.run(
                [str(CREDENTIAL_SCRIPT)],
                env=environment,
                check=False,
                capture_output=True,
                text=True,
            )
            self.assertNotEqual(rejected.returncode, 0)
            self.assertIn("PKCS#12 data or password is invalid", rejected.stderr)
            self.assertNotIn(environment["CERTIFICATE_PASSWORD"], rejected.stderr)

    def test_release_candidate_build_has_reproducible_safety_contract(self) -> None:
        requirements = Path("requirements-package.lock").read_text(encoding="utf-8")
        self.assertEqual(requirements, "pyinstaller==6.16.0\n")

        script = PACKAGE_SCRIPT.read_text(encoding="utf-8")
        self.assertIn("--onefile", script)
        self.assertIn("PYINSTALLER_CONFIG_DIR=", script)
        self.assertIn("CODE_SIGNING_ALLOWED=NO", script)
        self.assertIn("VLLM_APPLE_DAEMON_SOURCE=", script)
        self.assertIn("test ! -L", script)
        self.assertIn("shasum -a 256", script)
        self.assertIn("VLLMAppleChat-unsigned-arm64.zip", script)

        workflow = PACKAGE_WORKFLOW.read_text(encoding="utf-8")
        install = workflow.index("Install project and locked packaging tools")
        build = workflow.index("Build app with standalone daemon")
        verify = workflow.index("Verify package checksum")
        upload = workflow.index("actions/upload-artifact@v6")
        self.assertLess(install, build)
        self.assertLess(build, verify)
        self.assertLess(verify, upload)

    def test_notarized_release_signs_inside_out_and_requires_acceptance(self) -> None:
        script = NOTARIZE_SCRIPT.read_text(encoding="utf-8")
        daemon_sign = script.index('--timestamp "${daemon_path}"')
        app_sign = script.index('"${app_path}"\n/usr/bin/codesign --verify')
        submit = script.index("notarytool submit")
        accepted = script.index('!= "Accepted"')
        staple = script.index("stapler staple")
        assess = script.index("spctl --assess")
        self.assertLess(daemon_sign, app_sign)
        self.assertLess(app_sign, submit)
        self.assertLess(submit, accepted)
        self.assertLess(accepted, staple)
        self.assertLess(staple, assess)
        self.assertIn("--options runtime", script)
        self.assertIn("--timestamp", script)

        workflow = NOTARIZE_WORKFLOW.read_text(encoding="utf-8")
        self.assertIn("workflow_dispatch:", workflow)
        self.assertIn("environment: mac-release", workflow)
        self.assertIn("Validate release secrets", workflow)
        self.assertIn("scripts/validate_mac_release_credentials.sh", workflow)
        self.assertIn("APPLE_BUILD_KEYCHAIN_PASSWORD", workflow)
        self.assertNotIn("base64 --decode", workflow)
        self.assertIn("Remove ephemeral signing material", workflow)
        self.assertIn("if: always()", workflow)
        manifest = workflow.index("Create and verify bounded release manifest")
        attestation = workflow.index("actions/attest-build-provenance@v3")
        upload = workflow.index("actions/upload-artifact@v6")
        self.assertLess(manifest, attestation)
        self.assertLess(attestation, upload)
        self.assertIn("id-token: write", workflow)
        self.assertIn("attestations: write", workflow)

    def test_draft_release_reverifies_all_evidence_before_publishing(self) -> None:
        script = PROMOTE_SCRIPT.read_text(encoding="utf-8")
        checksum = script.index("shasum -a 256 -c")
        manifest = script.index("vllm-apple-release-manifest verify")
        attestation = script.index("gh attestation verify")
        commit = script.index('git rev-parse "${release_tag}^{commit}"')
        publish = script.index("gh release create")
        self.assertLess(checksum, manifest)
        self.assertLess(manifest, attestation)
        self.assertLess(attestation, commit)
        self.assertLess(commit, publish)
        self.assertIn("--draft", script)
        self.assertIn("--verify-tag", script)

        workflow = PROMOTE_WORKFLOW.read_text(encoding="utf-8")
        self.assertIn("environment: mac-release", workflow)
        self.assertIn("actions: read", workflow)
        self.assertIn("attestations: read", workflow)
        self.assertIn("contents: write", workflow)
        self.assertIn("fetch-depth: 0", workflow)
        validate = workflow.index("Validate bounded promotion inputs")
        download = workflow.index("Download notarized release evidence")
        promote = workflow.index("Verify evidence and create draft release")
        self.assertLess(validate, download)
        self.assertLess(download, promote)

    def test_project_declares_app_package_localizations_and_embed_phase(self) -> None:
        project = (SAMPLE / "project.yml").read_text(encoding="utf-8")
        self.assertIn("type: application", project)
        self.assertIn("path: ../../sdk/swift", project)
        self.assertIn("ENABLE_HARDENED_RUNTIME: YES", project)
        self.assertIn("Scripts/embed-daemon.sh", project)
        package = Path("samples/VLLMAppleChat/Package.swift").read_text(encoding="utf-8")
        self.assertIn('defaultLocalization: "en"', package)
        for language in ("en.lproj", "ja.lproj", "zh-Hans.lproj"):
            self.assertTrue(
                (Path("samples/VLLMAppleChat/Sources/VLLMAppleChat/Resources") / language).is_dir()
            )

    def test_embed_phase_places_executable_in_app_auxiliary_directory(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            daemon = root / "source-daemon"
            daemon.write_text("#!/bin/sh\nexit 0\n", encoding="utf-8")
            daemon.chmod(0o700)
            environment = {
                **os.environ,
                "VLLM_APPLE_DAEMON_SOURCE": str(daemon),
                "TARGET_BUILD_DIR": str(root / "Build"),
                "EXECUTABLE_FOLDER_PATH": "VLLMAppleChat.app/Contents/MacOS",
                "CODE_SIGNING_ALLOWED": "NO",
            }
            subprocess.run(
                [str(SAMPLE / "Scripts/embed-daemon.sh")],
                env=environment,
                check=True,
                capture_output=True,
                text=True,
            )
            embedded = (
                root / "Build/VLLMAppleChat.app/Contents/MacOS/vllm-appled"
            )
            self.assertTrue(embedded.is_file())
            self.assertFalse(embedded.is_symlink())
            self.assertTrue(embedded.stat().st_mode & stat.S_IXUSR)

    def test_embed_phase_rejects_symlink_source(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            daemon = root / "daemon"
            daemon.write_text("#!/bin/sh\n", encoding="utf-8")
            daemon.chmod(0o700)
            link = root / "link"
            link.symlink_to(daemon)
            completed = subprocess.run(
                [str(SAMPLE / "Scripts/embed-daemon.sh")],
                env={
                    **os.environ,
                    "VLLM_APPLE_DAEMON_SOURCE": str(link),
                    "TARGET_BUILD_DIR": str(root / "Build"),
                    "EXECUTABLE_FOLDER_PATH": "App.app/Contents/MacOS",
                    "CODE_SIGNING_ALLOWED": "NO",
                },
                check=False,
                capture_output=True,
                text=True,
            )
            self.assertNotEqual(completed.returncode, 0)
            self.assertIn("not a symlink", completed.stderr)


if __name__ == "__main__":
    unittest.main()
