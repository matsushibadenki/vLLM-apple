import copy
import hashlib
import json
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

from vllm_apple.p3_selection import Candidate, SelectionPolicy, Trial, select_candidate
from vllm_apple.p4_certification import (
    IDENTITY_FIELDS,
    RECOVERY_CASES,
    ROLES,
    digest,
    profile_activation,
    verify_certification,
)


class P4CertificationTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.commit, self.artifact = "a" * 40, "b" * 64
        self.identity = {name: hashlib.sha256(name.encode()).hexdigest()
                         for name in IDENTITY_FIELDS}
        self.raw = {
            "quality": dict(passed=True, slices={s: True for s in (
                "en", "ja", "zh", "coding", "tool", "long_context")}),
            "performance": dict(passed=True, independent_acquisition_verified=True,
                                standard_adoption_eligible=True, baseline_retained=False),
            "soak": dict(passed=True, duration_seconds=86400, clean_shutdown=True),
            "recovery": dict(passed=True, cases={c: True for c in RECOVERY_CASES}),
            "rollback": dict(passed=True, restored_profile_sha256="c" * 64,
                             inference_verified=True),
            "installation": dict(passed=True, independent_clean_machine_verified=True),
            "uninstallation": dict(passed=True, independent_clean_machine_verified=True),
            "dependencies": dict(passed=True, lock_verified=True),
            "licenses": dict(passed=True, sbom_license_review_verified=True),
        }
        policy = SelectionPolicy(digest(self.identity), ("en", "ja", "zh", "coding",
                                                        "tool", "long_context"), 1024)

        def candidate(name, kind, time):
            return Candidate(name, kind, tuple(
                Trial(f"{name}-{i}", policy.scope_sha256, policy.digest, time, 1, 1, 100,
                      tuple((s, True) for s in policy.quality_slices)) for i in range(3)))

        self.raw["performance"] = select_candidate(
            policy, candidate("base", "baseline", 10), (candidate("kernel", "kernel", 8),),
            independent_acquisition_verified=True, prerequisites_verified=True)
        self.path = self.root / "p4-certification-v1.json"

    def write(self, name, value):
        raw = json.dumps(value).encode()
        (self.root / name).write_bytes(raw)
        return dict(path=name, sha256=hashlib.sha256(raw).hexdigest())

    def bundle(self):
        refs = {}
        for role in ROLES:
            refs[role] = self.write(role + ".json", dict(
                schema_version=1, role=role, scope_sha256=digest(self.identity),
                source_commit=self.commit, artifact_sha256=self.artifact, passed=True,
                raw_evidence=self.write(role + "-raw.json", self.raw[role])))
        return dict(schema_version=1, source_commit=self.commit,
                    artifact_sha256=self.artifact,
                    cells=[dict(identity=self.identity, evidence=refs)])

    def verify(self, bundle=None):
        self.write(self.path.name, bundle or self.bundle())
        return verify_certification(self.path, source_commit=self.commit,
                                    artifact_sha256=self.artifact)

    def test_complete_synthetic_fixture_and_scope_invalidation(self):
        report = self.verify()
        self.assertTrue(report["passed"])
        self.assertTrue(profile_activation(report, self.identity, thermal_safe=True,
                                           memory_safe=True)["activate"])
        changed = dict(self.identity, backend="d" * 64)
        self.assertFalse(profile_activation(report, changed, thermal_safe=True,
                                            memory_safe=True)["activate"])
        self.assertFalse(profile_activation(report, self.identity, thermal_safe=False,
                                            memory_safe=True)["activate"])
        self.assertFalse(profile_activation(report, self.identity, thermal_safe="unknown",
                                            memory_safe=True)["activate"])
        report["report_id"] = "0" * 64
        self.assertFalse(profile_activation(report, self.identity, thermal_safe=True,
                                            memory_safe=True)["activate"])

    def test_each_failed_raw_role_blocks_release(self):
        for role in ROLES:
            with self.subTest(role=role):
                key = "standard_adoption_eligible" if role == "performance" else "passed"
                self.raw[role][key] = False
                report = self.verify()
                self.assertFalse(report["passed"])
                self.raw[role][key] = True

    def test_24hours_and_every_slice_are_required(self):
        mutations = (
            ("soak", "duration_seconds", 86399),
            ("soak", "clean_shutdown", False),
            ("quality", "slices", {"en": True}),
            ("performance", "standard_adoption_eligible", False),
            ("recovery", "cases", {"startup_failure": True}),
            ("rollback", "inference_verified", False),
            ("installation", "independent_clean_machine_verified", False),
            ("dependencies", "lock_verified", False),
            ("licenses", "sbom_license_review_verified", False),
        )
        for role, key, value in mutations:
            old = self.raw[role][key]
            self.raw[role][key] = value
            self.assertFalse(self.verify()["passed"], (role, key))
            self.raw[role][key] = old

    def test_hash_tamper_and_missing_report_rejected(self):
        bundle = self.bundle()
        (self.root / "soak-raw.json").write_text('{"passed":true}')
        with self.assertRaisesRegex(ValueError, "hash mismatch"):
            self.verify(bundle)
        bundle = self.bundle()
        del bundle["cells"][0]["evidence"]["soak"]
        with self.assertRaises(ValueError):
            self.verify(bundle)

    def test_traversal_symlinks_and_duplicate_cells_rejected(self):
        bundle = self.bundle()
        bundle["cells"][0]["evidence"]["soak"]["path"] = "../outside.json"
        with self.assertRaises(ValueError):
            self.verify(bundle)
        bundle = self.bundle()
        bundle["cells"].append(copy.deepcopy(bundle["cells"][0]))
        with self.assertRaises(ValueError):
            self.verify(bundle)
        bundle = self.bundle()
        (self.root / "soak.json").unlink()
        (self.root / "soak.json").symlink_to(self.root / "quality.json")
        with self.assertRaises(ValueError):
            self.verify(bundle)

    def test_cli_failed_soak_returns_nonzero(self):
        self.raw["soak"]["duration_seconds"] = 1802
        self.verify()
        result = subprocess.run(
            [sys.executable, "-m", "vllm_apple.p4_certification", str(self.path),
             "--source-commit", self.commit, "--artifact-sha256", self.artifact],
            capture_output=True, text=True)
        self.assertEqual(result.returncode, 1)
        self.assertFalse(json.loads(result.stdout)["passed"])

    def test_promotion_blocks_before_github_release_creation(self):
        script = Path("scripts/promote_mac_release.sh").read_text()
        self.assertLess(script.index("python3 -m vllm_apple.p4_certification"),
                        script.index("gh release create"))
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            for name in ("VLLMAppleChat-notarized-arm64.zip",
                         "VLLMAppleChat-notarized-arm64.zip.sha256",
                         "notary-result.json", "release-manifest-v1.json"):
                (root / name).write_text("placeholder")
            import os
            result = subprocess.run(
                ["sh", "scripts/promote_mac_release.sh", str(root), "v0.1.0", "test/repo"],
                env={**os.environ, "GH_TOKEN": "test-token"}, capture_output=True, text=True)
            self.assertNotEqual(result.returncode, 0)
            self.assertIn("p4-certification-v1.json", result.stderr)
