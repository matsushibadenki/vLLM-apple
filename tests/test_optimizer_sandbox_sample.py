from __future__ import annotations

import plistlib
import re
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SAMPLE = ROOT / "samples" / "VLLMAppleOptimizerSandbox"


class OptimizerSandboxSampleTests(unittest.TestCase):
    def test_entitlements_are_minimal_and_process_execution_is_absent(self) -> None:
        entitlements = plistlib.loads(
            (SAMPLE / "VLLMAppleOptimizerSandbox.entitlements").read_bytes()
        )
        self.assertEqual(set(entitlements), {
            "com.apple.security.app-sandbox",
            "com.apple.security.network.client",
            "com.apple.security.files.user-selected.read-only",
            "com.apple.security.files.bookmarks.app-scope",
        })
        self.assertTrue(all(entitlements.values()))
        source = "\n".join(
            path.read_text(encoding="utf-8")
            for path in (SAMPLE / "Sources").glob("*.swift")
        )
        for forbidden in ("Process(", "executableURL", "posix_spawn", "NSTask"):
            self.assertNotIn(forbidden, source)
        project = (SAMPLE / "project.yml").read_text(encoding="utf-8")
        self.assertIn("ENABLE_APP_SANDBOX: YES", project)
        self.assertNotIn("BoundedProcessRunner.swift", project)

    def test_all_locales_have_identical_nonempty_keys(self) -> None:
        key_sets = {}
        for locale in ("en", "ja", "zh-Hans"):
            text = (
                SAMPLE / "Resources" / f"{locale}.lproj" / "Localizable.strings"
            ).read_text(encoding="utf-8")
            entries = re.findall(r'^"([^"]+)"\s*=\s*"([^"]+)";$', text, re.MULTILINE)
            self.assertTrue(entries)
            self.assertTrue(all(value.strip() for _, value in entries))
            key_sets[locale] = {key for key, _ in entries}
        self.assertEqual(key_sets["en"], key_sets["ja"])
        self.assertEqual(key_sets["en"], key_sets["zh-Hans"])


if __name__ == "__main__":
    unittest.main()
