import json
import unittest
from pathlib import Path


class SchemaTests(unittest.TestCase):
    def test_committed_schemas_are_valid_json_and_versioned(self) -> None:
        schemas = sorted(Path("schemas").rglob("*.schema.json"))
        self.assertTrue(schemas)
        for path in schemas:
            payload = json.loads(path.read_text(encoding="utf-8"))
            self.assertEqual(payload["$schema"], "https://json-schema.org/draft/2020-12/schema")
            self.assertIn("v1", payload["$id"])


if __name__ == "__main__":
    unittest.main()
