import json
import threading
import unittest
import urllib.request
from pathlib import Path

from tests.schema_validator import (
    SchemaValidationError,
    ensure_supported_schema,
    validate_instance,
)
from vllm_apple.api import create_server
from vllm_apple.service import RuntimeService


SCHEMA_ROOT = Path("schemas")


def load_schema(relative_path: str) -> dict:
    return json.loads((SCHEMA_ROOT / relative_path).read_text(encoding="utf-8"))


class SchemaTests(unittest.TestCase):
    def test_committed_schemas_are_valid_json_versioned_and_supported(self) -> None:
        schemas = sorted(SCHEMA_ROOT.rglob("*.schema.json"))
        self.assertTrue(schemas)
        for path in schemas:
            payload = json.loads(path.read_text(encoding="utf-8"))
            self.assertEqual(payload["$schema"], "https://json-schema.org/draft/2020-12/schema")
            self.assertIn("v1", payload["$id"])
            ensure_supported_schema(payload)

    def test_validator_rejects_missing_wrong_and_additional_values(self) -> None:
        schema = load_schema("api/health-v1.schema.json")
        valid = {
            "api_version": "v1",
            "schema_version": 1,
            "runtime_version": "0.1.0",
            "minimum_client_version": "0.1.0",
            "status": "ready",
            "control_ready": True,
            "inference_ready": True,
        }
        validate_instance(valid, schema)
        for invalid in (
            {key: value for key, value in valid.items() if key != "status"},
            {**valid, "schema_version": 2},
            {**valid, "schema_version": True},
            {**valid, "control_ready": 1},
            {**valid, "unexpected": True},
        ):
            with self.assertRaises(SchemaValidationError):
                validate_instance(invalid, schema)


class LiveResponseSchemaTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.service = RuntimeService()
        cls.server = create_server("127.0.0.1", 0, cls.service)
        cls.thread = threading.Thread(target=cls.server.serve_forever, daemon=True)
        cls.thread.start()
        cls.base_url = f"http://127.0.0.1:{cls.server.server_port}"

    @classmethod
    def tearDownClass(cls) -> None:
        cls.service.events.publish("test.release", {})
        cls.server.shutdown()
        cls.server.server_close()
        cls.thread.join(timeout=2)

    def get_json(self, path: str) -> dict:
        with urllib.request.urlopen(self.base_url + path, timeout=2) as response:
            return json.load(response)

    def test_live_health_response_matches_v1_schema(self) -> None:
        validate_instance(
            self.get_json("/health"),
            load_schema("api/health-v1.schema.json"),
        )

    def test_live_runtime_response_matches_v1_schema_and_memory_invariant(self) -> None:
        payload = self.get_json("/v1/runtime")
        validate_instance(payload, load_schema("api/runtime-v1.schema.json"))
        scheduler = payload["scheduler"]
        self.assertEqual(
            scheduler["reserved_bytes"] + scheduler["available_bytes"],
            scheduler["capacity_bytes"],
        )

    def test_live_sse_event_matches_v1_schema(self) -> None:
        response = urllib.request.urlopen(self.base_url + "/v1/events", timeout=2)
        try:
            identifier = response.readline().decode("utf-8")
            event_type = response.readline().decode("utf-8")
            data = response.readline().decode("utf-8")
            self.assertTrue(identifier.startswith("id: "))
            self.assertTrue(event_type.startswith("event: "))
            payload = json.loads(data.removeprefix("data: "))
            validate_instance(
                payload,
                load_schema("events/runtime-event-v1.schema.json"),
            )
        finally:
            response.close()
            self.service.events.publish("test.release", {})


if __name__ == "__main__":
    unittest.main()
