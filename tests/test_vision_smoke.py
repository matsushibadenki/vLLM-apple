import base64
import json
import io
import struct
import unittest
import zlib
import tempfile
from pathlib import Path
from unittest.mock import patch

from vllm_apple.phase_probe import PhaseProbeConfig, PhaseMeasurement, StreamProbeResult, PhaseProbeError, measure_stream
from vllm_apple.vision_smoke import run_vision_smoke, solid_png


class VisionSmokeTests(unittest.TestCase):
    def test_truncated_stream_cannot_pass_even_with_text_and_usage(self):
        events = [{"choices": [{"delta": {"content": "red"}}]},
                  {"usage": {"prompt_tokens": 100, "completion_tokens": 1}}]
        stream = io.BytesIO(b"".join(b"data: " + json.dumps(e).encode() + b"\n\n" for e in events))
        with patch("vllm_apple.phase_probe.urllib.request.urlopen", return_value=stream):
            with self.assertRaises(PhaseProbeError) as raised:
                measure_stream(PhaseProbeConfig(base_url="http://127.0.0.1:8000", model="vision", hardware_fingerprint="test"), expected_text="red")
        self.assertEqual(raised.exception.code, "incomplete_stream")

    def test_sse_backend_error_is_not_misreported_as_empty_answer(self):
        stream = io.BytesIO(b'data: {"error":{"message":"private backend details"}}\n\ndata: [DONE]\n\n')
        with patch("vllm_apple.phase_probe.urllib.request.urlopen", return_value=stream):
            with self.assertRaises(PhaseProbeError) as raised:
                measure_stream(PhaseProbeConfig(base_url="http://127.0.0.1:8000", model="vision", hardware_fingerprint="test"))
        self.assertEqual(raised.exception.code, "backend_stream_error")
        self.assertNotIn("private", str(raised.exception))

    def test_cli_saves_failed_smoke_without_certifying_model(self):
        from vllm_apple.cli import main
        report = {"passed": False, "qualification": False, "model": "vision-test"}
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory) / "vision.json"
            with patch("vllm_apple.cli.run_vision_smoke", return_value=report), patch("sys.stdout", new_callable=io.StringIO):
                code = main(["vision-smoke", "--model", "vision-test", "--output", str(output)])
            self.assertEqual(code, 1)
            self.assertEqual(json.loads(output.read_text()), report)

    def test_png_is_sent_as_openai_image_content(self):
        image = solid_png((255, 0, 0))
        events = [
            {"choices": [{"delta": {"content": "red\n"}}]},
            {"usage": {"prompt_tokens": 100, "completion_tokens": 2}},
        ]
        stream = b"".join(b"data: " + json.dumps(e).encode() + b"\n\n" for e in events) + b"data: [DONE]\n\n"
        with patch("vllm_apple.phase_probe.urllib.request.urlopen", return_value=io.BytesIO(stream)) as opened:
            result = measure_stream(PhaseProbeConfig(
                base_url="http://127.0.0.1:8000", model="vision", hardware_fingerprint="test",
            ), image_png=image, expected_text="red", expected_match_mode="trimmed_exact")
        content = json.loads(opened.call_args.args[0].data)["messages"][0]["content"]
        self.assertEqual(content[0]["type"], "text")
        self.assertEqual(content[1]["type"], "image_url")
        self.assertEqual(base64.b64decode(content[1]["image_url"]["url"].split(",")[1]), image)
        self.assertTrue(result.expected_text_matched)

    def test_fixtures_decode_to_expected_rgb(self):
        for rgb in ((255, 0, 0), (0, 0, 255)):
            png = solid_png(rgb)
            self.assertTrue(png.startswith(b"\x89PNG\r\n\x1a\n"))
            offset = 8
            pixels = b""
            while offset < len(png):
                size = struct.unpack(">I", png[offset:offset + 4])[0]
                kind = png[offset + 4:offset + 8]
                data = png[offset + 8:offset + 8 + size]
                self.assertEqual(struct.unpack(">I", png[offset + 8 + size:offset + 12 + size])[0], zlib.crc32(kind + data))
                if kind == b"IDAT":
                    pixels += data
                offset += size + 12
            self.assertEqual(zlib.decompress(pixels), (b"\x00" + bytes(rgb) * 32) * 32)

    def test_paired_images_use_identical_prompts_and_fail_closed(self):
        calls = []
        def measure(config, **kwargs):
            calls.append((config.prompt, kwargs))
            return StreamProbeResult(PhaseMeasurement(1, 2, 3, 4, 1, 0), len(calls) != 2, 0)
        report = run_vision_smoke(PhaseProbeConfig(
            base_url="http://127.0.0.1:8000", model="vision", hardware_fingerprint="test",
        ), measure=measure)
        self.assertFalse(report["passed"])
        self.assertFalse(report["qualification"])
        self.assertEqual(report["sample_count"], 6)
        for i in (0, 2, 4):
            self.assertEqual(calls[i][0], calls[i + 1][0])
            self.assertNotEqual(calls[i][1]["image_png"], calls[i + 1][1]["image_png"])
            self.assertNotEqual(calls[i][1]["expected_text"], calls[i + 1][1]["expected_text"])
        self.assertNotIn("image_png", json.dumps(report))
