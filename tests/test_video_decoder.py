import json
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

from vllm_apple.video_decoder import FFmpegVideoToolboxDecoder


class VideoDecoderTests(unittest.TestCase):
    def test_inspects_and_decodes_bounded_bgra_frames(self):
        with tempfile.TemporaryDirectory() as directory:
            source = Path(directory) / "video.mp4"
            source.write_bytes(b"fixture")
            metadata = json.dumps({"streams": [{
                "codec_name": "h264", "width": 2, "height": 2,
                "avg_frame_rate": "30/1", "duration": "1.0",
            }]}).encode()
            calls = []

            def run(command, **kwargs):
                calls.append(command)
                output = metadata if "ffprobe" in command[0] else bytes(range(32))
                return SimpleNamespace(stdout=output, stderr=b"")

            with patch("vllm_apple.video_decoder.subprocess.run", side_effect=run):
                result = FFmpegVideoToolboxDecoder().decode(source, maximum_frames=2)
            self.assertEqual(len(result.frames), 2)
            self.assertEqual(result.stream.frame_rate, 30)
            self.assertEqual(result.hardware_accelerator, "videotoolbox")
            self.assertIn("videotoolbox", calls[-1])

    def test_rejects_unsupported_codec_and_output_budget_before_decode(self):
        with tempfile.TemporaryDirectory() as directory:
            source = Path(directory) / "video"
            source.write_bytes(b"fixture")
            unsupported = json.dumps({"streams": [{
                "codec_name": "mpeg2video", "width": 2, "height": 2,
                "avg_frame_rate": "30/1", "duration": "1",
            }]}).encode()
            with patch(
                "vllm_apple.video_decoder.subprocess.run",
                return_value=SimpleNamespace(stdout=unsupported, stderr=b""),
            ):
                with self.assertRaisesRegex(ValueError, "unsupported"):
                    FFmpegVideoToolboxDecoder().decode(source)
            supported = json.dumps({"streams": [{
                "codec_name": "h264", "width": 4, "height": 4,
                "avg_frame_rate": "30/1", "duration": "1",
            }]}).encode()
            with patch(
                "vllm_apple.video_decoder.subprocess.run",
                return_value=SimpleNamespace(stdout=supported, stderr=b""),
            ) as run:
                with self.assertRaisesRegex(ValueError, "output byte budget"):
                    FFmpegVideoToolboxDecoder(maximum_output_bytes=63).decode(
                        source, maximum_frames=1
                    )
                self.assertEqual(run.call_count, 1)

    def test_rejects_symlink_and_invalid_frame_rate(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source = root / "video"
            source.write_bytes(b"fixture")
            link = root / "link"
            link.symlink_to(source)
            with self.assertRaisesRegex(ValueError, "regular local file"):
                FFmpegVideoToolboxDecoder().inspect(link)
            metadata = json.dumps({"streams": [{
                "codec_name": "h264", "width": 2, "height": 2,
                "avg_frame_rate": "0/0", "duration": "1",
            }]}).encode()
            with patch(
                "vllm_apple.video_decoder.subprocess.run",
                return_value=SimpleNamespace(stdout=metadata, stderr=b""),
            ):
                with self.assertRaisesRegex(ValueError, "frame rate"):
                    FFmpegVideoToolboxDecoder().inspect(source)

    def test_mapped_decode_uses_caller_owned_stdout_and_zero_copy_views(self):
        with tempfile.TemporaryDirectory() as directory:
            source = Path(directory) / "video.mp4"
            source.write_bytes(b"fixture")
            metadata = json.dumps({"streams": [{
                "codec_name": "h264", "width": 2, "height": 2,
                "avg_frame_rate": "30/1", "duration": "1.0",
            }]}).encode()

            def run(command, **kwargs):
                if "ffprobe" in command[0]:
                    return SimpleNamespace(stdout=metadata, stderr=b"")
                kwargs["stdout"].write(bytes(range(32)))
                return SimpleNamespace(stdout=None, stderr=b"")

            with patch("vllm_apple.video_decoder.subprocess.run", side_effect=run):
                mapped = FFmpegVideoToolboxDecoder().decode_mapped(
                    source, maximum_frames=2
                )
            first = mapped.frame_view(0)
            self.assertTrue(first.readonly)
            self.assertEqual(bytes(first), bytes(range(16)))
            self.assertEqual(mapped.frame_count, 2)
            self.assertEqual(mapped.storage, "anonymous_mmap")
            first.release()
            mapped.close()
            with self.assertRaisesRegex(RuntimeError, "closed"):
                mapped.frame_view(0)


if __name__ == "__main__":
    unittest.main()
