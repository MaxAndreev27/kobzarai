import json
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import requests
import run_ollama


class Response:
    def __init__(self, value):
        self.value = value

    def raise_for_status(self):
        pass

    def json(self):
        return self.value


class RunnerTests(unittest.TestCase):
    def run_with(self, output, post_result, device="cpu", placement=None):
        version = Response({"version": "test"})
        tags = Response({"models": [{"name": "qwen3:1.7b", "digest": "test"}]})
        gets = [version, tags]
        if placement is not None:
            gets.append(Response({"models": [placement]}))
        with (
            patch.object(
                sys,
                "argv",
                [
                    "run_ollama.py",
                    "--limit",
                    "1",
                    "--output",
                    str(output),
                    "--device",
                    device,
                ],
            ),
            patch.object(run_ollama.requests, "get", side_effect=gets),
            patch.object(run_ollama.requests, "post", side_effect=post_result) as post,
        ):
            status = run_ollama.main()
        return status, post

    def test_records_development_case_and_cpu_request(self):
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory) / "results.jsonl"
            response = Response(
                {"done": True, "done_reason": "stop", "message": {"content": "Вітаю!"}}
            )
            status, post = self.run_with(output, [response])
            metadata, result = [
                json.loads(line)
                for line in output.read_text(encoding="utf-8").splitlines()
            ]
            self.assertEqual(status, 0)
            self.assertEqual(metadata["split"], "development")
            self.assertEqual(metadata["case_count"], 1)
            self.assertEqual(result["case"]["id"], "dialogue-01")
            self.assertEqual(result["response"]["message"]["content"], "Вітаю!")
            self.assertEqual(post.call_args.kwargs["json"]["options"]["num_gpu"], 0)

    def test_records_request_failure(self):
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory) / "results.jsonl"
            status, _ = self.run_with(output, requests.ConnectionError("unavailable"))
            records = [
                json.loads(line)
                for line in output.read_text(encoding="utf-8").splitlines()
            ]
            self.assertEqual(status, 1)
            self.assertEqual(records[1]["type"], "error")
            self.assertIn("unavailable", records[1]["error"])

    def test_rejects_empty_response(self):
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory) / "results.jsonl"
            status, _ = self.run_with(
                output, [Response({"done": True, "message": {"content": ""}})]
            )
            records = [
                json.loads(line)
                for line in output.read_text(encoding="utf-8").splitlines()
            ]
            self.assertEqual(status, 1)
            self.assertEqual(records[1]["type"], "error")
            self.assertIn("empty", records[1]["error"])

    def test_gpu_uses_automatic_placement_and_records_it(self):
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory) / "results.jsonl"
            response = Response(
                {"done": True, "done_reason": "stop", "message": {"content": "Вітаю!"}}
            )
            status, post = self.run_with(
                output, [response], "gpu", {"digest": "test", "size_vram": 1024}
            )
            records = [
                json.loads(line)
                for line in output.read_text(encoding="utf-8").splitlines()
            ]
            self.assertEqual(status, 0)
            self.assertNotIn("num_gpu", post.call_args.kwargs["json"]["options"])
            self.assertEqual(records[1]["placement"]["size_vram"], 1024)

    def test_gpu_cpu_fallback_is_an_error(self):
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory) / "results.jsonl"
            response = Response({"done": True, "message": {"content": "Вітаю!"}})
            status, _ = self.run_with(
                output, [response], "gpu", {"digest": "test", "size_vram": 0}
            )
            records = [
                json.loads(line)
                for line in output.read_text(encoding="utf-8").splitlines()
            ]
            self.assertEqual(status, 1)
            self.assertEqual(records[1]["type"], "error")
            self.assertIn("GPU memory", records[1]["error"])


if __name__ == "__main__":
    unittest.main()
