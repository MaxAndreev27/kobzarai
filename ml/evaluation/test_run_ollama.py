import hashlib
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

    def test_rag_selected_case_uses_query_not_expected_source(self):
        with tempfile.TemporaryDirectory() as directory:
            directory = Path(directory)
            output = directory / "results.jsonl"
            dataset = directory / "cases.jsonl"
            corpus = directory / "corpus.jsonl"
            case = {
                "id": "test-01",
                "split": "development",
                "messages": [
                    {
                        "role": "user",
                        "content": "Які слова про кохання в поемі «Катерина»?",
                    }
                ],
                "source_ids": [999],
            }
            dataset.write_text(json.dumps(case, ensure_ascii=False), encoding="utf-8")
            corpus.write_text(
                json.dumps(
                    {
                        "id": 1,
                        "title": "Катерина",
                        "text": "У поемі є слова про кохання.",
                        "source": "https://example.org/1",
                    },
                    ensure_ascii=False,
                ),
                encoding="utf-8",
            )
            response = Response({"done": True, "message": {"content": "Відповідь."}})
            with (
                patch.object(run_ollama, "DATASET", dataset),
                patch.object(
                    sys,
                    "argv",
                    [
                        "run_ollama.py",
                        "--rag",
                        "--corpus",
                        str(corpus),
                        "--case-id",
                        "test-01",
                        "--output",
                        str(output),
                    ],
                ),
                patch.object(
                    run_ollama.requests,
                    "get",
                    side_effect=[
                        Response({"version": "test"}),
                        Response(
                            {"models": [{"name": "qwen3:1.7b", "digest": "test"}]}
                        ),
                    ],
                ),
                patch.object(
                    run_ollama.requests, "post", return_value=response
                ) as post,
            ):
                status = run_ollama.main()
            metadata, result = [
                json.loads(line)
                for line in output.read_text(encoding="utf-8").splitlines()
            ]
            self.assertEqual(status, 0)
            self.assertTrue(metadata["rag"])
            self.assertEqual(
                metadata["corpus_sha256"],
                hashlib.sha256(corpus.read_bytes()).hexdigest(),
            )
            self.assertEqual(result["retrieval"]["status"], "matched")
            self.assertEqual(result["retrieval"]["hits"][0]["id"], 1)
            self.assertIn(
                "У поемі є слова про кохання.",
                post.call_args.kwargs["json"]["messages"][-1]["content"],
            )

    def test_quote_mode_uses_found_sources_and_refuses_ambiguous_without_generation(
        self,
    ):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            output, dataset, corpus = (
                root / name for name in ("output", "cases", "corpus")
            )
            cases = [
                {
                    "id": "unique",
                    "split": "development",
                    "messages": [
                        {"role": "user", "content": "Знайди «Вітер гне дерево донизу»."}
                    ],
                    "source_ids": [999],
                },
                {
                    "id": "ambiguous",
                    "split": "development",
                    "messages": [
                        {"role": "user", "content": "Знайди «Раз два три чотири»."}
                    ],
                },
                {
                    "id": "held-out",
                    "split": "holdout",
                    "messages": [
                        {"role": "user", "content": "Знайди «Вітер гне дерево донизу»."}
                    ],
                },
            ]
            works = [
                {
                    "id": 1,
                    "title": "Перший",
                    "text": "Вітер гне дерево донизу.\nРаз два три чотири.",
                    "source": "https://example.org/1",
                },
                {
                    "id": 2,
                    "title": "Другий",
                    "text": "Раз два три чотири.",
                    "source": "https://example.org/2",
                },
            ]
            dataset.write_text(
                "".join(json.dumps(case, ensure_ascii=False) + "\n" for case in cases),
                encoding="utf-8",
            )
            corpus.write_text(
                "".join(json.dumps(work, ensure_ascii=False) + "\n" for work in works),
                encoding="utf-8",
            )
            response = Response({"done": True, "message": {"content": "Перший"}})
            with (
                patch.object(run_ollama.retrieval, "CORPUS", corpus),
                patch.object(
                    sys,
                    "argv",
                    [
                        "run_ollama.py",
                        "--rag",
                        "--quote-search",
                        "--dataset",
                        str(dataset),
                        "--output",
                        str(output),
                    ],
                ),
                patch.object(
                    run_ollama.requests,
                    "get",
                    side_effect=[
                        Response({"version": "test"}),
                        Response(
                            {"models": [{"name": "qwen3:1.7b", "digest": "test"}]}
                        ),
                    ],
                ),
                patch.object(
                    run_ollama.requests, "post", return_value=response
                ) as post,
            ):
                status = run_ollama.main()
            records = [
                json.loads(line)
                for line in output.read_text(encoding="utf-8").splitlines()
            ]
            self.assertEqual(status, 0)
            self.assertEqual(post.call_count, 1)
            self.assertTrue(records[0]["quote_search"])
            self.assertEqual(records[0]["case_count"], 2)
            self.assertEqual(
                records[0]["system_prompt"], run_ollama.QUOTE_SYSTEM_PROMPT
            )
            self.assertEqual(records[1]["retrieval"]["hits"][0]["id"], 1)
            self.assertIn(
                "Вітер гне дерево донизу.",
                post.call_args.kwargs["json"]["messages"][-1]["content"],
            )
            self.assertIn(
                "Назва знайденого твору: «Перший».",
                post.call_args.kwargs["json"]["messages"][-1]["content"],
            )
            self.assertEqual(records[2]["type"], "refusal")
            self.assertEqual(records[2]["retrieval"]["status"], "ambiguous")
            self.assertIn("кількох записах", records[2]["answer"])


if __name__ == "__main__":
    unittest.main()
