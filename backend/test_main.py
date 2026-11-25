import hashlib
import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import Mock, patch

import requests
from fastapi.testclient import TestClient

from backend import main


class ApiTests(unittest.TestCase):
    def setUp(self):
        self.directory = tempfile.TemporaryDirectory()
        self.addCleanup(self.directory.cleanup)
        corpus = Path(self.directory.name) / "corpus.jsonl"
        records = [
            {
                "id": 1,
                "title": "Розрита могила",
                "source": "https://example.org/1",
                "text": "Дніпро, брат мій, висихає, мене покидає.",
            },
            {
                "id": 2,
                "title": "Інший твір",
                "source": "https://example.org/2",
                "text": "Цей рядок існує у двох записах.",
            },
            {
                "id": 3,
                "title": "Третій твір",
                "source": "https://example.org/3",
                "text": "Цей рядок існує у двох записах.",
            },
        ]
        data = "".join(
            json.dumps(record, ensure_ascii=False) + "\n" for record in records
        )
        corpus.write_text(data, encoding="utf-8")
        self.sha256 = hashlib.sha256(data.encode()).hexdigest()
        self.patcher = patch.object(main, "CORPUS_PATH", corpus)
        self.patcher.start()
        self.addCleanup(self.patcher.stop)

    @staticmethod
    def model_response():
        response = Mock()
        response.json.return_value = {
            "done": True,
            "done_reason": "stop",
            "message": {"content": "Дніпро."},
            "prompt_eval_count": 50,
        }
        return response

    def test_title_search_returns_answer_and_verified_source(self):
        with (
            TestClient(main.app) as client,
            patch.object(
                main.requests, "post", return_value=self.model_response()
            ) as post,
        ):
            response = client.post(
                "/chat",
                json={
                    "question": "Хто називає Дніпро братом у «Розритій могилі»?",
                    "mode": "title",
                },
            )
        self.assertEqual(response.status_code, 200)
        data = response.json()
        self.assertEqual(data["status"], "generated")
        self.assertTrue(data["experimental"])
        self.assertEqual(data["retrieval_status"], "matched")
        self.assertEqual(data["corpus_sha256"], self.sha256)
        self.assertEqual(data["sources"][0]["id"], 1)
        self.assertEqual(
            data["sources"][0]["text"], "Дніпро, брат мій, висихає, мене покидає."
        )
        self.assertIn(
            data["sources"][0]["text"],
            post.call_args.kwargs["json"]["messages"][-1]["content"],
        )

    def test_quote_mode_and_ambiguous_refusal_without_generation(self):
        with (
            TestClient(main.app) as client,
            patch.object(
                main.requests, "post", return_value=self.model_response()
            ) as post,
        ):
            result = client.post(
                "/chat",
                json={"mode": "quote", "question": "Звідки «Дніпро брат мій висихає»?"},
            )
            refusal = client.post(
                "/chat",
                json={"mode": "quote", "question": "«Цей рядок існує у двох записах»"},
            )
        self.assertEqual(result.status_code, 200)
        self.assertEqual(result.json()["sources"][0]["id"], 1)
        self.assertIn(
            "Назва знайденого твору: «Розрита могила»",
            post.call_args.kwargs["json"]["messages"][-1]["content"],
        )
        self.assertEqual(refusal.json()["status"], "refused")
        self.assertEqual(refusal.json()["retrieval_status"], "ambiguous")
        self.assertEqual(refusal.json()["sources"], [])
        self.assertEqual(post.call_count, 1)

    def test_invalid_input_and_unknown_title_are_not_generated(self):
        with (
            TestClient(main.app) as client,
            patch.object(main.requests, "post") as post,
        ):
            self.assertEqual(
                client.post(
                    "/chat", json={"question": "  ", "mode": "title"}
                ).status_code,
                422,
            )
            self.assertEqual(
                client.post("/chat", json={"question": "x"}).status_code, 422
            )
            response = client.post(
                "/chat",
                json={"question": "Про твір «Невідома назва»?", "mode": "title"},
            )
            no_fragment = client.post(
                "/chat", json={"question": "«Розрита могила»", "mode": "title"}
            )
        self.assertEqual(response.json()["retrieval_status"], "not_found")
        self.assertEqual(response.json()["status"], "refused")
        self.assertEqual(no_fragment.json()["retrieval_status"], "no_relevant_chunk")
        self.assertEqual(no_fragment.json()["status"], "refused")
        post.assert_not_called()

    def test_ollama_failure_and_truncated_answer_are_errors(self):
        request = {"mode": "quote", "question": "«Дніпро брат мій висихає»"}
        response = self.model_response()
        response.json.return_value["done_reason"] = "length"
        with TestClient(main.app) as client:
            with patch.object(main.requests, "post", side_effect=requests.Timeout()):
                self.assertEqual(client.post("/chat", json=request).status_code, 504)
            with patch.object(main.requests, "post", return_value=response):
                self.assertEqual(client.post("/chat", json=request).status_code, 502)
            response.json.return_value["done_reason"] = "stop"
            response.json.return_value["prompt_eval_count"] = 4096
            with patch.object(main.requests, "post", return_value=response):
                self.assertEqual(client.post("/chat", json=request).status_code, 502)
            with patch.object(
                main.requests, "post", side_effect=requests.ConnectionError()
            ):
                self.assertEqual(client.post("/chat", json=request).status_code, 502)

    def test_missing_corpus_fails_startup(self):
        with (
            patch.object(main, "CORPUS_PATH", Path(self.directory.name) / "missing"),
            self.assertRaises(FileNotFoundError),
            TestClient(main.app),
        ):
            pass


if __name__ == "__main__":
    unittest.main()
