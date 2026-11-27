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
        self.assertIn(
            "безпосередньо підтверджує",
            post.call_args.kwargs["json"]["messages"][0]["content"],
        )
        self.assertEqual(post.call_args.kwargs["json"]["options"]["temperature"], 0.2)

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
                json={
                    "mode": "quote",
                    "question": "«Цей рядок існує у двох записах»",
                    "allow_ungrounded": True,
                },
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
                json={
                    "question": "Про твір «Невідома назва»?",
                    "mode": "title",
                    "allow_ungrounded": True,
                },
            )
            no_fragment = client.post(
                "/chat",
                json={
                    "question": "«Розрита могила»",
                    "mode": "title",
                    "allow_ungrounded": True,
                },
            )
        self.assertEqual(response.json()["retrieval_status"], "not_found")
        self.assertEqual(response.json()["status"], "refused")
        self.assertEqual(no_fragment.json()["retrieval_status"], "no_relevant_chunk")
        self.assertEqual(no_fragment.json()["status"], "refused")
        post.assert_not_called()

    def test_opt_in_generates_for_general_and_word_definition_questions(self):
        requests_to_try = [
            {"mode": "title", "question": "Як справи?"},
            {"mode": "auto", "question": "Що означає слово «воля»?"},
        ]
        with (
            TestClient(main.app) as client,
            patch.object(
                main.requests, "post", return_value=self.model_response()
            ) as post,
        ):
            for body in requests_to_try:
                with self.subTest(body=body):
                    response = client.post(
                        "/chat", json={**body, "allow_ungrounded": True}
                    )
                    self.assertEqual(response.status_code, 200)
                    data = response.json()
                    self.assertEqual(data["status"], "ungrounded")
                    self.assertEqual(data["sources"], [])
                    self.assertEqual(data["model"], main.MODEL)
                    self.assertEqual(data["answer"], "Дніпро.")
                    messages = post.call_args.kwargs["json"]["messages"]
                    self.assertEqual(
                        messages[0]["content"], main.UNGROUNDED_SYSTEM_PROMPT
                    )
                    self.assertEqual(messages[1]["content"], body["question"])
        self.assertEqual(post.call_count, len(requests_to_try))

    def test_auto_does_not_generate_for_unmatched_or_ambiguous_quoted_sources(self):
        requests_to_try = [
            "Що відбувається у «Невідома назва»?",
            "У якому творі є «ця довга цитата відсутня в корпусі»?",
            "«Цей рядок існує у двох записах»",
        ]
        with (
            TestClient(main.app) as client,
            patch.object(main.requests, "post") as post,
        ):
            for question in requests_to_try:
                with self.subTest(question=question):
                    response = client.post(
                        "/chat",
                        json={
                            "mode": "auto",
                            "question": question,
                            "allow_ungrounded": True,
                        },
                    )
                    self.assertEqual(response.status_code, 200)
                    data = response.json()
                    self.assertEqual(data["status"], "refused")
                    self.assertEqual(data["sources"], [])
                    self.assertIsNone(data["model"])
        post.assert_not_called()

    def test_ungrounded_fallback_respects_sources_mode_and_ollama_errors(self):
        body = {"mode": "title", "question": "Як справи?", "allow_ungrounded": True}
        with TestClient(main.app) as client:
            with patch.object(main.requests, "post") as post:
                response = client.post("/chat", json={**body, "answer_mode": "sources"})
                self.assertEqual(response.json()["status"], "refused")
                post.assert_not_called()
            with patch.object(main.requests, "post", side_effect=requests.Timeout()):
                self.assertEqual(client.post("/chat", json=body).status_code, 504)
            invalid = self.model_response()
            invalid.json.return_value["done_reason"] = "length"
            with patch.object(main.requests, "post", return_value=invalid):
                self.assertEqual(client.post("/chat", json=body).status_code, 502)
            with patch.object(
                main.requests, "post", return_value=self.model_response()
            ) as post:
                self.assertEqual(client.post("/chat", json=body).status_code, 200)
                system_prompt = post.call_args.kwargs["json"]["messages"][0]["content"]
                self.assertIn("максимум двома короткими реченнями", system_prompt)
                self.assertIn(
                    "не вигадуй цитат, частоти вживання, причин", system_prompt
                )
                self.assertIn("попроси назвати твір", system_prompt)

    def test_auto_routes_quotes_and_titles_to_the_appropriate_search(self):
        with (
            TestClient(main.app) as client,
            patch.object(main.requests, "post", return_value=self.model_response()),
        ):
            chunk = client.app.state.chunks[0]
            with patch.object(
                client.app.state.hybrid_search,
                "search",
                return_value=[{"chunk": chunk, "score": 0.1}],
            ) as search:
                body = {
                    "mode": "auto",
                    "retrieval_method": "hybrid",
                    "allow_ungrounded": True,
                }
                title = client.post(
                    "/chat",
                    json={**body, "question": "Яка річка у «Розритій могилі»?"},
                )
                quote = client.post(
                    "/chat",
                    json={**body, "question": "Звідки «Дніпро брат мій висихає»?"},
                )
                general = client.post("/chat", json={**body, "question": "Як справи?"})
                ambiguous = client.post(
                    "/chat",
                    json={**body, "question": "«Цей рядок існує у двох записах»"},
                )
        self.assertEqual(title.status_code, 200)
        self.assertEqual(title.json()["status"], "generated")
        self.assertEqual(title.json()["retrieval_method"], "hybrid")
        self.assertEqual(title.json()["sources"][0]["id"], 1)
        self.assertEqual(search.call_count, 1)
        self.assertEqual(quote.json()["status"], "generated")
        self.assertEqual(quote.json()["retrieval_method"], "lexical")
        self.assertEqual(quote.json()["sources"][0]["id"], 1)
        self.assertEqual(general.json()["status"], "ungrounded")
        self.assertEqual(general.json()["sources"], [])
        self.assertEqual(ambiguous.json()["status"], "refused")
        self.assertEqual(ambiguous.json()["retrieval_status"], "ambiguous")
        self.assertEqual(ambiguous.json()["sources"], [])
        self.assertIsNone(ambiguous.json()["model"])

    def test_auto_lists_every_matching_word_occurrence_without_generation(self):
        works: list[main.retrieval.Work] = [
            {
                "id": 10,
                "title": "Перший твір",
                "source": "https://example.org/10",
                "text": "Сад шумить.\nЗатінок саду.\nБез збігу.",
            },
            {
                "id": 11,
                "title": "Другий твір",
                "source": "https://example.org/11",
                "text": "Сади, за садами.",
            },
        ]
        with (
            TestClient(main.app) as client,
            patch.object(main.requests, "post") as post,
        ):
            client.app.state.corpus = works
            client.app.state.chunks = main.retrieval.build_chunks(works)
            body = {
                "mode": "auto",
                "retrieval_method": "hybrid",
                "allow_ungrounded": True,
            }
            response = client.post(
                "/chat",
                json={
                    **body,
                    "question": 'Знайди всі місця у творах де зустрічається слово "сад"',
                },
            )
            unquoted = client.post(
                "/chat",
                json={
                    **body,
                    "question": "знайди у творах всі випадки коли "
                    "зустрічаеється слово сад",
                },
            )
            missing = client.post(
                "/chat",
                json={
                    **body,
                    "question": 'Знайди всі місця де зустрічається слово "відсутнє"',
                },
            )

        data = response.json()
        self.assertEqual(response.status_code, 200)
        self.assertEqual(data["status"], "sources")
        self.assertEqual(data["retrieval_status"], "term_matches")
        self.assertIn("Збігів з основою «сад» у корпусі: 4", data["answer"])
        self.assertIn("рядків з такими збігами: 3", data["answer"])
        self.assertEqual(
            [source["text"] for source in data["sources"]],
            ["Сад шумить.", "Затінок саду.", "Сади, за садами."],
        )
        self.assertEqual(unquoted.status_code, 200)
        self.assertEqual(unquoted.json()["retrieval_status"], "term_matches")
        self.assertEqual(unquoted.json()["sources"], data["sources"])
        self.assertIsNone(data["model"])
        self.assertEqual(missing.status_code, 200)
        self.assertEqual(missing.json()["retrieval_status"], "term_not_found")
        self.assertEqual(missing.json()["sources"], [])
        post.assert_not_called()

    def test_auto_hybrid_failure_is_not_treated_as_missing_sources(self):
        with (
            TestClient(main.app) as client,
            patch.object(
                client.app.state.hybrid_search,
                "search",
                side_effect=requests.ConnectionError("unavailable"),
            ),
            patch.object(main, "generate_answer") as generate,
        ):
            response = client.post(
                "/chat",
                json={
                    "mode": "auto",
                    "question": "Яка річка у «Розритій могилі»?",
                    "retrieval_method": "hybrid",
                    "allow_ungrounded": True,
                },
            )
        self.assertEqual(response.status_code, 502)
        generate.assert_not_called()

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

    def test_hybrid_search_and_sources_mode_do_not_call_chat_model(self):
        body = {
            "mode": "title",
            "question": "Яка річка у «Розритій могилі»?",
            "retrieval_method": "hybrid",
            "answer_mode": "sources",
        }
        with TestClient(main.app) as client:
            chunk = client.app.state.chunks[0]
            with (
                patch.object(
                    client.app.state.hybrid_search,
                    "search",
                    return_value=[{"chunk": chunk, "score": 0.1}],
                ) as search,
                patch.object(main.requests, "post") as post,
            ):
                response = client.post("/chat", json=body)
        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.json()["status"], "sources")
        self.assertEqual(response.json()["retrieval_method"], "hybrid")
        self.assertEqual(response.json()["sources"][0]["id"], 1)
        self.assertIsNone(response.json()["model"])
        self.assertIn("можуть не містити відповіді", response.json()["answer"])
        self.assertEqual(search.call_count, 1)
        post.assert_not_called()

    def test_hybrid_failure_is_explicit_and_never_falls_back(self):
        with TestClient(main.app) as client:
            with (
                patch.object(
                    client.app.state.hybrid_search,
                    "search",
                    side_effect=requests.ConnectionError("unavailable"),
                ),
                patch.object(main, "generate_answer") as generate,
            ):
                response = client.post(
                    "/chat",
                    json={
                        "mode": "title",
                        "question": "Що у «Розритій могилі»?",
                        "retrieval_method": "hybrid",
                    },
                )
            self.assertEqual(response.status_code, 502)
            generate.assert_not_called()
            self.assertEqual(
                client.post(
                    "/chat",
                    json={
                        "mode": "quote",
                        "question": "«Дніпро брат мій висихає»",
                        "retrieval_method": "hybrid",
                    },
                ).status_code,
                422,
            )

    def test_invalid_embedding_service_response_and_timeout_are_errors(self):
        body = {
            "mode": "title",
            "question": "Що у «Розритій могилі»?",
            "retrieval_method": "hybrid",
        }
        response = Mock()
        response.json.return_value = {"models": [None]}
        with (
            TestClient(main.app) as client,
            patch.object(main, "generate_answer") as generate,
        ):
            with patch("backend.hybrid.requests.get", return_value=response):
                self.assertEqual(client.post("/chat", json=body).status_code, 502)
            with patch("backend.hybrid.requests.get", side_effect=requests.Timeout()):
                self.assertEqual(client.post("/chat", json=body).status_code, 504)
            generate.assert_not_called()

    def test_sources_mode_preserves_quote_refusal(self):
        with (
            TestClient(main.app) as client,
            patch.object(main.requests, "post") as post,
        ):
            result = client.post(
                "/chat",
                json={
                    "mode": "quote",
                    "question": "«Цей рядок існує у двох записах»",
                    "answer_mode": "sources",
                },
            )
        self.assertEqual(result.json()["status"], "refused")
        self.assertEqual(result.json()["retrieval_status"], "ambiguous")
        post.assert_not_called()


if __name__ == "__main__":
    unittest.main()
