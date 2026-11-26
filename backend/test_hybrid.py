import unittest
from unittest.mock import Mock, patch

from backend.hybrid import HybridSearch
from ml.evaluation import retrieval


class HybridSearchTests(unittest.TestCase):
    def setUp(self):
        self.work: retrieval.Work = {
            "id": 1,
            "title": "Приклад",
            "source": "https://example.org",
            "text": "Перша частина." * 70 + "\nВідповідь друга частина.",
        }
        self.chunks = retrieval.build_chunks([self.work])

    def test_dense_retrieval_works_without_lexical_match_and_caches(self):
        search = HybridSearch()
        vectors = [[1.0, 0.0] for _ in self.chunks]
        with (
            patch.object(search, "_model_digest", return_value="v1"),
            patch.object(
                search, "_embed", side_effect=[vectors, [[1.0, 0.0]], [[1.0, 0.0]]]
            ) as embed,
        ):
            for _ in range(2):
                hits = search.search("запит без збігу", self.work, self.chunks)
                self.assertEqual(hits[0]["chunk"]["id"], 1)
        self.assertEqual(embed.call_count, 3)

    def test_model_change_invalidates_cached_vectors(self):
        search = HybridSearch()
        search.digest = "old"
        search.vectors[1] = [[0.0, 1.0] for _ in self.chunks]
        with (
            patch.object(search, "_model_digest", return_value="new"),
            patch.object(
                search,
                "_embed",
                side_effect=[
                    [[1.0, 0.0] for _ in self.chunks],
                    [[1.0, 0.0]],
                ],
            ) as embed,
        ):
            search.search("запит", self.work, self.chunks)
        self.assertEqual(search.digest, "new")
        self.assertEqual(embed.call_count, 2)

    def test_embed_rejects_bad_vectors(self):
        response = Mock()
        response.json.return_value = {"embeddings": [[0.0, 0.0]]}
        with (
            patch("backend.hybrid.requests.post", return_value=response),
            self.assertRaises(ValueError),
        ):
            HybridSearch()._embed(["запит"])

    def test_malformed_ollama_responses_raise_error(self):
        response = Mock()
        for payload in (None, {}, {"models": [None]}, {"models": [{"name": 1}]}):
            with (
                self.subTest(payload=payload),
                patch("backend.hybrid.requests.get", return_value=response),
                self.assertRaises((ValueError, TypeError)),
            ):
                response.json.return_value = payload
                HybridSearch()._model_digest()
        for payload in (None, {}, {"embeddings": None}):
            with (
                self.subTest(payload=payload),
                patch("backend.hybrid.requests.post", return_value=response),
                self.assertRaises(TypeError),
            ):
                response.json.return_value = payload
                HybridSearch()._embed(["запит"])

    def test_changing_model_during_request_is_an_error(self):
        search = HybridSearch()
        with (
            patch.object(search, "_model_digest", side_effect=["old", "new"]),
            patch.object(
                search,
                "_embed",
                side_effect=[
                    [[1.0, 0.0] for _ in self.chunks],
                    [[1.0, 0.0]],
                ],
            ),
            self.assertRaisesRegex(ValueError, "changed"),
        ):
            search.search("запит", self.work, self.chunks)
        self.assertEqual(search.vectors, {})
        self.assertIsNone(search.digest)


if __name__ == "__main__":
    unittest.main()
