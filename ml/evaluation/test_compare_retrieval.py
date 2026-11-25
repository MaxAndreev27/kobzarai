import unittest
from unittest.mock import Mock, patch

import compare_retrieval
import retrieval


def chunk(start: int) -> retrieval.Chunk:
    return {
        "id": 1,
        "title": "Приклад",
        "source": "https://example.org/1",
        "start_char": start,
        "end_char": start + 5,
        "text": "текст",
        "is_first": start == 0,
    }


class CompareRetrievalTests(unittest.TestCase):
    def test_dense_cosine_ranking_and_dimension_mismatch(self):
        hits = compare_retrieval.rank_dense(
            [1.0, 0.0], [chunk(0), chunk(10)], [[0.0, 1.0], [1.0, 0.0]]
        )
        self.assertEqual([hit["chunk"]["start_char"] for hit in hits], [10, 0])
        self.assertEqual(hits[0]["score"], 1.0)
        with self.assertRaises(ValueError):
            compare_retrieval.rank_dense([1.0], [chunk(0)], [[1.0, 0.0]])
        with self.assertRaises(ValueError):
            compare_retrieval.rank_dense([1.0], [chunk(0)], [])

    def test_embedding_request_disables_truncation_and_requires_gpu(self):
        embedding = Mock()
        embedding.json.return_value = {"embeddings": [[3.0, 4.0]]}
        placement = Mock()
        placement.json.return_value = {
            "models": [{"digest": "model-digest", "size_vram": 123}]
        }
        with (
            patch.object(
                compare_retrieval.requests, "post", return_value=embedding
            ) as post,
            patch.object(compare_retrieval.requests, "get", return_value=placement),
        ):
            vectors, gpu = compare_retrieval.embed(["текст"], "model", "model-digest")
        self.assertEqual(vectors, [[0.6, 0.8]])
        self.assertEqual(gpu["size_vram"], 123)
        self.assertFalse(post.call_args.kwargs["json"]["truncate"])

        placement.json.return_value = {
            "models": [{"digest": "model-digest", "size_vram": 0}]
        }
        with (
            patch.object(compare_retrieval.requests, "post", return_value=embedding),
            patch.object(compare_retrieval.requests, "get", return_value=placement),
            self.assertRaisesRegex(RuntimeError, "GPU placement not confirmed"),
        ):
            compare_retrieval.embed(["текст"], "model", "model-digest")

    def test_invalid_embedding_is_rejected(self):
        for vector in ([], [0.0, 0.0], [float("nan")]):
            with self.subTest(vector=vector), self.assertRaises(ValueError):
                compare_retrieval.normalize_vector(vector)


if __name__ == "__main__":
    unittest.main()
