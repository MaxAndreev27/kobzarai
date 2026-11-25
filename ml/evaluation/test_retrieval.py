import json
import unittest
from pathlib import Path
from unittest.mock import patch

import evaluate_retrieval
import retrieval


def work(identifier: int, title: str, text: str = "Початок твору.") -> retrieval.Work:
    return {
        "id": identifier,
        "title": title,
        "source": f"https://example.org/{identifier}",
        "text": text,
    }


class RetrievalTests(unittest.TestCase):
    def test_quote_search_finds_original_chunks_without_title_or_expected_id(self):
        corpus = [
            work(1, "Твір перший", "Реве та стогне\nДніпр широкий, сердитий вітер."),
            work(2, "Інший твір", "Це зовсім інший уривок."),
        ]
        chunks = retrieval.build_chunks(corpus)
        status, hits = retrieval.search_quote(
            "З якого твору «РЕВЕ ТА СТОГНЕ ДНІПР, широкий»?", corpus, chunks
        )
        self.assertEqual(status, "matched")
        self.assertEqual(hits[0]["chunk"]["id"], 1)
        self.assertEqual(hits[0]["chunk"]["text"], corpus[0]["text"])

    def test_quote_search_rejects_missing_short_multiple_and_ambiguous_quotes(self):
        quote = "Зелене поле вітер тихо колише"
        corpus = [work(1, "Перший", quote), work(2, "Другий", quote)]
        chunks = retrieval.build_chunks(corpus)
        for query, expected in (
            ("Знайди рядок без лапок", "no_quote"),
            ("Знайди «Зелене поле»", "quote_too_short"),
            ("Знайди «Я п'ю чай»", "quote_too_short"),
            ("Знайди «Не знайдено таких рядків»", "not_found"),
            ("Знайди «Зелене поле вітер тихо колише»", "ambiguous"),
            (
                "«Зелене поле вітер тихо колише» і «Ще одна довга цитата»",
                "multiple_quotes",
            ),
        ):
            with self.subTest(query=query):
                status, hits = retrieval.search_quote(query, corpus, chunks)
                self.assertEqual(status, expected)
                self.assertEqual(hits, [])

    def test_quote_search_requires_full_quote_in_a_chunk(self):
        corpus = [work(1, "Перший", "один два три чотири п'ять шість")]
        chunks = retrieval.build_chunks([work(1, "Перший", "один два три чотири")])
        status, hits = retrieval.search_quote("«два три чотири п'ять»", corpus, chunks)
        self.assertEqual(status, "no_relevant_chunk")
        self.assertEqual(hits, [])

    def test_quote_search_requires_word_boundaries(self):
        corpus = [work(1, "Перший", "передмова зелене поле вітер тихо колише")]
        status, hits = retrieval.search_quote(
            "«мова зелене поле вітер»", corpus, retrieval.build_chunks(corpus)
        )
        self.assertEqual((status, hits), ("not_found", []))

    def test_quote_in_overlapping_chunks_is_not_ambiguous(self):
        text = "Слово " * 120 + "зелене поле вітер тихо колише " + "слово " * 100
        corpus = [work(1, "Перший", text)]
        status, hits = retrieval.search_quote(
            "«зелене поле вітер тихо колише»", corpus, retrieval.build_chunks(corpus)
        )
        self.assertEqual(status, "matched")
        self.assertEqual(len(hits), 2)
        self.assertEqual({hit["chunk"]["id"] for hit in hits}, {1})

    def test_title_and_inflected_title_match(self):
        corpus = [
            work(1, "Катерина"),
            work(2, "Заповіт"),
            work(3, "Наймичка"),
            work(4, "Гайдамаки"),
            work(5, "Причинна"),
        ]
        for query, expected in [
            ("Про що поема «Катерина»?", 1),
            ("Які теми «Катерини»?", 1),
            ("Розкажи про «Заповіту».", 2),
            ("Що відбувається в «Наймичці»?", 3),
            ("Про що розповідається на початку «Гайдамаків»?", 4),
            ("Що відбувається у Наймичці?", 3),
            ("Що сказав автор Гайдамаками?", 4),
            ("Яку картину показано у «Причинній»?", 5),
            ("Про що в «Причинної»?", 5),
            ("Розкажи про Причинну.", 5),
            ("Що спільного з Причинною?", 5),
        ]:
            with self.subTest(query=query):
                status, result = retrieval.find_work(query, corpus)
                self.assertEqual(status, "matched")
                self.assertIsNotNone(result)
                assert result is not None
                self.assertEqual(result["id"], expected)

    def test_explicit_unknown_title_does_not_match_known_title_or_other_words(self):
        corpus = [work(1, "Заповіт"), work(2, "Доля")]
        for query in (
            "Про що твір «Заповіт космічного мандрівника»?",
            "Про що поема «Небесний зореліт», де згадується доля?",
            'Про що вірш "Заповіт мандрівника"?',
            "Що скажеш про «Заповіт космічного мандрівника»?",
        ):
            with self.subTest(query=query):
                self.assertEqual(
                    retrieval.find_work(query, corpus), ("not_found", None)
                )

    def test_inflected_multiword_title_is_matched_as_a_whole(self):
        corpus = [work(1, "Розрита могила"), work(2, "Могила")]
        for title in (
            "Розрита могила",
            "Розритої могили",
            "Розритій могилі",
            "Розриту могилу",
            "Розритою могилою",
        ):
            for query in (f"Що сказано у «{title}»?", f"Що сказано у {title}?"):
                with self.subTest(query=query):
                    status, result = retrieval.find_work(query, corpus)
                    self.assertEqual(status, "matched")
                    assert result is not None
                    self.assertEqual(result["id"], 1)
        self.assertEqual(
            retrieval.find_work("Про твір «Розритої могили історія»?", corpus),
            ("not_found", None),
        )

    def test_quoted_titles_and_unrelated_quotation_remain_independent(self):
        corpus = [work(1, "Заповіт"), work(2, "Наймичка")]
        status, result = retrieval.find_work(
            "Що означає «лан широкий» у творі «Заповіт»?", corpus
        )
        self.assertEqual(status, "matched")
        assert result is not None
        self.assertEqual(result["id"], 1)
        self.assertEqual(
            retrieval.find_work("Порівняй «Заповіт» і «Наймичку»", corpus),
            ("ambiguous", None),
        )

    def test_unknown_boundary_and_ambiguous_names(self):
        corpus = [work(1, "Катерина"), work(2, "Кавказ"), work(3, "Причинна")]
        self.assertEqual(
            retrieval.find_work("Катеринбург", corpus), ("not_found", None)
        )
        self.assertEqual(
            retrieval.find_work("«Катерина» і «Кавказ»", corpus), ("ambiguous", None)
        )
        query = "Чи існує твір «Невідомий»?"
        self.assertEqual(retrieval.find_work(query, corpus), ("not_found", None))
        self.assertIn("не доводить", retrieval.add_lookup_warning(query, "not_found"))
        for query in ("Про вірш «Причинність»?", "Про твір «Причинної історії»?"):
            with self.subTest(query=query):
                self.assertEqual(
                    retrieval.find_work(query, corpus), ("not_found", None)
                )
        self.assertEqual(
            retrieval.find_work("Порівняй «Причинній» і «Катерина»", corpus),
            ("ambiguous", None),
        )

    def test_longer_overlapping_title_wins(self):
        corpus = [work(1, "Берег"), work(2, "Берег мрій")]
        status, result = retrieval.find_work("Про «Берег мрій»", corpus)
        self.assertEqual(status, "matched")
        assert result is not None
        self.assertEqual(result["id"], 2)

    def test_context_is_bounded_and_identifies_source(self):
        entry = work(1, "Катерина", ("Перший рядок.\n" * 200))
        fragment = retrieval.excerpt(entry)
        context = retrieval.add_context("Про що твір?", entry)
        self.assertLessEqual(len(fragment), retrieval.MAX_EXCERPT_CHARS)
        self.assertTrue(entry["text"].startswith(fragment))
        self.assertIn(entry["source"], context)
        self.assertIn("лише початковий фрагмент", context)

    def test_corpus_validation(self):
        valid = work(1, "Катерина")
        raw = json.dumps(valid, ensure_ascii=False).encode()
        self.assertEqual(retrieval.load_corpus(raw)[0]["id"], 1)
        with self.assertRaises(ValueError):
            retrieval.load_corpus(b"")
        with self.assertRaises(ValueError):
            retrieval.load_corpus(raw + b"\n" + raw)
        with self.assertRaises(ValueError):
            retrieval.load_corpus(b'{"id": 1}')

    def test_explicit_work_beginning_selects_first_chunk(self):
        corpus = [
            work(1, "Гайдамаки", "На світанку тиха річка.\n" * 50 + "Сонце і зорі.")
        ]
        chunks = retrieval.build_chunks(corpus)
        hits = retrieval.search_chunks(
            "Що про сонце на початку «Гайдамаки»?",
            chunks,
            work_id=1,
            exclude_title="Гайдамаки",
            top_k=1,
        )
        self.assertEqual(hits[0]["chunk"]["start_char"], 0)
        hits = retrieval.search_chunks(
            "Що на початку «Гайдамаки»?",
            chunks,
            work_id=1,
            exclude_title="Гайдамаки",
            top_k=1,
        )
        self.assertEqual(hits[0]["chunk"]["start_char"], 0)

    def test_section_beginning_or_negated_beginning_is_not_work_beginning(self):
        corpus = [
            work(1, "Гайдамаки", "На світанку тиха річка.\n" * 50 + "Сонце і зорі.")
        ]
        chunks = retrieval.build_chunks(corpus)
        for query in (
            "Що про сонце на початку розділу в «Гайдамаки»?",
            "Що про сонце не на початку «Гайдамаки»?",
            "Що про сонце не в перших рядках «Гайдамаки»?",
        ):
            with self.subTest(query=query):
                hits = retrieval.search_chunks(
                    query, chunks, work_id=1, exclude_title="Гайдамаки", top_k=1
                )
                self.assertGreater(hits[0]["chunk"]["start_char"], 0)

    def test_position_requires_the_actual_first_chunk(self):
        corpus = [work(1, "Гайдамаки", " \n" + "Перший рядок.\n" * 100)]
        chunks = retrieval.build_chunks(corpus)
        self.assertGreater(chunks[0]["start_char"], 0)
        self.assertTrue(chunks[0]["is_first"])
        self.assertTrue(
            retrieval.search_chunks(
                "Що на початку «Гайдамаки»?",
                chunks,
                work_id=1,
                exclude_title="Гайдамаки",
            )
        )
        self.assertEqual(
            retrieval.search_chunks(
                "Що на початку «Гайдамаки»?",
                chunks[1:],
                work_id=1,
                exclude_title="Гайдамаки",
            ),
            [],
        )

    def test_inflected_title_does_not_dominate_chunk_ranking(self):
        chunks: list[retrieval.Chunk] = [
            {
                "id": 1,
                "title": "Наймичка",
                "source": "https://example.org/1",
                "start_char": 0,
                "end_char": 9,
                "text": "Наймичці",
                "is_first": True,
            },
            {
                "id": 1,
                "title": "Наймичка",
                "source": "https://example.org/1",
                "start_char": 10,
                "end_char": 15,
                "text": "Марко",
                "is_first": False,
            },
        ]
        hits = retrieval.search_chunks(
            "«Наймичці» Марко", chunks, work_id=1, exclude_title="Наймичка", top_k=1
        )
        self.assertEqual(hits[0]["chunk"]["start_char"], 10)

    def test_adjectival_title_is_excluded_from_chunk_ranking(self):
        corpus = [work(1, "Причинна", "Причинній\n" * 100 + "Марко")]
        hits = retrieval.search_chunks(
            "«Причинній» Марко",
            retrieval.build_chunks(corpus),
            work_id=1,
            exclude_title="Причинна",
        )
        self.assertTrue(hits)
        self.assertIn("Марко", hits[0]["chunk"]["text"])

    def test_retrieval_cases_find_the_expected_source_fragment(self):
        project = Path(__file__).resolve().parents[1]
        corpus = retrieval.load_corpus(retrieval.CORPUS.read_bytes())
        chunks = retrieval.build_chunks(corpus)
        cases = [
            json.loads(line)
            for line in (project / "evaluation/retrieval_cases.jsonl")
            .read_text(encoding="utf-8")
            .splitlines()
        ]
        self.assertEqual(len(cases), 8)
        for case in cases:
            with self.subTest(case=case["id"]):
                work = next(item for item in corpus if item["id"] == case["work_id"])
                status, found = retrieval.find_work(case["query"], corpus)
                self.assertEqual(status, "matched")
                self.assertIsNotNone(found)
                assert found is not None
                self.assertEqual(found["id"], work["id"])
                hits = retrieval.search_chunks(
                    case["query"],
                    chunks,
                    work_id=found["id"],
                    exclude_title=found["title"],
                )
                self.assertTrue(hits)
                self.assertTrue(
                    any(case["evidence"] in hit["chunk"]["text"] for hit in hits)
                )

    def test_end_to_end_evaluator_does_not_use_expected_id_for_search(self):
        corpus = [work(1, "Катерина", "Кохайтеся, чорнобриві.")]
        chunks = retrieval.build_chunks(corpus)
        case: evaluate_retrieval.Case = {
            "id": "test",
            "query": "Які слова про кохайтеся у «Катерині»?",
            "work_id": 999,
            "evidence": "Кохайтеся, чорнобриві.",
        }
        with patch.object(
            retrieval, "search_chunks", wraps=retrieval.search_chunks
        ) as search:
            result = evaluate_retrieval.evaluate_case(case, corpus, chunks)
        self.assertEqual(search.call_args.kwargs["work_id"], 1)
        self.assertEqual(result["actual_work_id"], 1)
        self.assertFalse(result["title_correct"])
        self.assertEqual(result["evidence_rank"], 1)

    def test_end_to_end_evaluator_handles_unknown_and_no_relevant_chunk(self):
        corpus = [work(1, "Катерина", "Кохайтеся, чорнобриві.")]
        chunks = retrieval.build_chunks(corpus)
        unknown: evaluate_retrieval.Case = {
            "id": "unknown",
            "query": "Про твір «Невідомий»?",
            "expected_status": "not_found",
        }
        result = evaluate_retrieval.evaluate_case(unknown, corpus, chunks)
        self.assertTrue(result["title_correct"])
        self.assertEqual(result["hits"], [])

        no_hits: evaluate_retrieval.Case = {
            "id": "no-hits",
            "query": "Про твір «Катерина»?",
            "work_id": 1,
            "evidence": "Кохайтеся, чорнобриві.",
        }
        result = evaluate_retrieval.evaluate_case(no_hits, corpus, chunks)
        self.assertTrue(result["title_correct"])
        self.assertIsNone(result["evidence_rank"])
        self.assertEqual(result["hits"], [])


if __name__ == "__main__":
    unittest.main()
