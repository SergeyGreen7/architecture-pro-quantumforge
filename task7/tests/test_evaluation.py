from __future__ import annotations

import sys
import unittest
from pathlib import Path


sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from evaluate import keyword_coverage, load_golden_set, score_result


class EvaluationRulesTest(unittest.TestCase):
    def test_golden_set_is_valid_and_balanced(self) -> None:
        path = Path(__file__).resolve().parents[1] / "golden_questions.txt"
        questions = load_golden_set(path)
        known = [item for item in questions if item["expected_answerable"]]
        absent = [item for item in questions if not item["expected_answerable"]]
        self.assertEqual(len(questions), 12)
        self.assertEqual(len(known), 8)
        self.assertEqual(len(absent), 4)

    def test_keyword_coverage_is_case_insensitive(self) -> None:
        coverage, matched = keyword_coverage(
            "Он использует Фазовую пушку между измерениями.",
            ["фазов", "измерен", "нет"],
        )
        self.assertEqual(coverage, 2 / 3)
        self.assertEqual(matched, ["фазов", "измерен"])

    def test_known_answer_requires_relevant_source(self) -> None:
        golden = {
            "expected_answerable": True,
            "expected_sources": ["Аркон Вейр"],
            "required_keywords": ["фазов"],
        }
        result = score_result(
            golden,
            answer="Аркон использует фазовую пушку.",
            answered_from_context=True,
            retrieved_titles=["Другой документ"],
        )
        self.assertEqual(result["status"], "fail_irrelevant_source")

    def test_absent_topic_passes_only_on_refusal(self) -> None:
        golden = {
            "expected_answerable": False,
            "expected_sources": [],
            "required_keywords": [],
        }
        refused = score_result(
            golden,
            answer="Я не знаю",
            answered_from_context=False,
            retrieved_titles=[],
        )
        leaked = score_result(
            golden,
            answer="Выдуманный достаточно длинный ответ",
            answered_from_context=True,
            retrieved_titles=["Нерелевантный"],
        )
        self.assertEqual(refused["status"], "pass")
        self.assertEqual(leaked["status"], "fail_unexpected_answer")


if __name__ == "__main__":
    unittest.main()
