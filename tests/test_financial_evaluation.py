import unittest

from evaluations.financial_eval import _expected_route, _numeric_tokens, _rank_parents, _strategy_allowed


class FinancialEvaluationTests(unittest.TestCase):
    def test_rank_parents_prioritizes_metric_terms(self):
        parents = [
            {"parent_id": "risk", "parent_content": "公司风险管理和治理情况。"},
            {"parent_id": "revenue", "parent_content": "营业收入为 100.00 亿元，营业成本为 20.00 亿元。"},
        ]
        ranked = _rank_parents("营业收入是多少？", parents)
        self.assertEqual("revenue", ranked[0]["parent_id"])

    def test_numeric_tokens_normalize_commas(self):
        self.assertEqual({"90703260964.48", "1.47%"}, _numeric_tokens("90,703,260,964.48 元，1.47%"))

    def test_runtime_names_are_normalized_from_dataset_contract(self):
        self.assertEqual("OUT_OF_SCOPE", _expected_route({"expected_route": "OOS"}))
        self.assertTrue(_strategy_allowed({"allowed_strategies": ["Direct", "HyDE"]}, "直接检索"))
        self.assertIsNone(_strategy_allowed({"allowed_strategies": ["OOS"]}, None))


if __name__ == "__main__":
    unittest.main()
