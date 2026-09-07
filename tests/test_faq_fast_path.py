import unittest

from mysql_qa.retrieval.bm25_search import BM25Search, ORIGIN_QUESTION_KEY


class FakeRedisClient:
    def __init__(self):
        self.answers = {}

    def get_answer(self, question):
        return self.answers.get(question)

    def set_answer(self, question, answer):
        self.answers[question] = answer


class FakeMysqlClient:
    def __init__(self, answers):
        self.answers = answers
        self.fetch_questions_called = False

    def fetch_questions(self):
        self.fetch_questions_called = True
        return []

    def fetch_answer(self, question):
        return self.answers.get(question)

    def fetch_faq_search_documents(self, questions):
        return list(questions)


class FakeBm25:
    def __init__(self, scores):
        self.scores = scores

    def get_scores(self, _tokens):
        return self.scores


class FaqFastPathTests(unittest.TestCase):
    def test_exact_mysql_question_bypasses_bm25_threshold_and_is_cached(self):
        redis_client = FakeRedisClient()
        mysql_client = FakeMysqlClient({"什么是营业收入？": "营业收入定义。"})
        search = BM25Search.__new__(BM25Search)
        search.redis_client = redis_client
        search.mysql_client = mysql_client
        search.bm25 = None
        search.original_questions = []

        answer, need_rag = search.query("什么是营业收入？", threshold=0.85)

        self.assertEqual("营业收入定义。", answer)
        self.assertFalse(need_rag)
        self.assertEqual("营业收入定义。", redis_client.answers["什么是营业收入？"])

    def test_specific_report_question_does_not_match_a_different_faq_question(self):
        redis_client = FakeRedisClient()
        mysql_client = FakeMysqlClient({"什么是营业收入？": "营业收入定义。"})
        search = BM25Search.__new__(BM25Search)
        search.redis_client = redis_client
        search.mysql_client = mysql_client
        search.bm25 = None
        search.original_questions = []

        answer, need_rag = search.query("贵州茅台2026年上半年营业收入是多少？", threshold=0.85)

        self.assertIsNone(answer)
        self.assertTrue(need_rag)

    def test_company_query_bypasses_even_a_populated_exact_answer_cache(self):
        redis_client = FakeRedisClient()
        redis_client.answers["招商银行2026年上半年净息差是多少？"] = "stale generic answer"
        mysql_client = FakeMysqlClient({"银行的净息差是什么？": "净息差定义。"})
        search = BM25Search.__new__(BM25Search)
        search.redis_client = redis_client
        search.mysql_client = mysql_client
        search.bm25 = FakeBm25([0.0, 1.0])
        search.original_questions = ["其他问题", "银行的净息差是什么？"]

        answer, need_rag = search.query("招商银行2026年上半年净息差是多少？", threshold=0.85)

        self.assertIsNone(answer)
        self.assertTrue(need_rag)

    def test_definition_paraphrase_uses_the_conservative_faq_fallback_threshold(self):
        redis_client = FakeRedisClient()
        mysql_client = FakeMysqlClient({"ROE 是什么，应该怎么看？": "ROE 定义。"})
        search = BM25Search.__new__(BM25Search)
        search.redis_client = redis_client
        search.mysql_client = mysql_client
        search.bm25 = FakeBm25([0.0, 1.0])
        search.original_questions = ["其他问题", "ROE 是什么，应该怎么看？"]

        answer, need_rag = search.query("ROE是什么？", threshold=0.85)

        self.assertEqual("ROE 定义。", answer)
        self.assertFalse(need_rag)

    def test_non_definition_query_keeps_the_caller_threshold(self):
        redis_client = FakeRedisClient()
        mysql_client = FakeMysqlClient({"ROE 是什么，应该怎么看？": "ROE 定义。"})
        search = BM25Search.__new__(BM25Search)
        search.redis_client = redis_client
        search.mysql_client = mysql_client
        search.bm25 = FakeBm25([0.0, 1.0])
        search.original_questions = ["其他问题", "ROE 是什么，应该怎么看？"]

        answer, need_rag = search.query("ROE怎么计算", threshold=0.85)

        self.assertIsNone(answer)
        self.assertTrue(need_rag)

    def test_definition_query_recognizes_natural_language_variants(self):
        self.assertTrue(BM25Search._is_definition_query("营业收入是什么意思？"))
        self.assertTrue(BM25Search._is_definition_query("ROE是干什么的？"))
        self.assertTrue(BM25Search._is_definition_query("什么叫经营现金流？"))

    def test_keyword_enriched_documents_keep_canonical_questions_separate(self):
        redis_client = FakeRedisClient()

        class KeywordMysqlClient(FakeMysqlClient):
            def fetch_faq_search_documents(self, questions):
                return ["ROE 是什么，应该怎么看？ ROE 净资产收益率 股东权益"]

        mysql_client = KeywordMysqlClient({"ROE 是什么，应该怎么看？": "ROE 定义。"})
        redis_client.get_data = lambda key: ["ROE 是什么，应该怎么看？"] if key == ORIGIN_QUESTION_KEY else [["旧"]]
        search = BM25Search(redis_client, mysql_client)

        self.assertEqual(["ROE 是什么，应该怎么看？"], search.original_questions)
        self.assertIn("净资产", search.questions[0])
        self.assertIn("收益率", search.questions[0])


if __name__ == "__main__":
    unittest.main()
