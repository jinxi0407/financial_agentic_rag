# retrieval/bm25_search.py
# 导入 BM25 算法
from rank_bm25 import BM25Okapi
# 导入数值计算库
import numpy as np
# 导入文本预处理
from mysql_qa.utils.preprocess import preprocess_text
# 导入日志
from base.logger import logger

from mysql_qa.db.mysql_client import MysqlClient
from mysql_qa.cache.redis_client import RedisClient
from base.config import config
from rag_qa.core.query_metadata import extract_query_metadata, should_bypass_faq

ORIGIN_QUESTION_KEY = f'{config.REDIS_KEY_PREFIX}faq:origin_questions'
QUESTION_KEY = f'{config.REDIS_KEY_PREFIX}faq:questions'


class BM25Search:
    def __init__(self, redis_client: RedisClient, mysql_client: MysqlClient):
        # 初始化日志
        self.logger = logger
        # 初始化 Redis 客户端
        self.redis_client = redis_client
        # 初始化 MySQL 客户端
        self.mysql_client = mysql_client

        # 初始化 BM25 模型
        self.bm25 = None
        # 初始化问题列表（分词以后的问题）
        self.questions = None
        # 初始化原始问题（没有分词的问题）
        self.original_questions = None

        # 0.17MB

        # 加载数据
        self._load_data()

    """
    需求：实现FAQ模块的数据加载部分。（从mysql到bm25）
    思路步骤：
        1. 判断系统是不是第一次启动
        2. 如果是第一次启动
        2.1 从mysql中拉取所有的问题
        2.2 把所有的问题写入redis中（origin_questions）
        2.3 把所有的问题调用预处理模块进行分词，写入redis中（questions）
        3. 构建bm25检索器，传入（questions）
    """

    def _load_data(self):
        # 1. 判断系统是不是第一次启动
        # 判断redis中是否存在问题和分词后的问题
        origin_questions = self.redis_client.get_data(ORIGIN_QUESTION_KEY)
        questions = self.redis_client.get_data(QUESTION_KEY)

        # if list 什么都不写，含义是： 1. 非空列表 2.非None
        # 2. 如果是第一次启动
        if not origin_questions or not questions:
            self.logger.info('第一次启动，redis中没有问题数据，尝试从mysql中进行加载')
            # 2.1 从mysql中拉取所有的问题
            origin_questions = self.mysql_client.fetch_questions()
            if not origin_questions:
                self.logger.warning("Financial FAQ 当前为空，跳过 BM25 索引初始化")
                self.original_questions = []
                self.questions = []
                self.bm25 = None
                return
            else:
                self.logger.info("从mysql中获取高频问题成功，共{}条".format(len(origin_questions)))

            # 2.2 把所有的问题写入redis中（origin_questions）
            # set_data : TODO 为什么要把所有的问题以JSON格式存储
            # 1. 使用方式上：问题和分词后的所有的问题，没有按照顺序读取、按index读取的使用方式。都是全部一次读写
            # 2. redis的数据结构中，字符串简单。并且所有的问题可以压缩成json字符串，直接按字符串存储，就可以满足使用方式。
            self.redis_client.set_data(ORIGIN_QUESTION_KEY, origin_questions)
            self.logger.info("所有的问题已经写入redis中，共{}条".format(len(origin_questions)))

            # 2.3 把所有的问题调用预处理模块进行分词，写入redis中（questions）
            questions = [preprocess_text(origin_question) for origin_question in origin_questions]
            self.redis_client.set_data(QUESTION_KEY, questions)
            self.logger.info("分词后的所有的问题已经写入redis中，共{}条".format(len(questions)))
        else:
            self.logger.info("redis中存在问题和分词后的问题缓存数据，直接加载".format(len(questions)))

        # Search texts are rebuilt in memory from canonical questions plus
        # MySQL keywords. Redis remains a cache for the canonical order only.
        search_documents = self.mysql_client.fetch_faq_search_documents(origin_questions)
        questions = [preprocess_text(search_document) for search_document in search_documents]

        # 3. 构建bm25检索器，传入（questions）
        self.original_questions = origin_questions
        self.questions = questions
        # questions: 467条待查询的文档
        self.bm25 = BM25Okapi(questions)

        logger.info(f"bm25模型初始化成功！")

    """
    需求：实现FAQ模块的查询功能，用户输入一个query，返回超过阈值的概率的问题对应的答案
    思路步骤：
    1. 判断用户的query是否合法，非空字符串
    2. 尝试去redis中查找，一模一样的问题
    3. 对query进行预处理，得到分词数据
    4. 通过bm25检索器得到query和每个问题的相似度分数：[n] 
    5. 对相似度分数进行归一化
    6. 取相似度分数归一化以后的最大值，判断是否大于给定的阈值；如果小于阈值，返回空答案
    7. 如果大于阈值，通过argmax找到最大分数对应的索引
    8. 根据索引找到对应的原始问题（python内存）
    9. 查看redis缓存中是否有该问题的答案
    10. 查看mysql中是否有该问题答案
    11. 返回答案，同时写入redis缓存

    """

    def query(self, query, threshold=0.5, query_metadata=None):
        # 1. 判断用户的query是否合法，非空字符串
        if not query or type(query) is not str:
            logger.info("用户输入的query非法：{}".format(query))
            # TODO 1. 问题非法，不需要进入RAG模块
            return None, False

        query_metadata = query_metadata or extract_query_metadata(query)
        if should_bypass_faq(query, query_metadata):
            logger.info("具体公司、期间或报告查询绕过 FAQ: {}".format(query))
            return None, True

        # 2. 尝试去redis中查找，一模一样的问题
        answer = self.redis_client.get_answer(query)
        if answer:
            logger.info("在redis找到了一模一样的问题: {}".format(answer))
            # TODO 2. 在redis中找到了一模一样的问题对应的答案，不需要进入RAG模块
            return answer, False

        # A database exact match is a FAQ fast path even before its answer has
        # been populated in Redis; it must not depend on fuzzy BM25 confidence.
        answer = self.mysql_client.fetch_answer(query)
        if answer:
            self.redis_client.set_answer(query, answer)
            logger.info("在mysql找到了一模一样的问题: {}".format(query))
            return answer, False

        if self.bm25 is None or not self.original_questions:
            logger.info("Financial FAQ 尚无数据，继续进入后续问答流程")
            return None, True

        # 3. 对query进行预处理，得到分词数据
        query_tokens = preprocess_text(query)

        # 4. 通过bm25检索器得到query和每个问题的相似度分数：[n]
        scores = self.bm25.get_scores(query_tokens)

        # 5. 对相似度分数进行归一化
        scores_softmax = self._soft_max(scores)

        # 6. 取相似度分数归一化以后的最大值，判断是否大于给定的阈值；如果小于阈值，返回空答案
        max_index = np.argmax(scores_softmax)
        max_score = scores_softmax[max_index]
        logger.info("softmax后，匹配度最高的分数为：{}".format(max_score))
        # Definition-style FAQ paraphrases (for example, "ROE是什么？")
        # use a conservative fallback. All other requests keep the caller's
        # production threshold.
        effective_threshold = min(threshold, 0.40) if self._is_definition_query(query) else threshold
        # 7. 如果大于阈值，通过argmax找到最大分数对应的索引
        if max_score >= effective_threshold:
            # 8. 根据索引找到对应的原始问题（python内存）
            origin_question = self.original_questions[max_index]
            # 9. 查看redis缓存中是否有该问题的答案
            answer = self.redis_client.get_answer(origin_question)
            if answer:
                logger.info("在redis缓存中找到了问题: {} 的答案".format(query))
                # TODO 3. 在redis中找到了对应的答案，不需要进入RAG模块
                return answer, False
            # 10. 查看mysql中是否有该问题答案
            answer = self.mysql_client. fetch_answer(origin_question)
            if answer:
                logger.info("在mysql中找到了问题: {} 的答案".format(query))
                # 11. 返回答案，同时写入redis缓存
                self.redis_client.set_answer(origin_question, answer)
                # TODO 4. 在mysql中找到了对应的答案，不需要进入RAG模块
                return answer, False
            else:
                logger.error("在mysql未找到问题: {} 的答案".format(query))
                # TODO 5. mysql没有找到匹配度较高的高频问题（可能是数据一致性的问题）。但是问题合法，需要继续执行
                return None, True
        # TODO 6. 小于阈值，没有找到匹配度较高的高频问题。但是问题合法，需要继续执行
        return None, True

    @staticmethod
    def _is_definition_query(query):
        compact_query = ''.join(query.lower().split())
        return (
            compact_query.startswith(('什么是', '什么叫', '怎么理解'))
            or compact_query.endswith(('是什么?', '是什么？', '是什么意思?', '是什么意思？', '是干什么的?', '是干什么的？'))
        )

    def _soft_max(self, scores):
        # 1. 转为负数
        # 2. 缩小差距
        exp_scores = np.exp(scores - np.max(scores))
        return exp_scores / np.sum(exp_scores)


if __name__ == '__main__':
    bm25 = BM25Search(RedisClient(), MysqlClient())
    answer, _ = bm25.query('win10如何安装python')
    print(answer)
