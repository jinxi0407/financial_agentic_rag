# core/rag_system.py 源码
# RAGPrompts包含： 1. augment提示词，用于结合query和上下文生成答案；2. 假设答案、 子查询、回溯问题查询对应的提示词模板
from rag_qa.core.prompts import RAGPrompts
#   导入 time 模块，用于计算时间
import time
from base.config import config
from base.logger import logger

from rag_qa.core.query_router import FinancialQueryRouter
# 将专业问题进一步分类，做策略选择
from rag_qa.core.strategy_selector import StrategySelector  # 导入策略选择器


# from rag_qa.core.vector_store import VectorStore


#  定义 RAGSystem 类，封装 RAG 系统的核心逻辑
class RAGSystem:
    #   初始化方法，设置 RAG 系统的基本参数
    # vector_store：milvus的查询相关的client
    # llm: 调用大模型的client
    def __init__(self, vector_store, llm, query_router):
        #   设置向量数据库对象
        self.vector_store = vector_store
        #   设置大语言模型调用函数
        self.llm = llm
        #   获取 RAG 提示模板
        self.rag_prompt = RAGPrompts.rag_prompt()
        self.no_context_response = RAGPrompts.INSUFFICIENT_CONTEXT_RESPONSE
        # 第一层路由器判断查询是否应进入 Financial RAG。
        self.query_router = query_router
        #   初始化策略选择器
        self.strategy_selector = StrategySelector()

    def _call_llm_for_text(self, prompt):
        """将同步字符串或流式生成器统一收集为策略所需的文本。"""
        result = self.llm(prompt)
        if isinstance(result, str):
            return result
        return "".join(chunk for chunk in result if chunk)

    @staticmethod
    def _merge_subquery_results_coverage_first(subquery_results, limit):
        """按子查询轮转选择各自最高的未重复 Parent，优先保留证据覆盖。"""
        selected_docs = []
        selected_parent_ids = set()
        positions = [0] * len(subquery_results)

        while len(selected_docs) < limit:
            selected_in_round = False
            for index, docs in enumerate(subquery_results):
                while positions[index] < len(docs):
                    candidate = docs[positions[index]]
                    positions[index] += 1
                    parent_id = candidate.metadata.get("parent_id")
                    if parent_id in selected_parent_ids:
                        continue

                    selected_docs.append(candidate)
                    selected_parent_ids.add(parent_id)
                    selected_in_round = True
                    break

                if len(selected_docs) == limit:
                    break

            if not selected_in_round:
                break

        return selected_docs

    """
    需求：实现假设文档嵌入（HyDE）检索策略
    思路步骤：
    1. 使用专用提示模板生成假设性答案
    2. 基于假设答案进行向量检索
    3. 基于假设答案查询父块作为上下文
    """

    #   定义私有方法，使用假设文档进行检索（HyDE）
    def _retrieve_with_hyde(self, query):
        logger.info(f"使用 HyDE 策略进行检索 (查询: '{query}')")
        #   获取假设问题生成的 Prompt 模板
        # TODO 1. 获取假设检索对应提示词模板
        hyde_prompt_template = RAGPrompts.hyde_prompt()  # 使用 template 后缀区分
        #   调用大语言模型生成假设答案
        try:
            # TODO 2. 基于传入的query，构造假设答案生成的提示词
            # TODO 3. 调用大模型生成假设答案
            hypo_answer = self._call_llm_for_text(
                hyde_prompt_template.format(query=query)
            ).strip()
            logger.info(f"HyDE 生成的假设答案: '{hypo_answer}'")
            #   使用假设答案进行检索，并返回检索结果
            #   注意：HyDE 通常只用于生成检索向量，不一定需要 rerank 这一步，但这里复用了
            # TODO 4. 基于假设答案查询父块作为上下文
            return self.vector_store.hybrid_search_with_rerank(
                # TODO 瞪大眼睛注意，这里传入的是hypo_answer而不是原始的query
                hypo_answer, k=config.RETRIEVAL_K  # 使用 K 而非 M
            )
        except Exception as e:
            logger.error(f"HyDE 策略执行失败: {e}")
            return []

    """
    需求：实现子查询检索策略
    思路步骤：
    1. 将复杂查询拆解为多个子查询
    2. 依次执行各子查询的混合检索
    3. 合并所有子查询的检索结果
    4. 以 Parent ID 去重并优先保留各子查询的证据覆盖
    """

    #   定义私有方法，使用子查询进行检索
    def _retrieve_with_subqueries(self, query):
        logger.info(f"使用子查询策略进行检索 (查询: '{query}')")
        #   获取子查询生成的 Prompt 模板
        subquery_prompt_template = RAGPrompts.subquery_prompt()  # 使用 template 后缀区分
        try:
            #   调用大语言模型生成子查询列表
            optimization_started_at = time.perf_counter()
            subqueries_text = self._call_llm_for_text(
                subquery_prompt_template.format(query=query)
            ).strip()
            optimization_seconds = time.perf_counter() - optimization_started_at
            subqueries = [q.strip() for q in subqueries_text.split("\n") if q.strip()]
            logger.info(f"生成的子查询: {subqueries}")
            if not subqueries:
                logger.warning("未能生成有效的子查询")
                return []

            subquery_results, diagnostics = (
                self.vector_store.hybrid_search_subqueries_with_batched_embedding(
                    subqueries, k=config.RETRIEVAL_K, return_diagnostics=True
                )
            )
            timing = diagnostics["timing"]
            for index, subquery_timing in enumerate(diagnostics["per_subquery"], start=1):
                logger.info(
                    "子查询 %d/%d Hybrid 阶段耗时: hybrid_search=%.3fs, "
                    "parent_dedup=%.3fs, parents=%d (查询: '%s')",
                    index,
                    len(subqueries),
                    subquery_timing["hybrid_search_seconds"],
                    subquery_timing["parent_dedup_seconds"],
                    subquery_timing["parent_count"],
                    subquery_timing["query"],
                )

            aggregation_started_at = time.perf_counter()
            final_docs = self._merge_subquery_results_coverage_first(
                subquery_results, config.CANDIDATE_M
            )
            aggregation_seconds = time.perf_counter() - aggregation_started_at
            logger.info(
                "子查询阶段耗时: optimization=%.3fs, embedding=%.3fs, "
                "hybrid_search=%.3fs, parent_dedup=%.3fs, reranker=%.3fs, "
                "aggregation=%.3fs, predict_calls=%d, subqueries=%d, final_contexts=%d",
                optimization_seconds,
                timing["embedding_seconds"],
                timing["hybrid_search_seconds"],
                timing["parent_dedup_seconds"],
                timing["reranker_seconds"],
                aggregation_seconds,
                diagnostics["reranker_predict_calls"],
                len(subqueries),
                len(final_docs),
            )
            return final_docs

        except Exception as e:
            logger.error(f"子查询策略执行失败: {e}")
            return []

    """
    需求：实现回溯问题检索策略
    思路步骤：
    1. 将复杂查询转化为基础问题
    2. 使用简化后的问题进行混合检索
    3. 返回重排序后的相关文档
    """

    #   定义私有方法，使用回溯问题进行检索
    def _retrieve_with_backtracking(self, query):
        logger.info(f"使用回溯问题策略进行检索 (查询: '{query}')")
        #   获取回溯问题生成的 Prompt 模板
        backtrack_prompt_template = RAGPrompts.backtracking_prompt()  # 使用 template 后缀区分
        try:
            #   调用大语言模型生成回溯问题
            simplified_query = self._call_llm_for_text(
                backtrack_prompt_template.format(query=query)
            ).strip()
            logger.info(f"生成的回溯问题: '{simplified_query}'")
            #   使用回溯问题进行检索，并返回检索结果
            return self.vector_store.hybrid_search_with_rerank(
                simplified_query, k=config.RETRIEVAL_K  # 使用 K
            )
        except Exception as e:
            logger.error(f"回溯问题策略执行失败: {e}")
            return []

    """
    需求：动态选择检索策略并整合结果
    思路步骤：
    1. 未指定策略时通过策略选择器决策
    2. 根据策略类型路由到对应检索方法
    3. 限制最终上下文文档数量（CANDIDATE_M）
    """

    # 定义方法，检索并合并相关文档
    def retrieve_and_merge(
            self, query, source_filter=None, strategy=None, strategy_selection_seconds=None
    ):
        retrieval_started_at = time.perf_counter()
        # 如果未指定检索策略，则使用策略选择器选择
        if not strategy:
            strategy_started_at = time.perf_counter()
            strategy = self.strategy_selector.select_strategy(query)
            strategy_selection_seconds = time.perf_counter() - strategy_started_at
        elif strategy_selection_seconds is None:
            strategy_selection_seconds = 0.0

        # 根据检索策略选择不同的检索方式
        ranked_parent_chunks = []  # 初始化
        if strategy == "回溯问题检索":
            ranked_parent_chunks = self._retrieve_with_backtracking(query)
        elif strategy == "子查询检索":
            ranked_parent_chunks = self._retrieve_with_subqueries(query)
        elif strategy == "假设问题检索":
            ranked_parent_chunks = self._retrieve_with_hyde(query)
        else:  # 默认或“直接检索”
            logger.info(f"使用直接检索策略 (查询: '{query}')")
            ranked_parent_chunks = self.vector_store.hybrid_search_with_rerank(
                query, k=config.RETRIEVAL_K, source_filter=source_filter
            )  # 注意 hybrid_search_with_rerank 返回的是 rerank 后的父文档

        logger.info(f"策略 '{strategy}' 检索到 {len(ranked_parent_chunks)} 个候选文档 (可能已是父文档)")

        final_context_docs = ranked_parent_chunks[:config.CANDIDATE_M]

        logger.info(f"最终选取 {len(final_context_docs)} 个文档作为上下文")
        logger.info(
            "检索全流程耗时: strategy_selector=%.3fs, total=%.3fs, strategy='%s'",
            strategy_selection_seconds,
            time.perf_counter() - retrieval_started_at,
            strategy,
        )
        return final_context_docs

    """
    需求：端到端处理用户查询并生成答案
    思路步骤：
    1. 使用 Financial Query Router 判断是否进入金融知识库检索
    2. OUT_OF_SCOPE：返回范围外提示，不执行检索
    3. RAG：
      3.1 选择最佳检索策略
      3.2 检索合并相关文档
      3.3 构建格式化上下文
      3.4 组合提示模板调用 LLM
    """

    # 定义方法，生成答案
    def generate_answer(self, query, history=None, source_filter=None):
        # 记录查询开始时间
        start_time = time.time()
        logger.info(f"开始处理查询: '{query}', 知识库过滤: {source_filter}")

        # 第一层路由只判断是否进入 Financial RAG，不负责回答问题。
        route = self.query_router.route(query)
        logger.info(f"查询路由结果：{route} (查询: '{query}')")

        if route == FinancialQueryRouter.OUT_OF_SCOPE:
            logger.info("查询超出 Financial RAG 范围，不执行检索")
            processing_time = time.time() - start_time
            logger.info(
                f"范围外查询处理完成 (耗时: {processing_time:.2f}s, 查询: '{query}')"
            )
            return self.no_context_response

        # RAG 路由或保守回退后，继续原有策略选择和检索流程。
        logger.info("查询进入 Financial RAG，执行检索流程")
        #   选择检索策略
        strategy_started_at = time.perf_counter()
        strategy = self.strategy_selector.select_strategy(query)
        strategy_selection_seconds = time.perf_counter() - strategy_started_at

        #   检索相关文档
        # list[Document]
        context_docs = self.retrieve_and_merge(
            query,
            source_filter=source_filter,
            strategy=strategy,
            strategy_selection_seconds=strategy_selection_seconds,
        )  # 传递 strategy

        #   准备上下文
        if context_docs:
            context = "\n\n".join([doc.page_content for doc in context_docs])  # 使用换行符分隔文档
            logger.info(f"构建上下文完成，包含 {len(context_docs)} 个文档块")
            # logger.debug(f"上下文内容:\n{context[:500]}...") # Debug 日志可以打印部分上下文
        else:
            logger.info("未检索到相关文档，上下文为空")
            return self.no_context_response

        # 准备历史对话
        if history:
            history_str = '\n\n'.join([f'human:{row["question"]} ; ai:{row["answer"]}' for row in history])
        else:
            history_str = ''

        #   构造 Prompt，调用大语言模型生成答案
        prompt_input = self.rag_prompt.format(
            context=context, question=query, history=history_str
        )
        # logger.debug(f"最终生成的 Prompt:\n{prompt_input}") # Debug 日志

        try:
            answer = self.llm(prompt_input)
        except Exception as e:
            logger.error(f"调用 LLM 生成最终答案失败: {e}")
            answer = "模型服务暂时不可用，无法完成回答。"

        #   记录查询处理完成的日志
        processing_time = time.time() - start_time
        logger.info(f"查询处理完成 (耗时: {processing_time:.2f}s, 查询: '{query}')")
        return answer
