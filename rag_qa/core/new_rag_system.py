# core/rag_system.py 源码
# RAGPrompts包含： 1. augment提示词，用于结合query和上下文生成答案；2. 假设答案、 子查询、回溯问题查询对应的提示词模板
from rag_qa.core.prompts import RAGPrompts
#   导入 time 模块，用于计算时间
import json
import time
from collections.abc import Mapping
from base.config import config
from base.logger import logger

from rag_qa.core.query_router import FinancialQueryRouter
from rag_qa.core.financial_calculator import (
    build_calculation_guardrail,
    build_calculation_note,
    document_mentions_metric,
)
from rag_qa.core.financial_evidence import (
    extract_verified_evidence,
    format_verified_evidence_block,
)
# 将专业问题进一步分类，做策略选择
from rag_qa.core.strategy_selector import StrategySelector  # 导入策略选择器


# from rag_qa.core.vector_store import VectorStore


#  定义 RAGSystem 类，封装 RAG 系统的核心逻辑
class RAGSystem:
    _MAX_DYNAMIC_CONTEXTS = 6
    _MAX_MULTI_EVIDENCE_CONTEXTS = 8
    _MAX_CONTEXT_CANDIDATE_POOL = 12
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
    def _parse_subqueries(subqueries_text, fallback_query):
        """只接受 SubQuery JSON 契约；任何其他输出安全回退到原查询。"""
        fallback = fallback_query.strip() if isinstance(fallback_query, str) else ""

        try:
            payload = json.loads(subqueries_text)
        except (TypeError, json.JSONDecodeError):
            logger.warning("SubQuery 输出不是合法 JSON，回退到原始查询")
            return [fallback] if fallback else []

        subqueries = payload.get("subqueries") if isinstance(payload, dict) else None
        if (
            not isinstance(subqueries, list)
            or not subqueries
            or any(not isinstance(subquery, str) or not subquery.strip() for subquery in subqueries)
        ):
            logger.warning("SubQuery JSON 缺少有效字符串数组 subqueries，回退到原始查询")
            return [fallback] if fallback else []

        return [subquery.strip() for subquery in subqueries]

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

    @classmethod
    def _context_limit(cls, query_metadata):
        """Keep simple requests at M; expand only explicit multi-target/metric requests."""
        if query_metadata is None:
            return config.CANDIDATE_M
        target_count = max(1, len(cls._target_bindings(query_metadata)))
        metric_count = max(1, len(query_metadata.requested_metrics))
        if target_count == 1 and metric_count == 1:
            return config.CANDIDATE_M
        required_cell_count = len(cls._required_evidence_cells(query_metadata))
        coverage_slots = max(target_count * metric_count, required_cell_count)
        if query_metadata.requires_deterministic_calculation() and target_count > 1:
            # Keep each target's top parent plus a metric-bearing parent for
            # every target, so a comparison is not starved of one base value.
            coverage_slots *= 2
        max_contexts = (
            cls._MAX_MULTI_EVIDENCE_CONTEXTS
            if required_cell_count > cls._MAX_DYNAMIC_CONTEXTS
            else cls._MAX_DYNAMIC_CONTEXTS
        )
        return min(max_contexts, max(config.CANDIDATE_M, coverage_slots))

    @classmethod
    def _context_candidate_pool_limit(cls, context_limit):
        """Bound the answer-layer selection pool without changing Milvus top-k."""
        return min(cls._MAX_CONTEXT_CANDIDATE_POOL, max(context_limit, context_limit * 2))

    @staticmethod
    def _target_bindings(query_metadata):
        if query_metadata is None:
            return ()
        bindings = []
        for target in query_metadata.subquery_targets:
            binding = (target.company_code, target.report_period)
            if (target.company_code or target.report_period) and binding not in bindings:
                bindings.append(binding)
        return tuple(bindings)

    @classmethod
    def _required_evidence_cells(cls, query_metadata):
        """Order cells by metric round so a capped budget still spans targets."""
        if query_metadata is None:
            return ()
        target_bindings = cls._target_bindings(query_metadata)
        metrics = query_metadata.requested_metrics
        if not target_bindings or not metrics:
            return ()
        return tuple(
            (company_code, report_period, metric)
            for metric in metrics
            for company_code, report_period in target_bindings
        )

    @staticmethod
    def _document_matches_evidence_cell(document, cell):
        company_code, report_period, metric = cell
        return (
            document.metadata.get("company_code") == company_code
            and document.metadata.get("report_period") == report_period
            and document_mentions_metric(document, metric)
        )

    @classmethod
    def _evidence_cell_status(cls, documents, query_metadata):
        """Return a serializable coverage view for diagnostics and evaluations."""
        return [
            {
                "company_code": company_code,
                "report_period": report_period,
                "metric": metric,
                "parent_id": next(
                    (
                        document.metadata.get("parent_id")
                        for document in documents
                        if cls._document_matches_evidence_cell(
                            document, (company_code, report_period, metric)
                        )
                    ),
                    None,
                ),
            }
            for company_code, report_period, metric in cls._required_evidence_cells(query_metadata)
        ]

    @classmethod
    def _select_context_docs(cls, ranked_docs, context_limit, query_metadata):
        """Preserve explicit target and metric coverage before score-order backfill."""
        target_bindings = cls._target_bindings(query_metadata)
        metrics = query_metadata.requested_metrics if query_metadata else ()
        if len(target_bindings) <= 1 and len(metrics) <= 1:
            return ranked_docs[:context_limit]

        selected = []
        selected_parent_ids = set()

        def score_and_order(item):
            index, document = item
            score = document.metadata.get("rerank_score")
            return (float(score) if score is not None else float("-inf"), -index)

        for cell in cls._required_evidence_cells(query_metadata):
            if any(cls._document_matches_evidence_cell(document, cell) for document in selected):
                continue
            candidates = [
                item for item in enumerate(ranked_docs)
                if item[1].metadata.get("parent_id") not in selected_parent_ids
                and cls._document_matches_evidence_cell(item[1], cell)
            ]
            if not candidates:
                continue
            _, document = max(candidates, key=score_and_order)
            selected.append(document)
            selected_parent_ids.add(document.metadata.get("parent_id"))
            if len(selected) >= context_limit:
                return selected

        for doc in ranked_docs:
            parent_id = doc.metadata.get("parent_id")
            if parent_id in selected_parent_ids:
                continue
            selected.append(doc)
            selected_parent_ids.add(parent_id)
            if len(selected) >= context_limit:
                break
        return selected

    """
    需求：实现假设文档嵌入（HyDE）检索策略
    思路步骤：
    1. 使用专用提示模板生成假设性答案
    2. 基于假设答案进行向量检索
    3. 基于假设答案查询父块作为上下文
    """

    #   定义私有方法，使用假设文档进行检索（HyDE）
    def _retrieve_with_hyde(
            self, query, source_filter=None, metadata_filter=None, result_limit=None
    ):
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
                hypo_answer,
                k=config.RETRIEVAL_K,
                source_filter=source_filter,
                metadata_filter=metadata_filter,
                result_limit=result_limit,
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
    def _retrieve_with_subqueries(
            self, query, source_filter=None, metadata_filter=None, subquery_filters=None,
            subquery_targets=None, result_limit=None,
    ):
        logger.info(f"使用子查询策略进行检索 (查询: '{query}')")
        #   获取子查询生成的 Prompt 模板
        subquery_prompt_template = RAGPrompts.subquery_prompt()  # 使用 template 后缀区分
        try:
            if subquery_targets is not None:
                if not isinstance(subquery_targets, (list, tuple)) or not subquery_targets:
                    raise ValueError("Deterministic subquery targets must be a non-empty sequence")
                subqueries = []
                target_filters = []
                for target in subquery_targets:
                    if not isinstance(target, Mapping) or set(target) != {"query", "metadata_filter"}:
                        raise ValueError(
                            "Each deterministic subquery target must contain query and metadata_filter"
                        )
                    target_query = target["query"]
                    target_filter = target["metadata_filter"]
                    if not isinstance(target_query, str) or not target_query.strip():
                        raise ValueError(
                            "Deterministic subquery target query must be a non-empty string"
                        )
                    if target_filter is not None and not isinstance(target_filter, Mapping):
                        raise TypeError(
                            "Deterministic subquery metadata_filter must be a mapping or None"
                        )
                    subqueries.append(target_query.strip())
                    target_filters.append(dict(target_filter) if target_filter is not None else None)
                if subquery_filters is not None:
                    raise ValueError(
                        "Do not combine deterministic subquery targets with subquery_filters"
                    )
                subquery_filters = target_filters
                optimization_seconds = 0.0
                logger.info("使用 %d 个确定性 SubQuery target", len(subqueries))
            else:
                #   调用大语言模型生成子查询列表
                optimization_started_at = time.perf_counter()
                subqueries_text = self._call_llm_for_text(
                    subquery_prompt_template.format(query=query)
                ).strip()
                optimization_seconds = time.perf_counter() - optimization_started_at
                subqueries = self._parse_subqueries(subqueries_text, query)
            logger.info(f"生成的子查询: {subqueries}")
            if not subqueries:
                logger.warning("未能生成有效的子查询")
                return []

            if subquery_filters is not None and (
                    not isinstance(subquery_filters, (list, tuple))
                    or len(subquery_filters) != len(subqueries)
            ):
                logger.warning(
                    "SubQuery metadata filters 与子查询数量不匹配，拒绝执行以避免期间错配"
                )
                return []

            if subquery_filters is None:
                per_subquery_filters = [metadata_filter] * len(subqueries)
            else:
                per_subquery_filters = []
                for subquery_filter in subquery_filters:
                    if metadata_filter is None:
                        per_subquery_filters.append(subquery_filter)
                        continue
                    if subquery_filter is None:
                        per_subquery_filters.append(metadata_filter)
                        continue
                    if not isinstance(metadata_filter, Mapping) or not isinstance(subquery_filter, Mapping):
                        raise TypeError("SubQuery metadata filters must be mappings or None")
                    merged_filter = dict(metadata_filter)
                    for field, value in subquery_filter.items():
                        if field in merged_filter and merged_filter[field] != value:
                            raise ValueError(
                                f"SubQuery metadata filter conflicts on field: {field}"
                            )
                        merged_filter[field] = value
                    per_subquery_filters.append(merged_filter)

            subquery_results, diagnostics = (
                self.vector_store.hybrid_search_subqueries_with_batched_embedding(
                    subqueries,
                    k=config.RETRIEVAL_K,
                    source_filter=source_filter,
                    metadata_filters=per_subquery_filters,
                    result_limit=result_limit,
                    return_diagnostics=True,
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
                subquery_results, result_limit or config.CANDIDATE_M
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
    def _retrieve_with_backtracking(
            self, query, source_filter=None, metadata_filter=None, result_limit=None
    ):
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
                simplified_query,
                k=config.RETRIEVAL_K,
                source_filter=source_filter,
                metadata_filter=metadata_filter,
                result_limit=result_limit,
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
            self,
            query,
            source_filter=None,
            strategy=None,
            strategy_selection_seconds=None,
            metadata_filter=None,
            subquery_filters=None,
            subquery_targets=None,
            query_metadata=None,
    ):
        retrieval_started_at = time.perf_counter()
        # 如果未指定检索策略，则使用策略选择器选择
        if not strategy:
            strategy_started_at = time.perf_counter()
            strategy = self.strategy_selector.select_strategy(query)
            strategy_selection_seconds = time.perf_counter() - strategy_started_at
        elif strategy_selection_seconds is None:
            strategy_selection_seconds = 0.0

        context_limit = self._context_limit(query_metadata)
        candidate_pool_limit = self._context_candidate_pool_limit(context_limit)

        # 根据检索策略选择不同的检索方式
        ranked_parent_chunks = []  # 初始化
        if strategy == "回溯问题检索":
            ranked_parent_chunks = self._retrieve_with_backtracking(
                query, source_filter, metadata_filter, candidate_pool_limit
            )
        elif strategy == "子查询检索":
            ranked_parent_chunks = self._retrieve_with_subqueries(
                query, source_filter, metadata_filter, subquery_filters, subquery_targets,
                candidate_pool_limit,
            )
        elif strategy == "假设问题检索":
            ranked_parent_chunks = self._retrieve_with_hyde(
                query, source_filter, metadata_filter, candidate_pool_limit
            )
        else:  # 默认或“直接检索”
            logger.info(f"使用直接检索策略 (查询: '{query}')")
            ranked_parent_chunks = self.vector_store.hybrid_search_with_rerank(
                query,
                k=config.RETRIEVAL_K,
                source_filter=source_filter,
                metadata_filter=metadata_filter,
                result_limit=candidate_pool_limit,
            )  # 注意 hybrid_search_with_rerank 返回的是 rerank 后的父文档

        logger.info(f"策略 '{strategy}' 检索到 {len(ranked_parent_chunks)} 个候选文档 (可能已是父文档)")

        final_context_docs = self._select_context_docs(
            ranked_parent_chunks, context_limit, query_metadata
        )

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
    def generate_answer(
            self, query, history=None, source_filter=None, metadata_filter=None, subquery_filters=None,
            subquery_targets=None, strategy=None, query_metadata=None,
    ):
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
        if strategy is None:
            strategy_started_at = time.perf_counter()
            strategy = self.strategy_selector.select_strategy(query)
            strategy_selection_seconds = time.perf_counter() - strategy_started_at
        else:
            strategy_selection_seconds = 0.0

        #   检索相关文档
        # list[Document]
        context_docs = self.retrieve_and_merge(
            query,
            source_filter=source_filter,
            metadata_filter=metadata_filter,
            subquery_filters=subquery_filters,
            subquery_targets=subquery_targets,
            strategy=strategy,
            strategy_selection_seconds=strategy_selection_seconds,
            query_metadata=query_metadata,
        )  # 传递 strategy

        #   准备上下文
        if context_docs:
            context = "\n\n".join([doc.page_content for doc in context_docs])  # 使用换行符分隔文档
            verified_evidence = extract_verified_evidence(
                context_docs,
                query_metadata.requested_metrics if query_metadata else None,
            )
            evidence_block = format_verified_evidence_block(verified_evidence)
            calculation_note = build_calculation_note(query_metadata, verified_evidence)
            calculation_guardrail = build_calculation_guardrail(
                query_metadata, verified_evidence
            )
            if evidence_block:
                context = f"{context}\n\n{evidence_block}"
                logger.info("已添加 %d 条已验证财务数值", len(verified_evidence))
            if calculation_note:
                context = f"{context}\n\n{calculation_note}"
                logger.info("已添加基于证据的确定性计算结果")
            if calculation_guardrail:
                context = f"{context}\n\n{calculation_guardrail}"
                logger.info("已添加未验证计算 Guardrail")
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
