# 导入 BGE-M3 嵌入函数，用于生成文档和查询的向量表示
from milvus_model.hybrid import BGEM3EmbeddingFunction
# 导入 Milvus 相关类，用于操作向量数据库
from pymilvus import MilvusClient, DataType, AnnSearchRequest, WeightedRanker
# 导入 Document 类，用于创建文档对象
from langchain_core.documents import Document
# 导入 CrossEncoder，用于重排序和 NLI 判断
from sentence_transformers import CrossEncoder
# 导入 hashlib 模块，用于生成唯一 ID 的哈希值
import hashlib
# 导入 time 模块，用于生成时间戳
import time
from base.config import config
from base.logger import logger
import sys
import os
import torch
from collections.abc import Mapping

# core/vector_store.py
# 定义 VectorStore 类，封装向量存储和检索功能

# local_path = os.path.abspath(os.path.dirname(__file__))
# rag_qa_path = os.path.abspath(os.path.dirname(local_path))
# sys.path.insert(0, rag_qa_path)
# project_root = os.path.dirname(rag_qa_path)
# sys.path.insert(0, project_root)

"""
需求：初始化VectorStore
思路步骤：
1. 构造milvus连接相关的参数: host、port、db、collection_name
2. 构造嵌入模型
3. 构造rerank模型
4. 创建或加载milvus集合，保证milvus表存在

"""


class VectorStore:
    SUBQUERY_RERANK_BATCH_SIZE = 8

    IDENTITY_FIELDS = {
        "document_id",
        "file_sha256",
        "source_filename",
    }
    REPORT_METADATA_FIELDS = (
        "company_name",
        "company_code",
        "report_year",
        "period_type",
        "report_period",
    )
    INGESTION_SCHEMA_FIELDS = IDENTITY_FIELDS | set(REPORT_METADATA_FIELDS)
    METADATA_FILTER_FIELDS = (
        "source",
        "company_name",
        "company_code",
        "report_year",
        "period_type",
        "report_period",
        "document_id",
    )

    # 初始化方法，设置向量存储的基本参数
    def __init__(self,
                 collection_name=config.MILVUS_COLLECTION_NAME,
                 host=config.MILVUS_HOST,
                 port=config.MILVUS_PORT,
                 database=config.MILVUS_DATABASE_NAME):
        # 1. 构造milvus连接相关的参数: host、port、db、collection_name
        # 设置 Milvus 集合名称
        self.collection_name = collection_name
        # 设置 Milvus 主机地址
        self.host = host
        # 设置 Milvus 端口号
        self.port = port
        # 设置 Milvus 数据库名称
        self.database = database
        # 初始化 Milvus 客户端，连接到指定主机和数据库
        # TODO 生产环境需要输入用户名和密码
        self.client = MilvusClient(uri=f"http://{self.host}:{self.port}", db_name=self.database)
        # 设置日志记录器
        self.logger = logger

        # 调用方法创建或加载 Milvus 集合
        self._create_or_load_collection()

        # 2. 构造嵌入模型
        bge_m3_model_path = config.BGE_M3_MODEL_PATH

        # 构造bge-m3模型，通过路径加载的方式把模型加载到内存中
        self.embedding_function = BGEM3EmbeddingFunction(
            # 传入模型名: 自动下载； 传入路径:直接加载
            model_name_or_path=bge_m3_model_path
            # f16: 半精度浮点数 这里取false相当于使用fp32
            # TODO 是否开启半精度
            , use_f16=False,
            # 推理（预测）设备
            # TODO 专业卡：A100, H100, A800. 消费卡：RTX4090, RTX5090
            device='cuda' if torch.cuda.is_available() else 'cpu'
        )



        # 3. 构造rerank模型
        # 4. 创建或加载milvus集合，保证milvus表存在

        rerank_model_path = config.RERANKER_MODEL_PATH
        # 初始化 BGE-Reranker 模型，用于重排序检索结果
        # TODO device代表设备： mps:m1系列的mac/ cpu: cpu / cuda: nvidia的gpu。和操作系统无关
        self.reranker = CrossEncoder(rerank_model_path
                                     , device='cuda' if torch.cuda.is_available() else 'cpu')


    """
    需求：实现创建或加载集合方法：检查并创建或加载Milvus集合，定义字段结构和索引参数
    思路步骤：
    1. 判断集合是否存在，若存在则进行加载
    2. 集合不存在，创建新集合
        2.1 定义集合各字段 id、text、dense_vector、sparse_vector、parent_id、 parent_content、source 、timestamp
        2.2 定义索引： 稠密向量索引：dense_vector ， 稀疏向量索引：sparse_vector
    3. 把集合的索引加载到内存中
    """

    # 定义私有方法，创建或加载 Milvus 集合
    def _create_or_load_collection(self):
        # 检查指定集合是否已存在
        # if self.collection_name not in client.list_collections()
        if not self.client.has_collection(self.collection_name):
            # 创建集合 Schema，禁用自动 ID，启用动态字段
            # TODO 主键不使用自增 开启动态字段
            schema = self.client.create_schema(auto_id=False, enable_dynamic_field=True)
            # 添加 ID 字段，作为主键，VARCHAR 类型，最大长度 100
            # TODO id 无序，基于子块的内容生成唯一的id。  (文本1) -> md5 -> (05ff)
            schema.add_field(field_name="id", datatype=DataType.VARCHAR, is_primary=True, max_length=100)
            # 添加子块文本字段，VARCHAR 类型，最大长度 65535
            schema.add_field(field_name="text", datatype=DataType.VARCHAR, max_length=65535)
            # 添加稠密向量字段，FLOAT_VECTOR 类型，维度由嵌入函数指定
            schema.add_field(field_name="dense_vector", datatype=DataType.FLOAT_VECTOR, dim=1024)
            # 添加稀疏向量字段，SPARSE_FLOAT_VECTOR 类型
            # TODO 稀疏向量由bge-m3模型生成，和bm25没有任何关系
            schema.add_field(field_name="sparse_vector", datatype=DataType.SPARSE_FLOAT_VECTOR)
            # 添加父块 ID 字段，VARCHAR 类型，最大长度 100
            schema.add_field(field_name="parent_id", datatype=DataType.VARCHAR, max_length=100)
            # 添加父块内容字段，VARCHAR 类型，最大长度 65535
            # TODO 父块内容
            schema.add_field(field_name="parent_content", datatype=DataType.VARCHAR, max_length=65535)
            # 添加知识库来源字段，VARCHAR 类型，最大长度 50
            # TODO 知识库来源，用于过滤
            schema.add_field(field_name="source", datatype=DataType.VARCHAR, max_length=50)
            # 添加时间戳字段，VARCHAR 类型，最大长度 50
            schema.add_field(field_name="timestamp", datatype=DataType.VARCHAR, max_length=50)
            # File-level provenance makes incremental ingestion deterministic and auditable.
            schema.add_field(field_name="document_id", datatype=DataType.VARCHAR, max_length=64)
            schema.add_field(field_name="file_sha256", datatype=DataType.VARCHAR, max_length=64)
            schema.add_field(field_name="source_filename", datatype=DataType.VARCHAR, max_length=512)
            schema.add_field(field_name="company_name", datatype=DataType.VARCHAR, max_length=256, nullable=True)
            schema.add_field(field_name="company_code", datatype=DataType.VARCHAR, max_length=32, nullable=True)
            schema.add_field(field_name="report_year", datatype=DataType.INT64, nullable=True)
            schema.add_field(field_name="period_type", datatype=DataType.VARCHAR, max_length=16, nullable=True)
            schema.add_field(field_name="report_period", datatype=DataType.VARCHAR, max_length=32, nullable=True)

            # 创建索引参数对象
            index_params = self.client.prepare_index_params()
            # 为稠密向量字段添加 IVF_FLAT 索引，度量类型为内积 (IP)
            index_params.add_index(
                field_name="dense_vector",
                index_name="dense_index",
                index_type="IVF_FLAT",
                metric_type="IP",
                params={"nlist": 128}
            )
            # 为稀疏向量字段添加 SPARSE_INVERTED_INDEX 索引，度量类型为内积 (IP)
            index_params.add_index(
                field_name="sparse_vector",
                index_name="sparse_index",
                # 稀疏向量最常用的索引类型： 倒排索引（全文检索索引）
                index_type="SPARSE_INVERTED_INDEX",
                metric_type="IP",
                params={"drop_ratio_build": 0.2}
            )

            # 创建 Milvus 集合，应用定义的 Schema 和索引参数
            self.client.create_collection(collection_name=self.collection_name, schema=schema,
                                          index_params=index_params)
            self.collection_fields = {
                "id", "text", "dense_vector", "sparse_vector", "parent_id",
                "parent_content", "source", "timestamp", *self.IDENTITY_FIELDS,
                *self.REPORT_METADATA_FIELDS,
            }
            # 记录创建集合的日志
            logger.info(f"已创建集合 {self.collection_name}")
        # 如果集合已存在
        else:
            # 记录加载集合的日志
            logger.info(f"已加载集合 {self.collection_name}")
            description = self.client.describe_collection(self.collection_name)
            self.collection_fields = {
                field["name"] for field in description.get("fields", [])
            }

        self.identity_schema_ready = self.INGESTION_SCHEMA_FIELDS.issubset(self.collection_fields)
        if not self.identity_schema_ready:
            missing = ", ".join(sorted(self.INGESTION_SCHEMA_FIELDS - self.collection_fields))
            logger.warning(
                "集合 %s 缺少入库 metadata 字段 (%s)；查询仍可用，但为避免新旧 schema 混写，入库已禁用。",
                self.collection_name,
                missing,
            )

        self.retrieval_output_fields = [
            "text", "parent_id", "parent_content", "source", "timestamp",
        ]
        self.retrieval_output_fields.extend(
            field for field in sorted(self.IDENTITY_FIELDS)
            if field in self.collection_fields
        )
        self.retrieval_output_fields.extend(
            field for field in self.REPORT_METADATA_FIELDS
            if field in self.collection_fields
        )

        # 将集合加载到内存，确保可立即查询
        self.client.load_collection(self.collection_name)

    def ensure_identity_schema_for_ingestion(self):
        """Prevent mixed legacy/new ingestion until the collection is rebuilt once."""
        if self.identity_schema_ready:
            return
        missing = ", ".join(sorted(self.INGESTION_SCHEMA_FIELDS - self.collection_fields))
        raise RuntimeError(
            f"Milvus collection '{self.collection_name}' uses the legacy schema and is "
            f"missing: {missing}. Rebuild this Financial collection before ingestion."
        )

    def is_document_ingested(self, document_id):
        """Return whether a source file's SHA-256 identity already exists in Milvus."""
        self.ensure_identity_schema_for_ingestion()
        result = self.client.query(
            collection_name=self.collection_name,
            filter=f'document_id == "{document_id}"',
            output_fields=["document_id"],
            limit=1,
        )
        return bool(result)

    """
      需求：将分块后的文档转换为向量并存储到Milvus集合
      思路步骤：
      1. 提取文本,从文档对象中提取文本内容
      2. 生成向量,使用BGE-M3模型生成稠密和稀疏向量。
      3. 构造数据，为每篇文档生成唯一ID（MD5哈希）。将向量和元数据组织成字典
      4. 使用upsert操作插入或更新数据
      5. 支持批量处理，避免内存溢出
      """

    # 定义方法，向向量存储添加文档
    def add_documents(self, documents, batch_size=1000):
        """
        :param documents: 分块以后的子块内容 list[Document]
            Document(page_content=正文, metadata={父id、子id、父块内容parent_content、时间戳、路径、知识库来源(source)})
        :param batch_size: 批处理大小，默认1000，可根据内存情况调整
        :return:  None 无返回值
        """
        self.ensure_identity_schema_for_ingestion()
        total_docs = len(documents)
        logger.info(f"开始处理 {total_docs} 个文档，批次大小: {batch_size}")

        # 分批处理文档
        for start_idx in range(0, total_docs, batch_size):
            end_idx = min(start_idx + batch_size, total_docs)
            batch_docs = documents[start_idx:end_idx]

            for child_chunk in batch_docs:
                required_fields = (
                    "document_id", "file_sha256", "source_filename", "parent_id",
                    "parent_content", "milvus_id",
                )
                missing = [field for field in required_fields if not child_chunk.metadata.get(field)]
                if missing:
                    raise ValueError(f"Child chunk is missing required metadata: {missing}")

            logger.info(f"正在处理批次 {start_idx // batch_size + 1}/{(total_docs + batch_size - 1) // batch_size}, "
                        f"文档范围: {start_idx}-{end_idx - 1}")

            # 提取当前批次文档的内容列表
            texts = [child_chunk.page_content for child_chunk in batch_docs]

            # 使用 BGE-M3 嵌入函数生成文档的嵌入 ，传入list[str]
            # TODO 对于不同的嵌入模型，因为作者的设计是不一样的，没法统一。对于不同的模型，需要自己去了解对应的数据格式
            # embeddings : 字典类型。 sparse/dense
            # 假设使用text-embedding-v4（阿里） -> [n, 1024] 矩阵, 只有稠密向量

            # TODO：稀疏向量， 关键词对应的词频向量
            # embeddings['sparse'][i].indices = 第i个文档的所有的关键词 返回个一个
            # embeddings['sparse'][i].data = 第i个文档所有的关键词对应的权重

            # TODO：稠密向量， 句子转成的语义向量
            # embeddings['dense'] ->  二维矩阵 [n , 1024] n个子块对应的句子向量，一个句子 -> 1024维的向量

            try:
                embeddings = self.embedding_function(texts)

                # 初始化空列表，用于存储插入的数据
                data = []
                # 遍历当前批次的每个文档，带上索引 i
                for i, child_chunk in enumerate(batch_docs):
                    # The ID incorporates document identity and chunk position, not text alone.
                    milvus_id = child_chunk.metadata["milvus_id"]

                    # 获取第 i 行的稀疏向量数据
                    sparse_row = embeddings['sparse'][i]
                    # 初始化稀疏向量字典
                    sparse_vector = {}
                    # TODO 示例，实际计算的时候原理类似，但是会有一些算法做了优化
                    # 文本1 -> (单词1=888，单词2=9，单词3=1001 ) 词表大小=2万。 -> {888:1, 9:1, 1001:1} -> 归一化 {888:0.33,9:0.33,1001:0.33}
                    # 获取稀疏向量的非零值索引
                    # col: [888,9,1001]
                    indices = sparse_row.col
                    # 获取稀疏向量的非零值
                    # data: [0.33,0.33,0.33]
                    values = sparse_row.data
                    # 将索引和值配对，填充稀疏向量字典
                    for idx, value in zip(indices, values):
                        sparse_vector[int(idx)] = float(value)

                    # 创建数据字典，包含所有字段
                    data.append({
                        # 通过hash算法生成的唯一ID
                        "id": milvus_id,
                        # 子块的原文
                        "text": child_chunk.page_content,
                        # 1024维的稠密向量
                        "dense_vector": embeddings["dense"][i].tolist(),  # 转换为list以确保兼容性
                        # dict类型的稀疏向量
                        "sparse_vector": sparse_vector,
                        # 父块ID
                        "parent_id": child_chunk.metadata["parent_id"],
                        # 父块内容
                        "parent_content": child_chunk.metadata["parent_content"],
                        # 知识库来源
                        "source": child_chunk.metadata.get("source", "unknown"),
                        # 时间戳
                        "timestamp": child_chunk.metadata.get("timestamp", "unknown"),
                        "document_id": child_chunk.metadata["document_id"],
                        "file_sha256": child_chunk.metadata["file_sha256"],
                        "source_filename": child_chunk.metadata["source_filename"],
                        "company_name": child_chunk.metadata.get("company_name"),
                        "company_code": child_chunk.metadata.get("company_code"),
                        "report_year": child_chunk.metadata.get("report_year"),
                        "period_type": child_chunk.metadata.get("period_type"),
                        "report_period": child_chunk.metadata.get("report_period"),
                    })

                # 检查是否有数据需要插入
                if data:
                    # 使用 upsert 操作插入数据，覆盖重复 ID
                    self.client.upsert(collection_name=self.collection_name, data=data)
                    # 记录插入或更新的文档数量日志
                    logger.info(f"批次完成: 已插入或更新 {len(data)} 个文档 (批次 {start_idx // batch_size + 1})")
                else:
                    logger.warning(f"批次 {start_idx // batch_size + 1} 中没有数据需要插入")

            except Exception as e:
                logger.error(f"处理批次 {start_idx // batch_size + 1} 时发生错误: {str(e)}")
                # 继续处理下一个批次而不是中断整个过程
                continue

        # Make all completed upserts visible to the file-level duplicate check immediately.
        self.client.flush(collection_name=self.collection_name)
        logger.info(f"完成处理并持久化 {total_docs} 个文档")

    """
    需求：对输入的query进行混合检索
    思路步骤：
    1. 生成查询向量：使用BGE-M3生成稠密和稀疏向量。
    2. 构造检索请求(混合检索)：
        2.1 构造稠密向量的AnnSearchRequest(列名、limit、查询参数、查询向量、过滤条件)
        2.2 构造稀疏向量的AnnSearchRequest(列名、limit、查询参数、查询向量、过滤条件)
    3. 混合检索： 使用WeightedRanker融合结果，调用hybrid_search()
    4. 重排序（精排），使用CrossEncoder:reranker重新排序父文档
    """

    @staticmethod
    def _sort_scored_documents(scores, parent_docs):
        """按 reranker 分数稳定降序排列，避免同分时比较 Document 对象。"""
        return sorted(zip(scores, parent_docs), key=lambda item: item[0], reverse=True)

    @staticmethod
    def _documents_from_ranked_pairs(ranked_pairs, result_limit):
        """Keep existing ordering while exposing the already-computed rerank score."""
        documents = []
        for score, document in ranked_pairs[:result_limit]:
            metadata = dict(document.metadata)
            metadata["rerank_score"] = float(score) if score is not None else None
            documents.append(Document(page_content=document.page_content, metadata=metadata))
        return documents

    def _embed_queries(self, queries):
        """按输入顺序批量生成 BGE-M3 稠密和稀疏向量。"""
        return self.embedding_function([str(query) for query in queries])

    @staticmethod
    def _query_vectors_from_embeddings(query_embeddings, index):
        dense_query_vector = query_embeddings["dense"][index]
        sparse_row = query_embeddings["sparse"][index]
        sparse_query_vector = {
            token_id: value for token_id, value in zip(sparse_row.col, sparse_row.data)
        }
        return dense_query_vector, sparse_query_vector

    @staticmethod
    def _restore_subquery_rankings(parent_docs_by_subquery, scores):
        """将统一 batch 的分数恢复到每个子查询原有的候选列表中。"""
        score_index = 0
        ranked_pairs_by_subquery = []
        for parent_docs in parent_docs_by_subquery:
            if not parent_docs:
                ranked_pairs_by_subquery.append([])
            elif len(parent_docs) == 1:
                ranked_pairs_by_subquery.append([(None, parent_docs[0])])
            else:
                current_scores = scores[score_index:score_index + len(parent_docs)]
                score_index += len(parent_docs)
                ranked_pairs_by_subquery.append(
                    VectorStore._sort_scored_documents(current_scores, parent_docs)
                )
        if score_index != len(scores):
            raise ValueError("Batch reranker score count does not match parent candidates")
        return ranked_pairs_by_subquery

    @classmethod
    def build_metadata_filter_expression(cls, metadata_filter=None, source_filter=None):
        """Build a Milvus expression only from the supported metadata contract."""
        if metadata_filter is None:
            normalized_filter = {}
        elif isinstance(metadata_filter, Mapping):
            normalized_filter = dict(metadata_filter)
        else:
            raise TypeError("metadata_filter must be a mapping or None")

        unknown_fields = set(normalized_filter) - set(cls.METADATA_FILTER_FIELDS)
        if unknown_fields:
            raise ValueError(
                f"Unsupported metadata filter fields: {', '.join(sorted(unknown_fields))}"
            )

        if source_filter:
            if not isinstance(source_filter, str):
                raise TypeError("source_filter must be a string or None")
            existing_source = normalized_filter.get("source")
            if existing_source is not None and existing_source != source_filter:
                raise ValueError("source_filter conflicts with metadata_filter['source']")
            normalized_filter["source"] = source_filter
        elif source_filter is not None and not isinstance(source_filter, str):
            raise TypeError("source_filter must be a string or None")

        clauses = []
        for field in cls.METADATA_FILTER_FIELDS:
            value = normalized_filter.get(field)
            if value is None:
                continue
            if field == "report_year":
                if type(value) is not int:
                    raise TypeError("report_year must be an integer")
                clauses.append(f"{field} == {value}")
                continue
            if not isinstance(value, str):
                raise TypeError(f"{field} must be a string")
            escaped_value = (
                value.replace("\\", "\\\\")
                .replace('"', '\\"')
                .replace("\n", "\\n")
                .replace("\r", "\\r")
                .replace("\t", "\\t")
            )
            clauses.append(f'{field} == "{escaped_value}"')
        return " AND ".join(clauses)

    @staticmethod
    def _validated_subquery_metadata_filters(subqueries, metadata_filters):
        if metadata_filters is None:
            return [None] * len(subqueries)
        if not isinstance(metadata_filters, (list, tuple)):
            raise TypeError("metadata_filters must be a list or tuple when provided")
        if len(metadata_filters) != len(subqueries):
            raise ValueError("metadata_filters must have the same length as subqueries")
        return list(metadata_filters)

    def _search_parent_docs(
            self, dense_query_vector, sparse_query_vector, k, source_filter=None, metadata_filter=None
    ):
        """执行原有 Hybrid Search 和 Parent dedup，不负责 embedding 或 rerank。"""
        filter_expr = self.build_metadata_filter_expression(metadata_filter, source_filter)
        dense_request = AnnSearchRequest(
            data=[dense_query_vector],
            anns_field='dense_vector',
            param={'metric_type': 'IP', 'params': {'nprobe': 10}},
            limit=k,
            expr=filter_expr,
        )
        sparse_request = AnnSearchRequest(
            data=[sparse_query_vector],
            anns_field='sparse_vector',
            param={'metric_type': 'IP', 'params': {}},
            limit=k,
            expr=filter_expr,
        )

        hybrid_search_started_at = time.perf_counter()
        results = self.client.hybrid_search(
            collection_name=self.collection_name,
            reqs=[dense_request, sparse_request],
            ranker=WeightedRanker(0.7, 1.0),
            limit=k,
            output_fields=self.retrieval_output_fields,
        )[0]
        hybrid_search_seconds = time.perf_counter() - hybrid_search_started_at

        sub_chunks = [self._doc_from_hit(hit['entity']) for hit in results]
        parent_dedup_started_at = time.perf_counter()
        parent_docs = self._get_unique_parent_docs(sub_chunks)
        parent_dedup_seconds = time.perf_counter() - parent_dedup_started_at
        return parent_docs, {
            "hybrid_search_seconds": hybrid_search_seconds,
            "parent_dedup_seconds": parent_dedup_seconds,
        }

    def hybrid_search_with_rerank(
            self,
            query,
            k=config.RETRIEVAL_K,
            source_filter=None,
            return_diagnostics=False,
            metadata_filter=None,
            result_limit=None,
    ):
        """
        对输入的query进行混合检索
        :param query: 用户的查询字符串（原始的或者是被改写过的）
        :param k: milvus返回的limit值
        :param source_filter: 知识库来源过滤条件
        :return: 返回父块candidate_m个(上下文)
        """
        total_started_at = time.perf_counter()

        embedding_started_at = time.perf_counter()
        query_embeddings = self._embed_queries([query])
        embedding_seconds = time.perf_counter() - embedding_started_at
        dense_query_vector, sparse_query_vector = self._query_vectors_from_embeddings(
            query_embeddings, 0
        )
        parent_docs, search_timing = self._search_parent_docs(
            dense_query_vector, sparse_query_vector, k, source_filter, metadata_filter
        )

        # -----------------------------------到此为止，已经完成了粗排-------------------------------------

        reranker_started_at = time.perf_counter()
        ranked_pairs = []
        if parent_docs:
            # 如果父块只有一个，进行返回
            if len(parent_docs) < 2:
                ranked_pairs = [(None, parent_docs[0])]
            # 这里的parent_docs其实就是context
            # 如果父块超过一个，需要进行重排序： 基于query 和context的匹配程度做重排序
            # 构造： (query, context) 对
            # TODO 注意：这里的query是用户提出的原始的问题，context是查询到的相关上下文
            # rerank模型的作用就是基于rerank模型，再次计算query和context的相关性
            # pairs = [ [query, contex1], [query, contex2]  ,[query, contex3] ....]
            # pairs = [n ,2] , n = 参与rerank的父块的数量
            else:
                pairs = [[query, doc.page_content] for doc in parent_docs]

                # TODO 通过rerank模型， 计算 query 和 context(一个父块) 的分数，相似度
                # scores: [n] ， query和每个父块的相似度分数
                scores = self.reranker.predict(pairs)

                # key 仅包含 score；Python 的稳定排序会在相同分数时保持原始父块顺序。
                ranked_pairs = self._sort_scored_documents(scores, parent_docs)
        reranker_seconds = time.perf_counter() - reranker_started_at

        result_limit = config.CANDIDATE_M if result_limit is None else result_limit
        if type(result_limit) is not int or result_limit < 1:
            raise ValueError("result_limit must be a positive integer")
        final_docs = self._documents_from_ranked_pairs(ranked_pairs, result_limit)
        timing = {
            "embedding_seconds": embedding_seconds,
            "hybrid_search_seconds": search_timing["hybrid_search_seconds"],
            "parent_dedup_seconds": search_timing["parent_dedup_seconds"],
            "reranker_seconds": reranker_seconds,
            "total_seconds": time.perf_counter() - total_started_at,
        }
        logger.info(
            "检索阶段耗时: embedding=%.3fs, hybrid_search=%.3fs, "
            "parent_dedup=%.3fs, reranker=%.3fs, total=%.3fs",
            timing["embedding_seconds"],
            timing["hybrid_search_seconds"],
            timing["parent_dedup_seconds"],
            timing["reranker_seconds"],
            timing["total_seconds"],
        )
        if return_diagnostics:
            return final_docs, {"timing": timing, "ranked_pairs": ranked_pairs}
        return final_docs

    def hybrid_search_subqueries_with_batched_rerank(
            self,
            subqueries,
            k=config.RETRIEVAL_K,
            source_filter=None,
            reranker_batch_size=SUBQUERY_RERANK_BATCH_SIZE,
            return_diagnostics=False,
            metadata_filters=None,
            result_limit=None,
    ):
        """对多个已确定的 SubQuery 批量 embedding 和 rerank，保留每个子查询的排序语义。"""
        total_started_at = time.perf_counter()
        if not subqueries:
            empty_diagnostics = {
                "timing": {
                    "embedding_seconds": 0.0,
                    "hybrid_search_seconds": 0.0,
                    "parent_dedup_seconds": 0.0,
                    "reranker_seconds": 0.0,
                    "total_seconds": 0.0,
                },
                "reranker_predict_calls": 0,
                "ranked_pairs_by_subquery": [],
                "per_subquery": [],
            }
            return ([], empty_diagnostics) if return_diagnostics else []

        embedding_started_at = time.perf_counter()
        query_embeddings = self._embed_queries(subqueries)
        embedding_seconds = time.perf_counter() - embedding_started_at
        per_subquery_metadata_filters = self._validated_subquery_metadata_filters(
            subqueries, metadata_filters
        )

        parent_docs_by_subquery = []
        per_subquery = []
        hybrid_search_seconds = 0.0
        parent_dedup_seconds = 0.0
        for index, query in enumerate(subqueries):
            dense_query_vector, sparse_query_vector = self._query_vectors_from_embeddings(
                query_embeddings, index
            )
            parent_docs, search_timing = self._search_parent_docs(
                dense_query_vector,
                sparse_query_vector,
                k,
                source_filter,
                per_subquery_metadata_filters[index],
            )
            parent_docs_by_subquery.append(parent_docs)
            hybrid_search_seconds += search_timing["hybrid_search_seconds"]
            parent_dedup_seconds += search_timing["parent_dedup_seconds"]
            per_subquery.append({
                "query": query,
                "parent_count": len(parent_docs),
                **search_timing,
            })

        pairs = []
        for query, parent_docs in zip(subqueries, parent_docs_by_subquery):
            if len(parent_docs) > 1:
                pairs.extend([[query, doc.page_content] for doc in parent_docs])

        reranker_started_at = time.perf_counter()
        scores = self.reranker.predict(pairs, batch_size=reranker_batch_size) if pairs else []
        reranker_seconds = time.perf_counter() - reranker_started_at
        ranked_pairs_by_subquery = self._restore_subquery_rankings(
            parent_docs_by_subquery, scores
        )
        result_limit = config.CANDIDATE_M if result_limit is None else result_limit
        if type(result_limit) is not int or result_limit < 1:
            raise ValueError("result_limit must be a positive integer")
        ranked_docs_by_subquery = [
            self._documents_from_ranked_pairs(ranked_pairs, result_limit)
            for ranked_pairs in ranked_pairs_by_subquery
        ]
        timing = {
            "embedding_seconds": embedding_seconds,
            "hybrid_search_seconds": hybrid_search_seconds,
            "parent_dedup_seconds": parent_dedup_seconds,
            "reranker_seconds": reranker_seconds,
            "total_seconds": time.perf_counter() - total_started_at,
        }
        diagnostics = {
            "timing": timing,
            "reranker_predict_calls": int(bool(pairs)),
            "ranked_pairs_by_subquery": ranked_pairs_by_subquery,
            "per_subquery": per_subquery,
        }
        logger.info(
            "SubQuery batch 阶段耗时: embedding=%.3fs, hybrid_search=%.3fs, "
            "parent_dedup=%.3fs, reranker=%.3fs, predict_calls=%d, total=%.3fs",
            timing["embedding_seconds"],
            timing["hybrid_search_seconds"],
            timing["parent_dedup_seconds"],
            timing["reranker_seconds"],
            diagnostics["reranker_predict_calls"],
            timing["total_seconds"],
        )
        return (ranked_docs_by_subquery, diagnostics) if return_diagnostics else ranked_docs_by_subquery

    def hybrid_search_subqueries_with_batched_embedding(
            self,
            subqueries,
            k=config.RETRIEVAL_K,
            source_filter=None,
            return_diagnostics=False,
            metadata_filters=None,
            result_limit=None,
    ):
        """批量编码 SubQuery；保留既有的每个子查询独立 rerank 语义。"""
        total_started_at = time.perf_counter()
        if not subqueries:
            empty_diagnostics = {
                "timing": {
                    "embedding_seconds": 0.0,
                    "hybrid_search_seconds": 0.0,
                    "parent_dedup_seconds": 0.0,
                    "reranker_seconds": 0.0,
                    "total_seconds": 0.0,
                },
                "reranker_predict_calls": 0,
                "ranked_pairs_by_subquery": [],
                "per_subquery": [],
            }
            return ([], empty_diagnostics) if return_diagnostics else []

        embedding_started_at = time.perf_counter()
        query_embeddings = self._embed_queries(subqueries)
        embedding_seconds = time.perf_counter() - embedding_started_at
        per_subquery_metadata_filters = self._validated_subquery_metadata_filters(
            subqueries, metadata_filters
        )

        ranked_pairs_by_subquery = []
        per_subquery = []
        hybrid_search_seconds = 0.0
        parent_dedup_seconds = 0.0
        reranker_seconds = 0.0
        reranker_predict_calls = 0
        for index, query in enumerate(subqueries):
            subquery_started_at = time.perf_counter()
            dense_query_vector, sparse_query_vector = self._query_vectors_from_embeddings(
                query_embeddings, index
            )
            parent_docs, search_timing = self._search_parent_docs(
                dense_query_vector,
                sparse_query_vector,
                k,
                source_filter,
                per_subquery_metadata_filters[index],
            )
            hybrid_search_seconds += search_timing["hybrid_search_seconds"]
            parent_dedup_seconds += search_timing["parent_dedup_seconds"]

            reranker_started_at = time.perf_counter()
            if len(parent_docs) < 2:
                ranked_pairs = [(None, parent_docs[0])] if parent_docs else []
            else:
                scores = self.reranker.predict([[query, doc.page_content] for doc in parent_docs])
                ranked_pairs = self._sort_scored_documents(scores, parent_docs)
                reranker_predict_calls += 1
            subquery_reranker_seconds = time.perf_counter() - reranker_started_at
            reranker_seconds += subquery_reranker_seconds
            ranked_pairs_by_subquery.append(ranked_pairs)
            per_subquery.append({
                "query": query,
                "parent_count": len(parent_docs),
                **search_timing,
                "reranker_seconds": subquery_reranker_seconds,
                "subtotal_seconds": time.perf_counter() - subquery_started_at,
            })

        result_limit = config.CANDIDATE_M if result_limit is None else result_limit
        if type(result_limit) is not int or result_limit < 1:
            raise ValueError("result_limit must be a positive integer")
        ranked_docs_by_subquery = [
            self._documents_from_ranked_pairs(ranked_pairs, result_limit)
            for ranked_pairs in ranked_pairs_by_subquery
        ]
        timing = {
            "embedding_seconds": embedding_seconds,
            "hybrid_search_seconds": hybrid_search_seconds,
            "parent_dedup_seconds": parent_dedup_seconds,
            "reranker_seconds": reranker_seconds,
            "total_seconds": time.perf_counter() - total_started_at,
        }
        diagnostics = {
            "timing": timing,
            "reranker_predict_calls": reranker_predict_calls,
            "ranked_pairs_by_subquery": ranked_pairs_by_subquery,
            "per_subquery": per_subquery,
        }
        logger.info(
            "SubQuery batch embedding 阶段耗时: embedding=%.3fs, hybrid_search=%.3fs, "
            "parent_dedup=%.3fs, reranker=%.3fs, predict_calls=%d, total=%.3fs",
            timing["embedding_seconds"],
            timing["hybrid_search_seconds"],
            timing["parent_dedup_seconds"],
            timing["reranker_seconds"],
            reranker_predict_calls,
            timing["total_seconds"],
        )
        return (ranked_docs_by_subquery, diagnostics) if return_diagnostics else ranked_docs_by_subquery

        # TODO 最后保留CANDIDATE_M个父块作为最终的context
        # TODO 切片操作[:config.CANDIDATE_M] -> 切片 相当于 只保存列表中下标从0到config.CANDIDATE_M的，[0,3)
        # 长度为 10的 list -> 长度不超过CANDIDATE_M
        # 类比 字符串的sub_string(0, CANDIDATE_M)
        # TODO CANDIDATE_M？ 考虑模型能支持的输入大小
        # 1. 考虑上下文(父块）大小：一个父块1200， 3个父块3600
        # 2. 考虑多轮对话
        # 总之：父块大小 + 多轮对话 + 提示词模板 < 模型上下文大小

        # TODO 注意：这里基于rerank的score排序以后得结果无论和问题的相关性多么小，总是取相对比较大的最大值。
        # TODO 这样会存在一个问题：有可能会查出来和问题完全不相干的。 所以可以在前面增加一个阈值判断

        # 返回CANDIDATE_M条数据
        # ranked_parent_docs: 相似度分数从大到小的10个父块
        # 通过切片的方式，截取前CANDIDATE_M个
        #  ranked_parent_docs[0:config.CANDIDATE_M] -> ranked_parent_docs数组截取下标从0到CANDIDATE_M, [0,CANDIDATE_M)


        # noinspection PyMethodMayBeStatic

    def _doc_from_hit(self, hit):
        """
        把milvus返回的dict格式的子块转成Document格式
        :param hit: milvus返回的子块的各个字段，dict类型
        :return: Document类型的子块
        """
        return Document(
            page_content=hit['text'],
            metadata={
                'source': hit['source'],
                'timestamp': hit['timestamp'],
                'parent_id': hit['parent_id'],
                'parent_content': hit['parent_content'],
                'document_id': hit.get('document_id'),
                'file_sha256': hit.get('file_sha256'),
                'source_filename': hit.get('source_filename'),
                'company_name': hit.get('company_name'),
                'company_code': hit.get('company_code'),
                'report_year': hit.get('report_year'),
                'period_type': hit.get('period_type'),
                'report_period': hit.get('report_period'),
            }
        )

        # noinspection PyMethodMayBeStatic

    def _get_unique_parent_docs(self, sub_chunks):
        """
        从子块中提取父块，并进行去重
        :param sub_chunks:  从milvus中查询到的所有的子块
        :return:    去重后的父块
        """
        parent_docs = set()
        # 返回值
        unique_parent_docs = []

        for chunk in sub_chunks:
            # 拿到每个子块的父块内容
            parent_content = chunk.metadata.get('parent_content', chunk.page_content)
            # 父块内容非空，且不重复
            if parent_content and parent_content not in parent_docs:
                # 构建一个父块对象，放到返回值集合中
                unique_parent_docs.append(
                    Document(
                        # TODO 需要注意，这里的内容存放的是父块的文本
                        page_content=parent_content
                        , metadata=chunk.metadata
                    )
                )
                # 放到set中，表示已经出现过
                parent_docs.add(parent_content)

        return unique_parent_docs


if __name__ == '__main__':
    # import document_processor
    #
    # documents = document_processor.process_documents(config.FINANCIAL_DATA_DIR)
    # store = VectorStore()
    # store.add_documents(documents)
    store = VectorStore()
    result = store.hybrid_search_with_rerank("大模型学什么")
    print(result)
