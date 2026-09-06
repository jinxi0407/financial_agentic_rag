# base/config.py
# 导入配置解析库
import configparser
import json
import os
import re
from pathlib import Path

from dotenv import load_dotenv

class Config:
    # 初始化配置，加载 config.ini 文件
    def __init__(self, config_file=None):
        # 创建配置解析器，启用插值功能
        self.config = configparser.ConfigParser(interpolation=configparser.ExtendedInterpolation())
        # 以配置文件自身的位置计算项目根目录，不依赖启动命令所在目录。
        self.PROJECT_ROOT_PATH = Path(__file__).resolve().parent.parent
        self.PROJECT_ROOT = str(self.PROJECT_ROOT_PATH)
        # The project-local .env wins so a key exported by another project
        # cannot be inherited accidentally by Financial Agentic RAG.
        load_dotenv(self.PROJECT_ROOT_PATH / '.env', override=True)

        self.MODELS_DIR = self._project_path('rag_qa/models')
        financial_data_dir = os.getenv('FINANCIAL_DATA_DIR', '').strip() or 'financial_data'
        self.FINANCIAL_DATA_DIR = self._project_path(financial_data_dir)
        self.DOCUMENT_LOADERS_DIR = self._project_path('rag_qa/document_loaders')

        if config_file is None:
            config_file = self.PROJECT_ROOT_PATH / 'config.ini'
        else:
            config_file = Path(config_file).expanduser()
            if not config_file.is_absolute():
                config_file = self.PROJECT_ROOT_PATH / config_file
        self.CONFIG_FILE = str(config_file)
        # 读取配置文件
        self.config.read(self.CONFIG_FILE)

        self.PROJECT_NAME = os.getenv(
            'APP_NAME', self.config.get('project', 'name', fallback='Financial Agentic RAG')
        )
        self.APP_HOST = os.getenv(
            'HOST', self.config.get('app', 'host', fallback='0.0.0.0')
        )
        self.APP_PORT = int(os.getenv(
            'PORT', self.config.get('app', 'port', fallback='8001')
        ))
        static_dir = os.getenv('STATIC_DIR', '').strip() or 'static'
        self.STATIC_DIR = self._project_path(static_dir)

        # MySQL 配置
        # MySQL 主机地址
        self.MYSQL_HOST = os.getenv('MYSQL_HOST', self.config.get('mysql', 'host', fallback='localhost'))
        self.MYSQL_PORT = int(os.getenv('MYSQL_PORT', self.config.get('mysql', 'port', fallback=3306)))
        # MySQL 用户名
        self.MYSQL_USER = os.getenv('MYSQL_USER', self.config.get('mysql', 'user', fallback='edu_rag'))
        # MySQL 密码
        self.MYSQL_PASSWORD = os.getenv('MYSQL_PASSWORD', self.config.get('mysql', 'password', fallback='123456'))
        # MySQL 数据库名
        self.MYSQL_DATABASE = os.getenv('MYSQL_DATABASE', self.config.get('mysql', 'database', fallback='financial_qa'))
        self.MYSQL_FAQ_TABLE = self._sql_identifier(os.getenv(
            'MYSQL_FAQ_TABLE', self.config.get('mysql', 'faq_table', fallback='financial_faq')
        ))
        self.MYSQL_CONVERSATION_TABLE = self._sql_identifier(os.getenv(
            'MYSQL_CONVERSATION_TABLE',
            self.config.get('mysql', 'conversation_table', fallback='financial_conversations')
        ))

        # Redis 配置
        # Redis 主机地址
        self.REDIS_HOST = os.getenv('REDIS_HOST', self.config.get('redis', 'host', fallback='localhost'))
        # Redis 端口
        self.REDIS_PORT = int(os.getenv('REDIS_PORT', self.config.get('redis', 'port', fallback=6379)))
        # Redis 密码
        self.REDIS_PASSWORD = os.getenv('REDIS_PASSWORD', self.config.get('redis', 'password', fallback='1234'))
        # Redis 数据库编号
        self.REDIS_DB = int(os.getenv('REDIS_DB', self.config.get('redis', 'db', fallback=1)))
        self.REDIS_KEY_PREFIX = os.getenv(
            'REDIS_KEY_PREFIX', self.config.get('redis', 'key_prefix', fallback='financial:')
        )

        # Milvus 配置
        # Milvus 主机地址
        self.MILVUS_HOST = os.getenv('MILVUS_HOST', self.config.get('milvus', 'host', fallback='localhost'))
        # Milvus 端口
        self.MILVUS_PORT = os.getenv('MILVUS_PORT', self.config.get('milvus', 'port', fallback='19530'))
        # Milvus 数据库名
        self.MILVUS_DATABASE_NAME = os.getenv(
            'MILVUS_DATABASE',
            os.getenv(
                'MILVUS_DATABASE_NAME',
                self.config.get('milvus', 'database_name', fallback='financial')
            )
        )
        # Milvus 集合名
        self.MILVUS_COLLECTION_NAME = os.getenv(
            'MILVUS_COLLECTION',
            os.getenv(
                'MILVUS_COLLECTION_NAME',
                self.config.get('milvus', 'collection_name', fallback='financial_rag_v1')
            )
        )

        # LLM 配置
        # LLM 模型名
        self.LLM_MODEL = os.getenv('LLM_MODEL', self.config.get('llm', 'model', fallback='')).strip()
        # DashScope API 密钥
        self.DASHSCOPE_API_KEY = os.getenv(
            'DASHSCOPE_API_KEY', self.config.get('llm', 'dashscope_api_key', fallback='')
        ).strip()
        # DashScope API 地址
        self.DASHSCOPE_BASE_URL = os.getenv(
            'DASHSCOPE_BASE_URL', self.config.get('llm', 'dashscope_base_url', fallback='')
        ).strip()

        self.BGE_M3_MODEL_PATH = self._model_path(
            'BGE_M3_MODEL_PATH', 'bge_m3_model_path', 'bge-m3'
        )
        self.RERANKER_MODEL_PATH = self._model_path(
            'RERANKER_MODEL_PATH', 'reranker_model_path', 'bge-reranker-large'
        )
        self.BERT_ROUTER_MODEL_PATH = self._model_path(
            'BERT_ROUTER_MODEL_PATH', 'bert_router_model_path', 'bert_query_classifier'
        )
        self.BERT_BASE_MODEL_PATH = self._model_path(
            'BERT_BASE_MODEL_PATH', 'bert_base_model_path', 'bert-base-chinese'
        )
        self.DOCUMENT_SEGMENTATION_MODEL_PATH = self._model_path(
            'DOCUMENT_SEGMENTATION_MODEL_PATH',
            'document_segmentation_model_path',
            'nlp_bert_document-segmentation_chinese-base'
        )

        # 检索参数
        # 父块大小
        self.PARENT_CHUNK_SIZE = self.config.getint('retrieval', 'parent_chunk_size', fallback=1200)
        # 子块大小
        self.CHILD_CHUNK_SIZE = self.config.getint('retrieval', 'child_chunk_size', fallback=300)
        # 块重叠大小
        self.CHUNK_OVERLAP = self.config.getint('retrieval', 'chunk_overlap', fallback=50)
        # 检索返回数量
        self.RETRIEVAL_K = self.config.getint('retrieval', 'retrieval_k', fallback=5)
        # 最终候选数量
        self.CANDIDATE_M = self.config.getint('retrieval', 'candidate_m', fallback=2)

        # 应用配置
        # 有效来源列表
        valid_sources = os.getenv(
            'VALID_SOURCES',
            self.config.get('app', 'valid_sources', fallback='["annual_reports", "financial_knowledge"]')
        )
        self.VALID_SOURCES = json.loads(valid_sources)
        # 客服电话
        self.CUSTOMER_SERVICE_PHONE = self.config.get('app', 'customer_service_phone', fallback='12345678')

        # 日志文件路径
        log_file = os.getenv(
            'LOG_FILE', self.config.get('logger', 'log_file', fallback='logs/app.log')
        ).strip() or 'logs/app.log'
        self.LOG_FILE = self._project_path(log_file)
        self.LOG_DIR = str(Path(self.LOG_FILE).parent)

    def _model_path(self, env_name, option_name, default_directory):
        configured_path = os.getenv(
            env_name, self.config.get('models', option_name, fallback='')
        ).strip()
        path = configured_path or str(Path(self.MODELS_DIR) / default_directory)
        return self._project_path(path)

    def _project_path(self, value):
        path = Path(value).expanduser()
        if not path.is_absolute():
            path = self.PROJECT_ROOT_PATH / path
        return str(path)

    @staticmethod
    def _sql_identifier(value):
        if not re.fullmatch(r'[A-Za-z_][A-Za-z0-9_]*', value):
            raise ValueError(f'非法的 MySQL 标识符: {value!r}')
        return value


config = Config()

if __name__ == '__main__':
    conf = Config()
    print(f'PROJECT_ROOT={conf.PROJECT_ROOT}')
    print(f'FINANCIAL_DATA_DIR={conf.FINANCIAL_DATA_DIR}')
    print(f'MYSQL_DATABASE={conf.MYSQL_DATABASE}')
    print(f'REDIS_DB={conf.REDIS_DB}')
    print(f'REDIS_KEY_PREFIX={conf.REDIS_KEY_PREFIX}')
    print(f'MILVUS={conf.MILVUS_DATABASE_NAME}/{conf.MILVUS_COLLECTION_NAME}')
    print(f'FASTAPI_PORT={conf.APP_PORT}')
    print('LLM API Key 已配置' if conf.DASHSCOPE_API_KEY else 'LLM API Key 尚未配置')
