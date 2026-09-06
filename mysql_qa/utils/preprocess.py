import jieba
# 导入日志
from base.logger import logger

"""
文本预处理的流程：
1. 英文统一转小写
2. 把句子进行分词

什么时候调用：
1. 写入redis “分词后的问题” 的时候（给bm25算法使用）
2. 用户提交query进行查询的时候
"""

def preprocess_text(text):
    # 预处理文本
    logger.debug("开始预处理文本: {}".format(text))
    try:
        # 分词并转换为小写
        return jieba.lcut(text.lower())
    except AttributeError as e:
        # 记录预处理失败
        logger.error(f"文本预处理失败: {e}")
        # 返回空列表
        return []
