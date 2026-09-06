# base/logger.py
# 导入日志库
import logging
from pathlib import Path
# 导入配置类
from base.config import config


def setup_logging(log_file=config.LOG_FILE):
    # 创建日志目录
    log_path = Path(log_file)
    log_path.parent.mkdir(parents=True, exist_ok=True)
    # 获取日志器
    logger = logging.getLogger("FinancialAgenticRAG")
    # 设置日志级别
    logger.setLevel(logging.INFO)
    # 避免重复添加处理器
    if not logger.handlers:
        # 创建文件处理器
        file_handler = logging.FileHandler(log_path, encoding='utf-8')
        # 设置文件处理器级别
        file_handler.setLevel(logging.INFO)
        # 创建控制台处理器
        console_handler = logging.StreamHandler()
        # 设置控制台处理器级别
        console_handler.setLevel(logging.INFO)
        # 设置日志格式
        formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(module)s %(lineno)d - %(message)s')
        # 为文件处理器设置格式
        file_handler.setFormatter(formatter)
        # 为控制台处理器设置格式
        console_handler.setFormatter(formatter)
        # 添加文件处理器
        logger.addHandler(file_handler)
        # 添加控制台处理器
        logger.addHandler(console_handler)
    # 返回日志器
    return logger


# 初始化日志器
logger = setup_logging()

if __name__ == '__main__':
    logger.info('你好')
