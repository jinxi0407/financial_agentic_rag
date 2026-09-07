from redis import StrictRedis

from base.logger import logger
from base.config import config
import json

"""
redis客户端，要实现以下几个功能：

1. 初始化
2. get_data: 读取redis，对json字符串的数据进行反序列化
3. set_data: 将list<str>格式的数据进行序列化转成json字符串，存入redis
4. get_answer: 根据问题读取答案, key:answer:问题
5. set_answer: 缓存问题和答案对，key:answer:问题

"""


class RedisClient(object):
    def __init__(self):
        self.redis = StrictRedis(host=config.REDIS_HOST
                                 , port=config.REDIS_PORT
                                 , db=config.REDIS_DB
                                 # TODO 如果遇到了返回值是编码（非正常字符串）
                                 , decode_responses=True
                                 , encoding='utf-8'
                                 , password=config.REDIS_PASSWORD
                                 )
        logger.info("redis初始化成功")

    def get_data(self, key):
        try:
            value = self.redis.get(key)
            if value:
                result = json.loads(value)
                logger.info(f"获取数据成功！key: {key}, result: {result}")
                return result
            else:
                logger.info("获取数据失败！key:{key}不存在".format(key=key))
        except Exception as e:
            logger.error(f'获取数据失败! error: {e}')
            return []

    def set_data(self, key, value):
        try:
            self.redis.set(key, json.dumps(value))
            logger.info(f"写入数据成功！value: {key}")
        except Exception as e:
            logger.error(f'写入数据失败！ {key} error: {e}')

    def delete_keys(self, *keys):
        """Delete only explicitly named cache keys; never flush the Redis database."""
        keys = [key for key in keys if key]
        if not keys:
            return 0
        try:
            deleted = self.redis.delete(*keys)
            logger.info("删除指定缓存键成功，共{}个".format(deleted))
            return deleted
        except Exception as e:
            logger.error(f'删除指定缓存键失败！ error: {e}')
            raise

    def get_answer(self, question):
        try:
            key = f"{config.REDIS_KEY_PREFIX}answer:{question}"
            value = self.redis.get(key)
            if value:
                logger.info(f"获取数据成功！key: {key}, result: {value}")
                self.redis.expire(key, 24 * 60 * 60)
            else:
                logger.info(f"没有获取到对应的数据, question: {question}")
            return value
        except Exception as e:
            logger.error(f'获取数据失败！error: {e}')
            return None

    def set_answer(self, question, answer):
        try:
            key = f"{config.REDIS_KEY_PREFIX}answer:{question}"
            self.redis.set(key, answer, ex=24 * 60 * 60)
            logger.debug(f"写入数据成功！value: {key}, answer: {answer}")
        except Exception as e:
            logger.error(f'写入数据失败！ error: {e}')


if __name__ == '__main__':
    client = RedisClient()
    answer = client.get_answer("")
    print(answer)
