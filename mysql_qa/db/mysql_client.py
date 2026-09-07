"""
mysql-client模块

需要实现的功能列表：
1. 初始化Mysql客户端
2. 创建高频问答表
3. csv数据写入mysql
4. 读取所有的问题
5. 根据问题获取对应的答案

"""

from pymysql import connect, cursors
from base.config import config
from base.logger import logger
import pandas as pd
import json


class MysqlClient:

    # 1. 初始化Mysql客户端
    def __init__(self):
        try:
            # mysql链接
            self.connect = connect(
                host=config.MYSQL_HOST
                , user=config.MYSQL_USER
                , password=config.MYSQL_PASSWORD
                , db=config.MYSQL_DATABASE
                , port=config.MYSQL_PORT
                , charset='utf8mb4'
            )
            # mysql游标
            self.cursor = self.connect.cursor()
            logger.info("mysql初始化成功")
        except Exception as e:
            logger.error("mysql初始化失败: {}".format(e))
            raise

    # 2. 创建高频问答表
    def create_table(self):
        create_table_query = f'''
                             CREATE TABLE IF NOT EXISTS {config.MYSQL_FAQ_TABLE}
                             (
                                 id       BIGINT AUTO_INCREMENT PRIMARY KEY,
                                 category VARCHAR(100),
                                 question VARCHAR(1000) NOT NULL,
                                 answer   TEXT NOT NULL,
                                 keywords JSON NULL,
                                 intent_id VARCHAR(191) NULL,
                                 UNIQUE KEY uq_financial_faq_intent_id (intent_id)
                             )
                             '''
        try:
            self.cursor.execute(create_table_query)
            self.connect.commit()
            logger.info("建表语句执行成功！")
        except Exception as e:
            logger.error("建表语句执行失败: {}".format(e))
            raise

    def ensure_faq_schema(self):
        """Add the structured fields needed by the financial FAQ import."""
        self.create_table()
        try:
            self.cursor.execute(f"SHOW COLUMNS FROM {config.MYSQL_FAQ_TABLE}")
            columns = {row[0] for row in self.cursor.fetchall()}

            if 'keywords' not in columns:
                self.cursor.execute(
                    f"ALTER TABLE {config.MYSQL_FAQ_TABLE} ADD COLUMN keywords JSON NULL"
                )
            if 'intent_id' not in columns:
                self.cursor.execute(
                    f"ALTER TABLE {config.MYSQL_FAQ_TABLE} ADD COLUMN intent_id VARCHAR(191) NULL"
                )

            self.cursor.execute(f"SHOW INDEX FROM {config.MYSQL_FAQ_TABLE}")
            index_names = {row[2] for row in self.cursor.fetchall()}
            if 'uq_financial_faq_intent_id' not in index_names:
                self.cursor.execute(
                    f"ALTER TABLE {config.MYSQL_FAQ_TABLE} "
                    "ADD UNIQUE KEY uq_financial_faq_intent_id (intent_id)"
                )

            self.connect.commit()
            logger.info("financial FAQ schema is ready")
        except Exception as e:
            self.connect.rollback()
            logger.error("financial FAQ schema migration failed: {}".format(e))
            raise

    # 3. csv数据写入mysql
    def insert_data(self, csv_path):
        data_frame = pd.read_csv(csv_path)
        try:
            # series = data_frame['问题']
            # for i in range(data_frame.count()):
            #     question = data_frame['问题'][i]
            #     answer = data_frame['答案'][i]
            #     sql = 'insert into jpkb (question, answer, subject_name) values (%s, %s, %s)'
            #     self.cursor.execute(sql, (question, answer, subject))
            # TODO data_frame.iterrows() 其中的一种遍历方式，返回一个迭代器
            for _, row in data_frame.iterrows():
                question = row['问题']
                answer = row['答案']
                category = row.get('金融类别', row.get('类别', row.get('学科名称')))
                # 防注入
                # insert into jpkb (question, answer, subject_name) values ('a','a','a');drop table jpkb
                sql = f'''insert into {config.MYSQL_FAQ_TABLE}
                          (question, answer, category) values (%s, %s, %s)'''
                self.cursor.execute(sql, (question, answer, category))
            # 把这几百条数据的提交作为一次事务，要么全部成功，要不全部失败
            # TODO connect.commit(): 提交事务
            self.connect.commit()
            logger.info("输入导入成功，共：{}条".format(data_frame.count()))
        except Exception as e:
            # 事务回滚
            # TODO connect.rollback(): 事务回滚，如果事务执行到一半的时候失败了，回滚之前全部的操作。 （要么全部执行成功，要么全部失败）
            self.connect.rollback()
            logger.error("数据导入失败: {}".format(e))

    # 4. 读取所有的问题
    def fetch_questions(self):
        try:
            sql = f"select question from {config.MYSQL_FAQ_TABLE}"
            self.cursor.execute(sql)
            # tuple[tuple] -> [条数=467，列=1]
            # TODO 执行查询类SQL的时候，如果要拿到结果，除了execute以外，还需要执行cursor.fetchall()
            # TODO fetchall()-> tuple[tuple]: 拿到所有的结果; fetchone()->tuple: 拿到一条结果; fetchmany(size) -> tuple[tuple]: 拿到size条数据
            results = self.cursor.fetchall()
            # row in results ：拿到每条数据（包含多个列）
            # row[0] ：拿到question列
            questions = [row[0] for row in results]
            logger.info("获取到了所有的问题：{}条".format(len(questions)))
            return questions
        except Exception as e:
            logger.error("获取所有的问题失败: {}".format(e))
            return []

    def fetch_faq_search_documents(self, questions):
        """Return canonical questions enriched with their stored keyword aliases."""
        try:
            self.cursor.execute(
                f"SELECT question, keywords FROM {config.MYSQL_FAQ_TABLE}"
            )
            keywords_by_question = {}
            for question, keywords in self.cursor.fetchall():
                if isinstance(keywords, str):
                    keywords = json.loads(keywords)
                if not isinstance(keywords, list):
                    keywords = []
                keywords_by_question[question] = [
                    keyword for keyword in keywords if isinstance(keyword, str)
                ]
            return [
                " ".join([question, *keywords_by_question.get(question, [])])
                for question in questions
            ]
        except Exception as e:
            logger.error("获取 FAQ 搜索文本失败: {}".format(e))
            return list(questions)

    # 5. 根据问题获取对应的答案
    def fetch_answer(self, question):
        try:
            # TODO %s 加上执行cursor.execute 再传入变量，这种方式是为了解决SQL注入的问题
            sql = f"select answer from {config.MYSQL_FAQ_TABLE} where question = %s"
            self.cursor.execute(sql, (question,))
            # TODO 返回的是tuple类型的
            result = self.cursor.fetchone()
            if result:
                logger.info("获取问题: {} 对应的答案成功".format(question))
                return result[0]
            else:
                logger.error("没有找到问题对应的答案: {}".format(question))
                return None
        except Exception as e:
            logger.error("查询问题对应的答案失败: {}".format(e))
            return None


    def close(self):
        # 关闭数据库连接
        try:
            self.cursor.close()
            # 关闭连接
            self.connection.close()
            # 记录关闭成功
            self.logger.info("MySQL 连接已关闭")
        except pymysql.MySQLError as e:
            # 记录关闭失败
            self.logger.error(f"关闭连接失败: {e}")




if __name__ == '__main__':
    mysql_client = MysqlClient()
    # mysql_client.create_table()
    # mysql_client.insert_data('path/to/financial_faq.csv')

    # questions = mysql_client.fetch_questions()
    # print(questions)

    answer = mysql_client.fetch_answer("关联子查询的执行顺序是什么")
    print(answer)
