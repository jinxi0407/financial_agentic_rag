# core/prompts.py
# 导入 PromptTemplate 类，用于创建 Prompt 模板
from langchain_core.prompts import PromptTemplate


# 定义 RAGPrompts 类，用于管理所有 Prompt 模板
class RAGPrompts:
    INSUFFICIENT_CONTEXT_RESPONSE = "当前知识库中缺少足够的可靠信息，无法基于现有资料回答。"

    # 定义 RAG 提示模板
    # @: 装饰器， 注解(是给计算看的),注释(给码农看的)
    @staticmethod
    def rag_prompt():
        # 创建并返回 PromptTemplate 对象
        return PromptTemplate(
            template="""  
            你是 Financial Agentic RAG 金融分析助手，帮助用户回答金融问题。你的语言专业、严谨、言简意赅。
            必须仅依据检索上下文中的信息回答，不得使用模型自身知识补充上下文中没有提供的事实、数字或结论。
            历史对话只用于理解用户的指代和问题背景，不能替代检索上下文作为事实依据。
            如果上下文为空、与问题无关、证据不足或内容互相冲突，请原样回复：
            “{insufficient_context_response}”
            如果能够回答，请明确说明回答依据来自当前知识库资料。

            上下文: 「 {context} 」
            问题: 「 {question} 」
            历史对话：「 {history} 」

            回答:  
            """,
            #   定义输入变量
            input_variables=["context", "question", "history"],
            partial_variables={
                "insufficient_context_response": RAGPrompts.INSUFFICIENT_CONTEXT_RESPONSE
            },
        )

    # 定义假设问题生成的 Prompt 模板
    @staticmethod
    def hyde_prompt():
        #   创建并返回 PromptTemplate 对象
        return PromptTemplate(
            template="""  
            假设你是用户，想了解以下问题，请生成一个简短的假设答案：  
            问题: 「 {query} 」  
            假设答案:  
            """,
            #   定义输入变量
            input_variables=["query"],
        )

    #   定义子查询生成的 Prompt 模板
    @staticmethod
    def subquery_prompt():
        #   创建并返回 PromptTemplate 对象
        return PromptTemplate(
            template="""  
            将以下复杂查询分解为多个简单子查询，每行一个子查询：  
            查询: 「 {query} 」  
            子查询:  
            """,
            #   定义输入变量
            input_variables=["query"],
        )

    #   定义回溯问题生成的 Prompt 模板
    @staticmethod
    def backtracking_prompt():
        #   创建并返回 PromptTemplate 对象
        return PromptTemplate(
            template="""  
            将以下复杂查询简化为一个更简单的问题：  
            查询: 「 {query} 」  
            简化问题:  
            """,
            #   定义输入变量
            input_variables=["query"],
        )


if __name__ == '__main__':
    prompt = RAGPrompts.hyde_prompt()
    prompt_str = prompt.format(query="公司营收增长但经营现金流下降，可能有哪些原因？")
    print(prompt_str)
