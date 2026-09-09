# 导入 FastAPI 相关模块，用于构建 API 和 WebSocket
from fastapi import FastAPI, WebSocket, HTTPException, Query, Depends
# 导入 FastAPI 响应类型，用于流式响应和文件服务
from fastapi.responses import StreamingResponse, FileResponse
# 导入 CORS 中间件，支持跨域请求
from fastapi.middleware.cors import CORSMiddleware
# 导入静态文件服务模块
# 给前端使用
from fastapi.staticfiles import StaticFiles
# 导入 WebSocket 断开异常
from starlette.websockets import WebSocketDisconnect
from pathlib import Path
# 导入 Pydantic 模型，用于请求验证
from pydantic import BaseModel

# 导入异步事件循环模块
import asyncio
import os
import threading

# 导入 JSON 处理模块
import json
# 导入 UUID 模块，生成唯一会话 ID
import uuid
# 导入类型注解模块
from typing import Optional, List, Dict, Any
# 导入时间模块，记录处理时间
import time
# 导入正则表达式模块，用于匹配日常问候
import re
# 导入优化后的问答系统
from new_main import IntegratedQASystem
from agent.langgraph_agent import LangGraphFinancialAgent
from agent.streaming import AgentStreamingAdapter, serve_agent_websocket

from base.config import config
from base.logger import logger

# 创建 FastAPI 应用实例，设置标题和描述
app = FastAPI(
    title="Financial Agentic RAG API",
    description="金融智能问答与金融知识库 API"
)

# 配置 CORS 中间件，允许跨域请求
def _allowed_origins() -> list[str]:
    configured = os.getenv("ALLOWED_ORIGINS", "").strip()
    if configured:
        origins = [origin.strip() for origin in configured.split(",") if origin.strip() and origin.strip() != "*"]
        if origins:
            return origins
    return [
        "http://localhost:3000",
        "http://127.0.0.1:3000",
        "http://localhost:5173",
        "http://127.0.0.1:5173",
    ]


app.add_middleware(
    CORSMiddleware,
    allow_origins=_allowed_origins(),
    allow_credentials=True,
    allow_methods=["*"],  # 允许所有 HTTP 方法
    allow_headers=["*"],  # 允许所有头部
)

# 创建静态文件目录（如果不存在）
STATIC_DIR = Path(config.STATIC_DIR)
STATIC_DIR.mkdir(parents=True, exist_ok=True)

# 创建全局问答系统实例
qa_system = IntegratedQASystem()
agent_stream_adapter: Optional[AgentStreamingAdapter] = None
_agent_stream_lock = threading.Lock()


def get_agent_stream_adapter() -> AgentStreamingAdapter:
    """Create the persistent Agent only on the first Agent WebSocket request."""
    global agent_stream_adapter
    with _agent_stream_lock:
        if agent_stream_adapter is None:
            agent = LangGraphFinancialAgent(
                qa_system=qa_system,
                checkpointer=LangGraphFinancialAgent.redis_checkpointer(),
            )
            agent_stream_adapter = AgentStreamingAdapter(agent)
    return agent_stream_adapter

# 定义日常问候用语模式和回复
GREETING_PATTERNS = [
    {
        "pattern": r"^(你好|您好|hi|hello)",  # 匹配问候语
        "response": "你好！我是 Financial Agentic RAG 金融分析助手，很高兴为你服务！"
    },
    {
        "pattern": r"^(你是谁|您是谁|你叫什么|你的名字|who are you)",  # 匹配身份询问
        "response": "我是 Financial Agentic RAG 金融分析助手。"
    },
    {
        "pattern": r"^(在吗|在不在|有人吗)",  # 匹配在线确认
        "response": "我在，随时可以协助你查询和分析金融信息。"
    },
    {
        "pattern": r"^(干嘛呢|你在干嘛|做什么)",  # 匹配状态询问
        "response": "我正在待命，有什么金融问题可以帮你分析？"
    }
]


# 定义查询请求模型
# TODO 相当于java中的注解 @RequestBody
class QueryRequest(BaseModel):
    # TODO @Param(param = "query", nullable = false)
    query: str  # 查询内容，必填
    # TODO @Param(param = "query", nullable = true)
    source_filter: Optional[str] = None  # 金融知识库过滤，可选
    session_id: Optional[str] = None  # 会话 ID，可选


# 定义查询响应模型
# TODO 相当于java中的注解 @ResponseBody
class QueryResponse(BaseModel):
    answer: str  # 答案内容
    is_streaming: bool  # 是否流式响应
    session_id: str  # 会话 ID
    processing_time: float  # 处理时间


# 挂载静态文件目录，服务前端文件
# TODO app.mount("/static" -> 127.0.0.1/static
app.mount("/static", StaticFiles(directory=str(STATIC_DIR)), name="static")


# 根路径重定向到 index.html
# TODO 127.0.0.1/static/index.html -> 127.0.0.1/
@app.get("/")
async def read_root():
    return FileResponse(STATIC_DIR / "index.html")

# 创建新会话接口
# TODO 类似于java中的@PostMapping
@app.post("/api/create_session")
async def create_session():
    session_id = str(uuid.uuid4())  # 生成唯一会话 ID
    return {"session_id": session_id}  # 返回会话 ID


# 查询历史消息接口
# TODO restful api 规范 。比如某厂规范：internal-api.xxxx.com/{业务线: driver/passanger}/{二级:运营、打车}/xxxx
# TODO 对外的api(openapi) :  /api/v1/financial-rag/history/{session_id}
@app.get("/api/history/{session_id}")
async def get_history(session_id: str):
    try:
        # 获取指定会话的历史记录
        history = qa_system.get_session_history(session_id)
        # 返回会话 ID 和历史记录
        # TODO 规范操作：{"data": {"session_id": session_id, "history": history} ,"errno":0,"errmsg":报错信息, "log_id":日志编号}
        return {"session_id": session_id, "history": history}
    except Exception as e:
        # 抛出 HTTP 异常，包含错误信息
        raise HTTPException(status_code=500, detail=f"获取历史记录失败: {str(e)}")


# 清除历史消息接口
@app.delete("/api/history/{session_id}")
async def clear_history(session_id: str):
    # 清除指定会话的历史记录
    success = qa_system.clear_session_history(session_id)
    if success:
        # 返回成功状态
        return {"status": "success", "message": "历史记录已清除"}
    else:
        # 抛出 HTTP 异常
        raise HTTPException(status_code=500, detail="清除历史记录失败")

# 检查是否为日常问候用语并返回模板回复
def check_greeting(query: str) -> Optional[str]:
    query_text = query.strip()  # 去除首尾空格
    for pattern_info in GREETING_PATTERNS:
        # 使用正则匹配，忽略大小写
        if re.match(pattern_info["pattern"], query_text, re.IGNORECASE):
            return pattern_info["response"]  # 返回匹配的回复
    return None  # 无匹配返回 None


# 非流式查询接口
@app.post("/api/query")
async def query(request: QueryRequest):
    start_time = time.time()  # 记录开始时间
    # 使用请求中的 session_id 或生成新 ID
    session_id = request.session_id or str(uuid.uuid4())
    # 检查是否为日常问候
    greeting_response = check_greeting(request.query)
    if greeting_response:
        # 返回问候回复
        return {
            "answer": greeting_response,
            "is_streaming": False,
            "session_id": session_id,
            "processing_time": time.time() - start_time
        }
    # 执行 BM25 搜索
    answer, need_rag = qa_system.faq.query(request.query, threshold=0.85)
    if need_rag:
        # 需要 RAG，提示使用 WebSocket
        return {
            "answer": "请使用WebSocket接口获取流式响应",
            "is_streaming": True,
            "session_id": session_id,
            "processing_time": time.time() - start_time
        }
    # 返回 MySQL 答案
    return {
        "answer": answer,
        "is_streaming": False,
        "session_id": session_id,
        "processing_time": time.time() - start_time
    }



# 流式查询 WebSocket 接口
# TODO 这里用的还是fastapi框架，但是这里接受的对象不会转成我们定义的结构体。 使用的是WebSocket
@app.websocket("/api/stream")
async def websocket_endpoint(websocket: WebSocket):
    await websocket.accept()  # 接受 WebSocket 连接
    try:
        while True:
            # 接收客户端消息
            data = await websocket.receive_text()
            request_data = json.loads(data)  # 解析 JSON 数据
            # 获取查询参数
            query = request_data.get("query")
            source_filter = request_data.get("source_filter")
            session_id = request_data.get("session_id", str(uuid.uuid4()))
            start_time = time.time()  # 记录开始时间
            # 发送开始标志
            if websocket.client_state == websocket.client_state.CONNECTED:
                # TODO 这里注意，这里的json格式是和前端约定好的（可以理解为接口的设计）
                # "type": 事件类型。 start代表开始接收 ,token就是大模型的输出的内容 ,end结束标志, error报错

                await websocket.send_json({
                    "type": "start",
                    "session_id": session_id
                })
            # 检查是否为日常问候
            greeting_response = check_greeting(query)
            if greeting_response:
                if websocket.client_state == websocket.client_state.CONNECTED:
                    # 发送问候回复
                    await websocket.send_json({
                        "type": "token",
                        "token": greeting_response,
                        "session_id": session_id
                    })
                    # 发送结束标志
                    await websocket.send_json({
                        "type": "end",
                        "session_id": session_id,
                        "is_complete": True,
                        "processing_time": time.time() - start_time
                    })
                break
            # 调用问答系统，流式处理查询
            collected_answer = ""
            for token, is_complete in qa_system.query(query, source_filter=source_filter, session_id=session_id):
                collected_answer += token  # 累积答案
                if is_complete:
                    if websocket.client_state == websocket.client_state.CONNECTED:
                        # 发送结束标志
                        await websocket.send_json({
                            "type": "end",
                            "session_id": session_id,
                            "is_complete": True,
                            "processing_time": time.time() - start_time
                        })
                    break
                if token and websocket.client_state == websocket.client_state.CONNECTED:
                    # 发送 token 数据
                    await websocket.send_json({
                        "type": "token",
                        "token": token,
                        "session_id": session_id
                    })
                # 每10ms输出一次token
                await asyncio.sleep(0.01)  # 控制流式输出的速度
    except WebSocketDisconnect as e:
        # 记录 WebSocket 断开信息
        logger.error(f"WebSocket disconnected: code={e.code}, reason={e.reason}")
    except Exception as e:
        # 记录错误信息
        logger.error(f"WebSocket error: {str(e)}")
        if websocket.client_state == websocket.client_state.CONNECTED:
            # 发送错误消息
            await websocket.send_json({
                "type": "error",
                "error": str(e)
            })
    finally:
        try:
            if websocket.client_state == websocket.client_state.CONNECTED:
                # 关闭 WebSocket 连接
                await websocket.close()
        except Exception as e:
            # 记录关闭连接时的错误
            logger.error(f"Error closing WebSocket: {str(e)}")


@app.websocket("/api/agent/stream")
async def agent_stream_endpoint(websocket: WebSocket):
    """Streaming lifecycle events for the complete LangGraph Agent."""
    await serve_agent_websocket(
        websocket,
        get_agent_stream_adapter,
        log=lambda event: logger.info("agent_stream event=%s", event),
    )

# 健康检查接口
@app.get("/health")
async def health_check():
    return {
        "status": "healthy",
        "service": qa_system.config.PROJECT_NAME,
        "agent": "endpoint_available",
    }


# 获取有效金融知识库类别接口
@app.get("/api/sources")
async def get_sources():
    return {"sources": qa_system.config.VALID_SOURCES}




# 主程序入口
if __name__ == "__main__":
    # springboot = springcore + spirngmvc + tomcat
    # fastapi = springmvc (url -> 方法调用)
    # uvicorn = tomcat (服务容器，负责处理多线程、高并发等)

    import uvicorn

    # 从环境变量获取主机和端口，默认值为 0.0.0.0:8001
    host = config.APP_HOST
    port = config.APP_PORT
    
    # 运行 FastAPI 应用，监听指定的主机和端口
    uvicorn.run(app, host=host, port=port, reload=False)
