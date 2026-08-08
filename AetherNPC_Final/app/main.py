"""FastAPI 入口 + WebSocket 支持"""

import json
import logging
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Any, AsyncIterator

from fastapi import FastAPI, HTTPException, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles

from app import config
from app.schemas import DialogueRequest, TestRunRequest
from app.services.dialogue_service import get_dialogue_service
from app.agent_tester import get_agent_tester
from app.llm_client import get_llm_client
from app.rag import get_rag_store
from app.memory import get_memory_manager
from app.story_engine import get_story_engine

logger = logging.getLogger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncIterator[None]:
    # 预热所有单例
    get_llm_client()
    get_rag_store()
    get_memory_manager()
    get_story_engine()
    get_dialogue_service()
    get_agent_tester()
    yield
    # 关闭
    await get_llm_client().close()


app = FastAPI(
    title="AetherNPC - 智能 NPC 对话系统",
    description="基于大语言模型的游戏 NPC 对话与剧情推进系统",
    version="2.0.0",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# 挂载静态文件（demo.html）
static_dir = Path(__file__).resolve().parent.parent / "static"
try:
    static_dir.mkdir(exist_ok=True)
except Exception as exc:
    logger.warning("静态目录创建失败: %s", exc)
app.mount("/static", StaticFiles(directory=str(static_dir)), name="static")


@app.post("/sessions")
async def create_session(player_id: str = "player-001") -> dict[str, str]:
    """创建会话"""
    session_id = get_dialogue_service().create_session(player_id)
    return {"session_id": session_id, "player_id": player_id}


@app.post("/dialogue")
async def dialogue(req: DialogueRequest) -> dict[str, Any]:
    """NPC 对话"""
    result = await get_dialogue_service().process_dialogue(req)
    if "error" in result:
        raise HTTPException(status_code=result.get("code", 400), detail=result["error"])
    return result


@app.websocket("/ws/{session_id}")
async def websocket_dialogue(websocket: WebSocket, session_id: str) -> None:
    """
    WebSocket 多轮对话。
    1. await websocket.accept()
    2. 循环接收 JSON：{"npc_id": "...", "message": "...", "choice_id": "..."}
    3. 构造 DialogueRequest，调用 process_dialogue
    4. 发送 JSON 结果
    5. 客户端发送 {"action": "close"} 或断开时退出
    """
    await websocket.accept()
    try:
        while True:
            raw = await websocket.receive_text()
            data = json.loads(raw)
            if data.get("action") == "close":
                await websocket.close()
                return
            req = DialogueRequest(
                session_id=session_id,
                npc_id=data["npc_id"],
                message=data["message"],
                choice_id=data.get("choice_id"),
            )
            result = await get_dialogue_service().process_dialogue(req)
            if "error" in result:
                await websocket.send_json(
                    {
                        "type": "error",
                        "error": result["error"],
                        "code": result.get("code"),
                    }
                )
            else:
                await websocket.send_json(result)
    except WebSocketDisconnect:
        logger.info("WebSocket 客户端断开连接: %s", session_id)
    except Exception as exc:
        logger.warning("WebSocket 连接异常: %s", exc)
        try:
            await websocket.send_json({"type": "error", "error": str(exc)})
        except Exception:
            pass


@app.get("/sessions/{session_id}")
async def get_session(session_id: str) -> dict[str, Any]:
    state = get_dialogue_service().get_session_state(session_id)
    if not state:
        raise HTTPException(status_code=404, detail="会话不存在或已过期")
    return state


@app.post("/tests/run")
async def run_tests(req: TestRunRequest) -> dict[str, Any]:
    """运行剧情覆盖测试"""
    result = await get_agent_tester().run(req)
    return result.model_dump()


@app.get("/npcs")
async def list_npcs() -> dict[str, Any]:
    with open(config.NPCS_PATH, "r", encoding="utf-8") as f:
        return json.load(f)


@app.get("/story/graph")
async def get_story_graph() -> dict[str, Any]:
    with open(config.STORY_GRAPH_PATH, "r", encoding="utf-8") as f:
        return json.load(f)


@app.get("/health")
async def health_check() -> dict[str, Any]:
    return {
        "status": "ok",
        "mock_mode": get_llm_client().use_mock,
        "version": "2.0.0",
    }
