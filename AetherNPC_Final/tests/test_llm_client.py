"""LLM 客户端测试：Mock 分支、JSON 解析、嵌入向量与失败降级。"""

from __future__ import annotations

import asyncio
import math

import httpx
import pytest

import app.config as config
import app.llm_client as llm_module
from app.llm_client import LLMClient, get_llm_client

SYSTEM_PROMPT = (
    "你是一位名为「城门守卫巴顿」的 NPC。\n"
    "【世界观知识】\n（无）\n"
    "【当前剧情】\n你抵达银月城城门。\n"
    "【可用选项】\n"
    "  [ask_guard] 询问情况\n"
    "  [sneak_past] 偷偷进城\n"
)

CHOICE_PROMPT = (
    "你是一位名为「城门守卫巴顿」的 NPC。\n"
    "【当前剧情】\n当前节点：city_gate\n银月城城门巍峨。\n"
    "【刚刚发生的动作】玩家在节点 city_gate 选择了选项「ask_guard」：上前向守卫询问情况\n"
    "【可用选项】\n  [ask_guard] 询问情况\n"
)

END_PROMPT = (
    "你是一位名为「城门守卫巴顿」的 NPC。\n"
    "【当前剧情】\n当前节点：ending_hero\n你取回虚空之钥并加固封印，银月城恢复宁静。\n"
    "【可用选项】\n（当前无可用选项）\n"
)

LIBRARY_PROMPT = (
    "你是一位名为「伊莲」的 NPC。\n"
    "【当前剧情】\n当前节点：library\n月神图书馆穹顶高悬。\n"
    "【刚刚发生的动作】玩家在节点 library 选择了选项「read_history」：借阅第一纪元的历史卷轴\n"
    "【可用选项】\n  [read_history] 借阅历史卷轴\n"
)


@pytest.fixture(autouse=True)
def _reset_llm_singleton():
    llm_module._llm_client = None
    yield
    llm_module._llm_client = None


def _mock_client(monkeypatch) -> LLMClient:
    monkeypatch.setattr(config, "OPENAI_API_KEY", "")
    monkeypatch.setattr(config, "USE_MOCK", True)
    return LLMClient()


def test_mock_chat_keywords(monkeypatch) -> None:
    client = _mock_client(monkeypatch)
    response = asyncio.run(client.chat_completion(SYSTEM_PROMPT, "进城调查失踪商队"))
    assert "城门守卫" in response.dialogue
    assert response.emotion == "suspicious"
    assert response.selected_choice_id == "ask_guard"
    assert response.memory_updates == ["玩家对失踪商队感兴趣"]
    assert "knowledge_001" in response.cited_knowledge_ids
    assert response.confidence == 0.9


def test_mock_chat_thanks(monkeypatch) -> None:
    client = _mock_client(monkeypatch)
    response = asyncio.run(client.chat_completion(SYSTEM_PROMPT, "谢谢你的帮助"))
    assert "不必谢我" in response.dialogue
    assert response.emotion == "neutral"
    assert response.confidence == 0.8


def test_mock_chat_other(monkeypatch) -> None:
    client = _mock_client(monkeypatch)
    response = asyncio.run(client.chat_completion(SYSTEM_PROMPT, "zzzz"))
    assert response.dialogue == "『嗯……我不太确定你在说什么。』"
    assert response.confidence == 0.5
    assert response.emotion in {"neutral", "suspicious", "surprised", "sad"}


def test_mock_chat_greeting(monkeypatch) -> None:
    client = _mock_client(monkeypatch)
    response = asyncio.run(client.chat_completion(SYSTEM_PROMPT, "你好"))
    assert "欢迎来到银月城" in response.dialogue
    assert "城门守卫巴顿" in response.dialogue
    assert response.confidence == 0.9


def test_mock_chat_who(monkeypatch) -> None:
    client = _mock_client(monkeypatch)
    response = asyncio.run(client.chat_completion(SYSTEM_PROMPT, "你是谁"))
    assert "城门守卫巴顿" in response.dialogue
    assert response.confidence == 0.9


def test_mock_chat_world(monkeypatch) -> None:
    client = _mock_client(monkeypatch)
    response = asyncio.run(client.chat_completion(SYSTEM_PROMPT, "银月城的历史"))
    assert "月长石" in response.dialogue
    assert "knowledge_001" in response.cited_knowledge_ids


def test_mock_chat_forest(monkeypatch) -> None:
    client = _mock_client(monkeypatch)
    response = asyncio.run(client.chat_completion(SYSTEM_PROMPT, "黑森林危险吗"))
    assert "黑森林" in response.dialogue
    assert "knowledge_005" in response.cited_knowledge_ids
    assert response.emotion == "suspicious"


def test_mock_chat_story(monkeypatch) -> None:
    client = _mock_client(monkeypatch)
    response = asyncio.run(client.chat_completion(SYSTEM_PROMPT, "商队线索是什么"))
    assert "黑森林" in response.dialogue
    assert "knowledge_004" in response.cited_knowledge_ids


def test_mock_choice_reply(monkeypatch) -> None:
    client = _mock_client(monkeypatch)
    response = asyncio.run(client.chat_completion(CHOICE_PROMPT, "我选择这个选项"))
    assert "巴顿" in response.dialogue
    assert "外乡人" in response.dialogue
    assert response.emotion == "neutral"


def test_mock_end_reply(monkeypatch) -> None:
    client = _mock_client(monkeypatch)
    response = asyncio.run(client.chat_completion(END_PROMPT, "你好"))
    assert "故事结束了" in response.dialogue
    assert response.emotion == "happy"


def test_mock_library_choice_reply(monkeypatch) -> None:
    client = _mock_client(monkeypatch)
    response = asyncio.run(client.chat_completion(LIBRARY_PROMPT, "我选择这个选项"))
    assert "第一纪元" in response.dialogue
    assert "knowledge_007" in response.cited_knowledge_ids


def test_mock_parses_suggested_choices(monkeypatch) -> None:
    client = _mock_client(monkeypatch)
    response = asyncio.run(client.chat_completion(SYSTEM_PROMPT, "你好"))
    assert response.suggested_choices == [
        {"choice_id": "ask_guard", "text": "询问情况"},
        {"choice_id": "sneak_past", "text": "偷偷进城"},
    ]


def test_mock_embedding_shape(monkeypatch) -> None:
    client = _mock_client(monkeypatch)
    embedding = asyncio.run(client.get_embedding("黑森林在哪里"))
    assert len(embedding) == config.VECTOR_DIM


def test_mock_embedding_normalized(monkeypatch) -> None:
    client = _mock_client(monkeypatch)
    first = asyncio.run(client.get_embedding("黑森林在哪里"))
    second = asyncio.run(client.get_embedding("黑森林在哪里"))
    assert first == second
    norm = math.sqrt(sum(value * value for value in first))
    assert norm == pytest.approx(1.0)


class _FakeResponse:
    def __init__(self, content: str) -> None:
        self._content = content

    def raise_for_status(self) -> None:
        return None

    def json(self) -> dict:
        return {"choices": [{"message": {"content": self._content}}]}


def _real_client(monkeypatch) -> LLMClient:
    monkeypatch.setattr(config, "OPENAI_API_KEY", "sk-test")
    monkeypatch.setattr(config, "USE_MOCK", False)
    return LLMClient()


def test_real_mode_parses_json(monkeypatch) -> None:
    payload = (
        '{"dialogue": "好的，走吧。", "emotion": "happy", "action": "挥手",'
        ' "selected_choice_id": "ask_guard", "memory_updates": ["memory.hello=true"],'
        ' "cited_knowledge_ids": ["knowledge_001"], "confidence": 0.95,'
        ' "suggested_choices": [{"choice_id": "ask_guard", "text": "询问情况"}]}'
    )

    async def _fake_post(self, *args, **kwargs):
        return _FakeResponse(payload)

    monkeypatch.setattr(httpx.AsyncClient, "post", _fake_post)
    client = _real_client(monkeypatch)
    response = asyncio.run(client.chat_completion(SYSTEM_PROMPT, "走"))
    assert response.dialogue == "好的，走吧。"
    assert response.emotion == "happy"
    assert response.confidence == 0.95
    assert response.selected_choice_id == "ask_guard"
    asyncio.run(client.close())


def test_real_mode_falls_back_on_error(monkeypatch) -> None:
    async def _boom(*args, **kwargs):
        raise httpx.ConnectError("network unavailable")

    monkeypatch.setattr(httpx.AsyncClient, "post", _boom)
    client = _real_client(monkeypatch)
    response = asyncio.run(client.chat_completion(SYSTEM_PROMPT, "zzzz"))
    assert response.dialogue == "『嗯……我不太确定你在说什么。』"
    assert response.confidence == 0.5
    asyncio.run(client.close())


def test_real_embedding_falls_back_on_error(monkeypatch) -> None:
    async def _boom(*args, **kwargs):
        raise httpx.ConnectError("network unavailable")

    monkeypatch.setattr(httpx.AsyncClient, "post", _boom)
    client = _real_client(monkeypatch)
    embedding = asyncio.run(client.get_embedding("测试文本"))
    assert len(embedding) == config.VECTOR_DIM
    asyncio.run(client.close())


def test_close_is_idempotent(monkeypatch) -> None:
    client = _real_client(monkeypatch)
    asyncio.run(client.close())
    asyncio.run(client.close())


def test_singleton(monkeypatch) -> None:
    _mock_client(monkeypatch)
    first = get_llm_client()
    second = get_llm_client()
    assert first is second
