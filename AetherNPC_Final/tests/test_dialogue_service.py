"""对话服务测试：process_dialogue 完整流程（信封式响应）。"""

from __future__ import annotations

import asyncio

import httpx
import pytest

import app.config as config
from app.prompts import build_history_summary, build_npc_system_prompt
from app.schemas import DialogueRequest, StoryChoice
from app.services.dialogue_service import DialogueService, get_dialogue_service


def _request(
    session_id: str = "s1",
    npc_id: str = "gate_guard",
    message: str = "你好",
    choice_id: str | None = None,
) -> DialogueRequest:
    return DialogueRequest(
        session_id=session_id,
        npc_id=npc_id,
        message=message,
        choice_id=choice_id,
    )


def _new_session(service) -> str:
    return service.create_session("player")


def test_create_session(service) -> None:
    session_id = _new_session(service)
    assert session_id
    assert service.get_session_state(session_id) is not None


def test_process_dialogue_mock(service) -> None:
    session_id = _new_session(service)
    result = asyncio.run(
        service.process_dialogue(
            _request(session_id=session_id, message="谢谢你的帮助")
        )
    )
    assert result["session_id"] == session_id
    assert result["npc_id"] == "gate_guard"
    assert result["current_node"] == "city_gate"
    assert "不必谢我" in result["response"]["dialogue"]
    assert result["response"]["emotion"] == "neutral"
    assert result["validated_choice"] is None
    assert result["relationship"] == 0
    assert result["response"]["suggested_choices"]


def test_process_dialogue_invalid_choice(service) -> None:
    session_id = _new_session(service)
    result = asyncio.run(
        service.process_dialogue(
            _request(session_id=session_id, message="随便", choice_id="ghost_choice")
        )
    )
    assert result["validated_choice"] is None
    assert result["current_node"] == "city_gate"


def test_process_dialogue_invalid_session(service) -> None:
    result = asyncio.run(service.process_dialogue(_request(session_id="missing")))
    assert result["error"] == "会话不存在或已过期"
    assert result["code"] == 404


def test_process_dialogue_invalid_npc(service) -> None:
    session_id = _new_session(service)
    result = asyncio.run(
        service.process_dialogue(_request(session_id=session_id, npc_id="ghost"))
    )
    assert result["error"] == "NPC ghost 不存在"
    assert result["code"] == 404


def test_relationship_update(service) -> None:
    session_id = _new_session(service)
    asyncio.run(
        service.process_dialogue(
            _request(session_id=session_id, message="谢谢", choice_id="ask_guard")
        )
    )
    result = asyncio.run(
        service.process_dialogue(
            _request(session_id=session_id, message="进城调查失踪商队")
        )
    )
    assert result["response"]["emotion"] == "suspicious"
    # ask_guard 在 guard_dialogue 节点非法，不推进剧情；suspicious 情感 -2：2 - 2 = 0
    assert result["validated_choice"] is None
    assert result["current_node"] == "guard_dialogue"
    assert result["relationship"] == 0


def test_choice_hint_advances_story(service) -> None:
    session_id = _new_session(service)
    result = asyncio.run(
        service.process_dialogue(
            _request(session_id=session_id, message="谢谢", choice_id="ask_guard")
        )
    )
    assert result["validated_choice"] == "ask_guard"
    assert result["current_node"] == "guard_dialogue"
    assert result["relationship"] == 2
    assert "巴顿" in result["response"]["dialogue"]
    state = service.get_session_state(session_id)
    assert state["relationship_values"]["gate_guard"] == 2


def test_choice_dialogue_matches_world(service) -> None:
    session_id = _new_session(service)
    result = asyncio.run(
        service.process_dialogue(
            _request(session_id=session_id, message="我选择这个选项", choice_id="ask_guard")
        )
    )
    assert "外乡人" in result["response"]["dialogue"]
    result = asyncio.run(
        service.process_dialogue(
            _request(session_id=session_id, message="我选择这个选项", choice_id="ask_caravan")
        )
    )
    assert "黑森林" in result["response"]["dialogue"]
    assert "knowledge_004" in result["response"]["cited_knowledge_ids"]


def test_ending_dialogue(service) -> None:
    session_id = _new_session(service)
    service.memory.set_current_node(session_id, "ending_hero")
    result = asyncio.run(
        service.process_dialogue(_request(session_id=session_id, message="你好"))
    )
    assert "故事结束了" in result["response"]["dialogue"]
    assert result["response"]["emotion"] == "happy"


def test_library_branch_dialogue(service) -> None:
    session_id = _new_session(service)
    asyncio.run(
        service.process_dialogue(
            _request(session_id=session_id, message="我选择这个选项", choice_id="ask_guard")
        )
    )
    asyncio.run(
        service.process_dialogue(
            _request(session_id=session_id, message="我选择这个选项", choice_id="salute")
        )
    )
    result = asyncio.run(
        service.process_dialogue(
            _request(session_id=session_id, message="我选择这个选项", choice_id="visit_library")
        )
    )
    assert "图书馆" in result["response"]["dialogue"]
    assert result["current_node"] == "library"
    result = asyncio.run(
        service.process_dialogue(
            _request(session_id=session_id, message="我选择这个选项", choice_id="read_history")
        )
    )
    assert "第一纪元" in result["response"]["dialogue"]
    assert "knowledge_007" in result["response"]["cited_knowledge_ids"]


def test_temple_branch_dialogue(service) -> None:
    session_id = _new_session(service)
    for choice_id in ("ask_guard", "salute", "visit_temple"):
        asyncio.run(
            service.process_dialogue(
                _request(session_id=session_id, message="我选择这个选项", choice_id=choice_id)
            )
        )
    result = asyncio.run(
        service.process_dialogue(
            _request(session_id=session_id, message="我选择这个选项", choice_id="pray_blessing")
        )
    )
    assert "月光" in result["response"]["dialogue"]
    assert result["current_node"] == "temple_ritual"


def test_precondition_failure_ignored(service) -> None:
    session_id = _new_session(service)
    asyncio.run(
        service.process_dialogue(
            _request(session_id=session_id, message="谢谢", choice_id="ask_guard")
        )
    )
    result = asyncio.run(
        service.process_dialogue(
            _request(session_id=session_id, message="塞金币", choice_id="bribe_guard")
        )
    )
    assert result["validated_choice"] is None
    assert result["current_node"] == "guard_dialogue"


def test_rag_domain_filter_through_service(service) -> None:
    session_id = _new_session(service)
    result = asyncio.run(
        service.process_dialogue(
            _request(session_id=session_id, message="黑森林在哪里")
        )
    )
    assert "knowledge_005" in result["cited_knowledge"]


def test_llm_memory_updates_stored(service) -> None:
    session_id = _new_session(service)
    result = asyncio.run(
        service.process_dialogue(
            _request(session_id=session_id, message="进城调查失踪商队")
        )
    )
    assert result["response"]["memory_updates"] == ["玩家对失踪商队感兴趣"]
    state = service.get_session_state(session_id)
    assert state["memory"]["玩家对失踪商队感兴趣"] is True


def test_history_recorded_with_meta(service) -> None:
    session_id = _new_session(service)
    asyncio.run(
        service.process_dialogue(
            _request(session_id=session_id, message="谢谢", choice_id="ask_guard")
        )
    )
    state = service.get_session_state(session_id)
    assert state["history_count"] == 2
    session = service.memory.get_session(session_id)
    assert session.history[0]["role"] == "user"
    assert session.history[0]["npc_id"] == "gate_guard"
    assert session.history[1]["role"] == "npc"
    assert session.history[1]["validated_choice"] == "ask_guard"
    assert session.history[1]["emotion"] == "neutral"


def test_get_session_state_snapshot(service) -> None:
    session_id = _new_session(service)
    state = service.get_session_state(session_id)
    assert set(state) == {
        "session_id",
        "player_id",
        "current_node",
        "relationship_values",
        "memory",
        "history_count",
        "covered_nodes",
        "covered_edges",
    }
    assert state["covered_nodes"] == []
    assert service.get_session_state("missing") is None


def test_delete_session(service) -> None:
    session_id = _new_session(service)
    assert service.delete_session(session_id) is True
    assert service.get_session_state(session_id) is None
    assert service.delete_session(session_id) is False


def test_singleton(service) -> None:
    assert get_dialogue_service() is service


def test_load_npcs_rejects_bad_file(tmp_path, monkeypatch) -> None:
    bad_path = tmp_path / "bad_npcs.json"
    bad_path.write_text("{不是合法JSON", encoding="utf-8")
    monkeypatch.setattr(config, "NPCS_PATH", bad_path)
    with pytest.raises(ValueError):
        DialogueService()


def test_load_npcs_rejects_missing_list(tmp_path, monkeypatch) -> None:
    bad_path = tmp_path / "empty_npcs.json"
    bad_path.write_text("{}", encoding="utf-8")
    monkeypatch.setattr(config, "NPCS_PATH", bad_path)
    with pytest.raises(ValueError):
        DialogueService()


class _JsonFakeResponse:
    def __init__(self, content: str) -> None:
        self._content = content

    def raise_for_status(self) -> None:
        return None

    def json(self) -> dict:
        return {"choices": [{"message": {"content": self._content}}]}


def test_real_json_reply_is_parsed(monkeypatch) -> None:
    monkeypatch.setattr(config, "OPENAI_API_KEY", "sk-test")
    monkeypatch.setattr(config, "USE_MOCK", False)
    payload = (
        '{"dialogue": "好的，走吧。", "emotion": "happy", "action": "挥手",'
        ' "selected_choice_id": null, "memory_updates": ["memory.hello=true"],'
        ' "cited_knowledge_ids": [], "confidence": 0.95, "suggested_choices": []}'
    )

    async def _fake_post(self, *args, **kwargs):
        return _JsonFakeResponse(payload)

    monkeypatch.setattr(httpx.AsyncClient, "post", _fake_post)
    service = get_dialogue_service()
    session_id = _new_session(service)
    result = asyncio.run(
        service.process_dialogue(_request(session_id=session_id, message="走"))
    )
    assert result["response"]["dialogue"] == "好的，走吧。"
    assert result["response"]["emotion"] == "happy"
    assert result["response"]["confidence"] == 0.95


def test_real_client_failure_degrades_to_mock(monkeypatch) -> None:
    monkeypatch.setattr(config, "OPENAI_API_KEY", "sk-test")
    monkeypatch.setattr(config, "USE_MOCK", False)

    async def _boom(*args, **kwargs):
        raise httpx.ConnectError("no network")

    monkeypatch.setattr(httpx.AsyncClient, "post", _boom)
    service = get_dialogue_service()
    session_id = _new_session(service)
    result = asyncio.run(
        service.process_dialogue(_request(session_id=session_id, message="zzzz"))
    )
    assert "我不太确定" in result["response"]["dialogue"]
    assert result["response"]["confidence"] == 0.5


def test_malformed_llm_json_falls_back(monkeypatch) -> None:
    monkeypatch.setattr(config, "OPENAI_API_KEY", "sk-test")
    monkeypatch.setattr(config, "USE_MOCK", False)

    async def _fake_post(self, *args, **kwargs):
        return _JsonFakeResponse("就按你说的办。")

    monkeypatch.setattr(httpx.AsyncClient, "post", _fake_post)
    service = get_dialogue_service()
    session_id = _new_session(service)
    result = asyncio.run(
        service.process_dialogue(_request(session_id=session_id, message="走"))
    )
    assert "我不太确定" in result["response"]["dialogue"]


def test_build_npc_system_prompt_contains_all_sections(service) -> None:
    npc = service.npcs["gate_guard"]
    prompt = build_npc_system_prompt(
        npc=npc,
        story_ctx="当前节点：city_gate\n银月城城门巍峨。",
        knowledge_ctx="- 《知识001》银月城建于第一纪元。",
        history_summary="玩家: 你好\nNPC: 欢迎",
        available_choices=[StoryChoice(choice_id="ask_guard", text="询问情况")],
    )
    for section in (
        "【角色设定】",
        "【世界观知识】",
        "【当前剧情】",
        "【对话历史摘要】",
        "【可用选项】",
        "【输出规则 - 严格遵守】",
        "ask_guard",
        '"dialogue"',
    ):
        assert section in prompt
    assert "你是一位名为「" in prompt
    assert "不要提及你是 AI" in prompt


def test_build_history_summary() -> None:
    assert build_history_summary([]) == "（无对话历史）"
    summary = build_history_summary(
        [
            {"role": "user", "content": "你好"},
            {"role": "npc", "content": "欢迎"},
            {"role": "user", "content": "再见"},
        ],
        max_turns=5,
    )
    assert summary == "玩家: 你好\nNPC: 欢迎\n玩家: 再见"


def test_build_history_summary_limits_turns() -> None:
    history = [
        item
        for index in range(5)
        for item in (
            {"role": "user", "content": f"m{index}"},
            {"role": "npc", "content": f"r{index}"},
        )
    ]
    summary = build_history_summary(history, max_turns=2)
    assert "m0" not in summary
    assert summary == "玩家: m3\nNPC: r3\n玩家: m4\nNPC: r4"
