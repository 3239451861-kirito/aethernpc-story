"""Pydantic 数据模型测试：严格遵循数据模型精确定义。"""

from __future__ import annotations

import pytest
from pydantic import ValidationError

from app.schemas import (
    ChatMessage,
    CreateSessionRequest,
    DialogueRequest,
    HealthResponse,
    KnowledgeDoc,
    MessageRole,
    NPCProfile,
    NPCResponse,
    SessionState,
    SessionSummary,
    StoryChoice,
    StoryGraph,
    TestRunRequest,
    TestRunResult,
)


def _profile(**overrides) -> NPCProfile:
    values = {
        "npc_id": "gate_guard",
        "name": "城门守卫巴顿",
        "personality": "严肃、忠诚、略带多疑。对陌生人保持警惕，但对证明自己价值的人会给予尊重。",
        "background": "在银月城守卫了十五年城门，见过太多阴谋和伪装。",
        "dialogue_style": "简短有力，常使用命令式语句，偶尔引用军规",
    }
    values.update(overrides)
    return NPCProfile(**values)


def test_npc_profile_round_trip() -> None:
    profile = _profile()
    restored = NPCProfile.model_validate(profile.model_dump())
    assert restored == profile
    assert profile.voice_tone == "neutral"
    assert profile.initial_relationship == 0


def test_npc_profile_defaults() -> None:
    profile = _profile()
    assert profile.voice_tone == "neutral"
    assert profile.initial_relationship == 0
    assert profile.knowledge_domains == []


def test_npc_profile_rejects_short_personality() -> None:
    with pytest.raises(ValidationError):
        _profile(personality="太短")


def test_npc_profile_rejects_bad_id() -> None:
    with pytest.raises(ValidationError):
        _profile(npc_id="Bad-ID")


def test_story_choice_next_node_optional() -> None:
    choice = StoryChoice(choice_id="end_it", text="结束")
    assert choice.next_node is None
    assert choice.preconditions == {}


def test_story_graph_round_trip() -> None:
    graph = StoryGraph.model_validate(
        {
            "start_node": "a",
            "nodes": {
                "a": {"node_id": "a", "description": "起点"},
                "b": {"node_id": "b", "description": "终点", "is_end": True},
            },
        }
    )
    assert graph.start_node == "a"
    assert graph.nodes["b"].is_end is True
    assert StoryGraph.model_validate(graph.model_dump()).start_node == "a"


def test_story_graph_deserialization_full() -> None:
    graph = StoryGraph.model_validate(
        {
            "start_node": "start",
            "nodes": {
                "start": {
                    "node_id": "start",
                    "description": "起点",
                    "choices": [
                        {
                            "choice_id": "go",
                            "text": "出发",
                            "preconditions": {"memory.has_weapon": True},
                            "next_node": "end",
                            "effects": {"memory.done": True},
                        }
                    ],
                    "is_end": False,
                },
                "end": {"node_id": "end", "description": "终点", "is_end": True},
            },
        }
    )
    assert graph.nodes["start"].choices[0].preconditions == {"memory.has_weapon": True}
    assert graph.nodes["end"].is_end is True


def test_dialogue_request_with_choice() -> None:
    request = DialogueRequest(
        session_id="s1",
        npc_id="gate_guard",
        message="  我接受  ",
        choice_id="ask_guard",
    )
    assert request.message == "我接受"
    assert request.choice_id == "ask_guard"


def test_dialogue_request_rejects_empty_message() -> None:
    with pytest.raises(ValidationError):
        DialogueRequest(session_id="s1", npc_id="gate_guard", message="   ")


def test_create_session_request() -> None:
    request = CreateSessionRequest(player_id="  player1  ")
    assert request.player_id == "player1"
    with pytest.raises(ValidationError):
        CreateSessionRequest(player_id="")


def test_npc_response_serialization() -> None:
    response = NPCResponse(
        dialogue="你好",
        emotion="happy",
        suggested_choices=[{"choice_id": "a", "text": "选项A"}],
    )
    data = response.model_dump()
    assert data["suggested_choices"] == [{"choice_id": "a", "text": "选项A"}]
    assert NPCResponse.model_validate(data).dialogue == "你好"


def test_npc_response_rejects_bad_emotion() -> None:
    with pytest.raises(ValidationError):
        NPCResponse(dialogue="你好", emotion="curious")


def test_npc_response_rejects_bad_choice_keys() -> None:
    with pytest.raises(ValidationError):
        NPCResponse(dialogue="你好", suggested_choices=[{"choice_id": "a"}])


def test_npc_response_confidence_out_of_range() -> None:
    with pytest.raises(ValidationError):
        NPCResponse(dialogue="你好", confidence=1.5)
    with pytest.raises(ValidationError):
        NPCResponse(dialogue="你好", confidence=-0.1)


def test_session_state_excludes_coverage() -> None:
    state = SessionState(
        session_id="s1",
        player_id="p1",
        current_node="city_gate",
        created_at=1.0,
        last_active=1.0,
    )
    state.covered_nodes.add("city_gate")
    state.covered_edges.add(("city_gate", "ask_guard", "guard_dialogue"))
    data = state.model_dump()
    assert "covered_nodes" not in data
    assert "covered_edges" not in data
    assert data["current_node"] == "city_gate"


def test_knowledge_doc_embedding_optional() -> None:
    doc = KnowledgeDoc(doc_id="d1", content="内容")
    assert doc.embedding is None
    doc_with_embedding = KnowledgeDoc(doc_id="d2", content="内容", embedding=[0.1, 0.2])
    assert doc_with_embedding.embedding == [0.1, 0.2]


def test_test_run_request_bounds() -> None:
    assert TestRunRequest().target_coverage == 0.8
    with pytest.raises(ValidationError):
        TestRunRequest(target_coverage=1.5)
    with pytest.raises(ValidationError):
        TestRunRequest(max_steps=0)


def test_test_run_result_round_trip() -> None:
    result = TestRunResult(
        node_coverage=0.5,
        edge_coverage=0.4,
        total_nodes=2,
        covered_nodes=1,
        total_edges=4,
        covered_edges=2,
        steps_taken=3,
        reached_target=False,
    )
    assert result.path_trace == []
    assert TestRunResult.model_validate(result.model_dump()).node_coverage == 0.5


def test_remaining_models_round_trip() -> None:
    message = ChatMessage(role=MessageRole.USER, content="你好")
    assert ChatMessage.model_validate(message.model_dump()) == message
    health = HealthResponse(
        status="ok",
        app="AetherNPC",
        version="2.0.0",
        mock_mode=True,
        timestamp="2026-01-01T00:00:00+00:00",
    )
    assert HealthResponse.model_validate(health.model_dump()) == health
    summary = SessionSummary(
        session_id="s1",
        message_count=2,
        last_message="回复",
        updated_at="2026-01-01T00:00:00+00:00",
    )
    assert SessionSummary.model_validate(summary.model_dump()) == summary
