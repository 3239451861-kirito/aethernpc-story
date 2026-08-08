"""剧情引擎测试：加载校验、选项校验、效果与图分析（银月城剧情图）。"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

import app.config as config
from app.schemas import StoryGraph
from app.story_engine import StoryEngine, get_story_engine


def _write_json(path: Path, data: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as handle:
        json.dump(data, handle, ensure_ascii=False, indent=2)


def test_load_valid_graph() -> None:
    engine = StoryEngine()
    assert engine.story_graph is not None
    assert engine.get_start_node() in engine.get_all_nodes()


def test_get_node_exists() -> None:
    node = StoryEngine().get_node("city_gate")
    assert node is not None
    assert node.node_id == "city_gate"


def test_get_node_not_exists() -> None:
    assert StoryEngine().get_node("ghost") is None


def test_validate_choice_valid() -> None:
    engine = StoryEngine()
    valid, next_node, error = engine.validate_choice(
        "city_gate",
        "ask_guard",
        {"memory": {}, "relationship_values": {"gate_guard": 0}},
    )
    assert valid is True
    assert next_node == "guard_dialogue"
    assert error == ""


def test_validate_choice_not_exists() -> None:
    engine = StoryEngine()
    valid, _, error = engine.validate_choice(
        "city_gate",
        "ghost_choice",
        {"memory": {}, "relationship_values": {}},
    )
    assert valid is False
    assert "选项 ghost_choice 不属于节点 city_gate" in error


def test_validate_choice_precondition_memory() -> None:
    engine = StoryEngine()
    state = {"memory": {}, "relationship_values": {"gate_guard": 0}}
    valid, _, error = engine.validate_choice("city_gate", "bribe_guard", state)
    assert valid is False
    assert "memory.has_gold" in error
    state["memory"]["has_gold"] = True
    valid, next_node, _ = engine.validate_choice("city_gate", "bribe_guard", state)
    assert valid is True
    assert next_node == "guard_bribed"


def test_validate_choice_precondition_relationship() -> None:
    engine = StoryEngine()
    state = {"memory": {}, "relationship_values": {"gate_guard": 9}}
    valid, _, error = engine.validate_choice("city_gate", "show_badge", state)
    assert valid is False
    assert "relationship.gate_guard" in error
    state["relationship_values"]["gate_guard"] = 10
    valid, next_node, _ = engine.validate_choice("city_gate", "show_badge", state)
    assert valid is True
    assert next_node == "city_plaza"


def test_validate_choice_next_node_not_exists() -> None:
    engine = StoryEngine()
    engine.story_graph = StoryGraph.model_validate(
        {
            "start_node": "a",
            "nodes": {
                "a": {
                    "node_id": "a",
                    "description": "起点",
                    "choices": [
                        {"choice_id": "go", "text": "走", "next_node": "ghost"}
                    ],
                }
            },
        }
    )
    valid, _, error = engine.validate_choice(
        "a",
        "go",
        {"memory": {}, "relationship_values": {}},
    )
    assert valid is False
    assert "目标节点不存在: ghost" in error


def test_apply_effects_memory() -> None:
    state = {"memory": {}, "relationship_values": {}}
    StoryEngine().apply_effects(state, {"memory.mission": True})
    assert state["memory"]["mission"] is True


def test_apply_effects_relationship_bounds() -> None:
    engine = StoryEngine()
    state = {"memory": {}, "relationship_values": {"gate_guard": 90}}
    engine.apply_effects(state, {"relationship.gate_guard": 50})
    assert state["relationship_values"]["gate_guard"] == 100
    engine.apply_effects(state, {"relationship.gate_guard": -300})
    assert state["relationship_values"]["gate_guard"] == -100


def test_find_dead_ends() -> None:
    dead_ends = StoryEngine().find_dead_ends()
    assert set(dead_ends) == {"ending_hero", "ending_betrayal", "ending_rescuer"}


def test_find_unreachable() -> None:
    engine = StoryEngine()
    assert engine.find_unreachable_from("city_gate") == []
    assert set(engine.find_unreachable_from("ghost")) == set(engine.get_all_nodes())


def test_graph_integrity() -> None:
    assert StoryEngine().validate_graph_integrity() == []


def test_get_all_nodes_and_edges() -> None:
    engine = StoryEngine()
    assert len(engine.get_all_nodes()) == 36
    assert len(engine.get_all_edges()) == 76
    assert ("city_gate", "ask_guard", "guard_dialogue") in engine.get_all_edges()
    assert ("city_gate", "bribe_guard", "guard_bribed") in engine.get_all_edges()
    assert ("city_gate", "show_badge", "city_plaza") in engine.get_all_edges()
    assert ("city_plaza", "visit_library", "library") in engine.get_all_edges()
    assert ("city_plaza", "visit_temple", "temple") in engine.get_all_edges()
    assert ("city_plaza", "visit_mine", "mine") in engine.get_all_edges()
    assert ("forest", "visit_ranger", "ranger_hut") in engine.get_all_edges()
    assert ("library", "read_history", "library_lore") in engine.get_all_edges()
    assert ("library_map", "follow_map", "ruins") in engine.get_all_edges()
    assert ("mine_rumors", "buy_ore", "mine_ledger") in engine.get_all_edges()


def test_new_branch_nodes_exist() -> None:
    engine = StoryEngine()
    for node_id in (
        "library",
        "library_lore",
        "library_map",
        "temple",
        "temple_ritual",
        "temple_lore",
        "mine",
        "mine_ledger",
        "mine_rumors",
        "ranger_hut",
    ):
        assert engine.get_node(node_id) is not None


def test_missing_graph_raises(tmp_path, monkeypatch) -> None:
    monkeypatch.setattr(config, "STORY_GRAPH_PATH", tmp_path / "missing.json")
    with pytest.raises(FileNotFoundError):
        StoryEngine()


def test_load_rejects_bad_next_node(tmp_path, monkeypatch) -> None:
    graph = {
        "start_node": "a",
        "nodes": {
            "a": {
                "node_id": "a",
                "description": "起点",
                "choices": [{"choice_id": "go", "text": "走", "next_node": "ghost"}],
            }
        },
    }
    path = tmp_path / "bad_graph.json"
    _write_json(path, graph)
    monkeypatch.setattr(config, "STORY_GRAPH_PATH", path)
    with pytest.raises(ValueError):
        StoryEngine()


def test_load_rejects_key_mismatch(tmp_path, monkeypatch) -> None:
    graph = {
        "start_node": "a",
        "nodes": {
            "a": {"node_id": "b", "description": "起点", "choices": []},
        },
    }
    path = tmp_path / "bad_graph.json"
    _write_json(path, graph)
    monkeypatch.setattr(config, "STORY_GRAPH_PATH", path)
    with pytest.raises(ValueError):
        StoryEngine()


def test_load_rejects_missing_start(tmp_path, monkeypatch) -> None:
    graph = {
        "start_node": "ghost",
        "nodes": {"a": {"node_id": "a", "description": "起点", "choices": []}},
    }
    path = tmp_path / "bad_graph.json"
    _write_json(path, graph)
    monkeypatch.setattr(config, "STORY_GRAPH_PATH", path)
    with pytest.raises(ValueError):
        StoryEngine()


def test_singleton(tmp_path, monkeypatch) -> None:
    import app.story_engine as story_module

    graph = {
        "start_node": "a",
        "nodes": {"a": {"node_id": "a", "description": "起点", "choices": []}},
    }
    path = tmp_path / "ok_graph.json"
    _write_json(path, graph)
    monkeypatch.setattr(config, "STORY_GRAPH_PATH", path)
    story_module._story_engine = None
    first = get_story_engine()
    second = get_story_engine()
    assert first is second
