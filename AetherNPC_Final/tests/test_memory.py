"""记忆管理器测试：会话生命周期、关系值、历史、超时与线程安全。"""

from __future__ import annotations

import time

import app.config as config
from app.memory import MemoryManager, get_memory_manager


def test_create_session() -> None:
    memory = MemoryManager()
    session_id = memory.create_session("player", "city_gate")
    assert session_id
    state = memory.get_session(session_id)
    assert state is not None
    assert state.player_id == "player"
    assert state.current_node == "city_gate"


def test_get_session_exists() -> None:
    memory = MemoryManager()
    session_id = memory.create_session("player", "city_gate")
    assert memory.get_session(session_id) is not None


def test_get_session_not_exists() -> None:
    assert MemoryManager().get_session("missing") is None


def test_get_session_expired() -> None:
    memory = MemoryManager()
    session_id = memory.create_session("player", "city_gate")
    state = memory.get_session(session_id)
    state.last_active -= config.SESSION_TIMEOUT + 1
    assert memory.get_session(session_id) is None


def test_update_relationship_bounds() -> None:
    memory = MemoryManager()
    session_id = memory.create_session("player", "city_gate")
    memory.update_relationship(session_id, "gate_guard", 150)
    assert memory.get_relationship(session_id, "gate_guard") == 100
    memory.update_relationship(session_id, "gate_guard", -250)
    assert memory.get_relationship(session_id, "gate_guard") == -100
    assert memory.get_relationship("missing", "gate_guard") == 0


def test_append_history_trimming(monkeypatch) -> None:
    monkeypatch.setattr(config, "MAX_HISTORY_TURNS", 2)
    memory = MemoryManager()
    session_id = memory.create_session("player", "city_gate")
    for index in range(6):
        memory.append_history(session_id, "user", f"m{index}")
    state = memory.get_session(session_id)
    assert len(state.history) == 4
    assert state.history[0]["content"] == "m2"
    assert state.history[-1]["content"] == "m5"


def test_append_history_meta_and_summary() -> None:
    memory = MemoryManager()
    session_id = memory.create_session("player", "city_gate")
    memory.append_history(session_id, "user", "  你好  ", {"npc_id": "gate_guard"})
    memory.append_history(session_id, "npc", "欢迎")
    state = memory.get_session(session_id)
    assert state.history[0]["content"] == "你好"
    assert state.history[0]["npc_id"] == "gate_guard"
    assert "timestamp" in state.history[0]
    assert memory.get_history_summary(session_id) == "玩家: 你好\nNPC: 欢迎"
    assert memory.get_history_summary("missing") == "（会话不存在）"


def test_memory_and_current_node() -> None:
    memory = MemoryManager()
    session_id = memory.create_session("player", "city_gate")
    memory.add_memory(session_id, "has_gold", True)
    memory.set_current_node(session_id, "tavern")
    state = memory.get_session(session_id)
    assert state.memory["has_gold"] is True
    assert state.current_node == "tavern"
    assert time.time() - state.last_active < 5


def test_cleanup_expired() -> None:
    memory = MemoryManager()
    old = memory.create_session("player", "city_gate")
    new = memory.create_session("player", "city_gate")
    state = memory.get_session(old)
    state.created_at -= config.SESSION_TIMEOUT + 1
    removed = memory.cleanup_expired()
    assert removed == [old]
    assert memory.get_session(new) is not None


def test_list_and_delete_sessions() -> None:
    memory = MemoryManager()
    first = memory.create_session("player", "city_gate")
    second = memory.create_session("player", "city_gate")
    assert set(memory.list_sessions()) == {first, second}
    assert memory.delete_session(first) is True
    assert memory.get_session(first) is None
    assert memory.delete_session(first) is False


def test_singleton() -> None:
    import app.memory as memory_module

    memory_module._memory_manager = None
    first = get_memory_manager()
    second = get_memory_manager()
    assert first is second
