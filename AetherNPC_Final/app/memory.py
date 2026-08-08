"""会话记忆管理：上下文、关系值、历史、线程安全"""

import time
import threading
import logging
from typing import Any
from uuid import uuid4

from app import config
from app.schemas import SessionState
from app.prompts import build_history_summary

logger = logging.getLogger(__name__)


class MemoryManager:
    def __init__(self) -> None:
        self._sessions: dict[str, SessionState] = {}
        self._lock = threading.Lock()

    def create_session(
        self,
        player_id: str,
        start_node: str,
        session_id: str | None = None,
    ) -> str:
        """
        创建会话。
        1. 默认生成 uuid4 作为 session_id（也可显式传入，兼容客户端会话）
        2. SessionState(created_at=time.time(), last_active=time.time(), ...)
        3. 加锁写入 _sessions
        4. 返回 session_id
        """
        now = time.time()
        resolved_id = session_id or str(uuid4())
        state = SessionState(
            session_id=resolved_id,
            player_id=player_id,
            current_node=start_node,
            created_at=now,
            last_active=now,
        )
        with self._lock:
            self._sessions[resolved_id] = state
        return resolved_id

    def get_session(self, session_id: str) -> SessionState | None:
        """
        获取会话。
        1. 加锁读取
        2. 不存在 → None
        3. 存在但 time.time() - last_active > SESSION_TIMEOUT → 删除并返回 None
        4. 否则更新 last_active = time.time() 并返回
        """
        with self._lock:
            state = self._sessions.get(session_id)
            if state is None:
                return None
            if time.time() - state.last_active > config.SESSION_TIMEOUT:
                self._sessions.pop(session_id, None)
                return None
            state.last_active = time.time()
            return state

    def update_relationship(self, session_id: str, npc_id: str, delta: int) -> None:
        """
        更新关系值。
        1. 获取 session
        2. new_val = current + delta，限制在 [-100, 100]
        3. 加锁更新
        """
        with self._lock:
            state = self._sessions.get(session_id)
            if state is None:
                return
            current = state.relationship_values.get(npc_id, 0)
            state.relationship_values[npc_id] = max(-100, min(100, current + delta))
            state.last_active = time.time()

    def add_memory(self, session_id: str, key: str, value: Any) -> None:
        """加锁更新 session.memory[key] = value"""
        with self._lock:
            state = self._sessions.get(session_id)
            if state is None:
                return
            state.memory[key] = value
            state.last_active = time.time()

    def append_history(
        self,
        session_id: str,
        role: str,
        content: str,
        meta: dict[str, Any] | None = None,
    ) -> None:
        """
        追加历史。
        1. entry = {"role": role, "content": content.strip(), "timestamp": time.time()}
        2. 如果 meta 不为 None，合并到 entry
        3. 加锁 append 到 session.history
        4. 如果 len(history) > MAX_HISTORY_TURNS * 2，保留后 MAX_HISTORY_TURNS * 2 条
        """
        entry: dict[str, Any] = {
            "role": role,
            "content": content.strip(),
            "timestamp": time.time(),
        }
        if meta:
            entry.update(meta)
        with self._lock:
            state = self._sessions.get(session_id)
            if state is None:
                return
            state.history.append(entry)
            limit = config.MAX_HISTORY_TURNS * 2
            if len(state.history) > limit:
                del state.history[: len(state.history) - limit]
            state.last_active = time.time()

    def get_history_summary(self, session_id: str, max_turns: int = 5) -> str:
        """
        获取历史摘要。
        1. 获取 session
        2. 调用 build_history_summary(session.history, max_turns)
        3. 如果 session 不存在返回 "（会话不存在）"
        """
        state = self.get_session(session_id)
        if state is None:
            return "（会话不存在）"
        return build_history_summary(state.history, max_turns)

    def set_current_node(self, session_id: str, node_id: str) -> None:
        """更新 current_node 和 last_active，加锁"""
        with self._lock:
            state = self._sessions.get(session_id)
            if state is None:
                return
            state.current_node = node_id
            state.last_active = time.time()

    def get_relationship(self, session_id: str, npc_id: str) -> int:
        """获取关系值，session 不存在返回 0"""
        with self._lock:
            state = self._sessions.get(session_id)
            if state is None:
                return 0
            return state.relationship_values.get(npc_id, 0)

    def cleanup_expired(self) -> list[str]:
        """
        清理超时会话。
        1. 遍历所有 session
        2. time.time() - created_at > SESSION_TIMEOUT 的删除
        3. 返回被删除的 session_id 列表
        """
        removed: list[str] = []
        now = time.time()
        with self._lock:
            for session_id, state in list(self._sessions.items()):
                if now - state.created_at > config.SESSION_TIMEOUT:
                    self._sessions.pop(session_id, None)
                    removed.append(session_id)
        return removed

    def list_sessions(self) -> list[str]:
        """返回所有 session_id"""
        with self._lock:
            return list(self._sessions)

    def delete_session(self, session_id: str) -> bool:
        """删除会话（供 API 删除接口使用）。"""
        with self._lock:
            return self._sessions.pop(session_id, None) is not None


# 全局单例
_memory_manager: MemoryManager | None = None


def get_memory_manager() -> MemoryManager:
    global _memory_manager
    if _memory_manager is None:
        _memory_manager = MemoryManager()
    return _memory_manager
