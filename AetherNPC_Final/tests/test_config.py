"""配置中心测试：模块常量 + 运行时 Settings。"""

from __future__ import annotations

import app.config as config
import pytest
from pydantic import ValidationError

from app.config import BASE_DIR, Settings


def test_module_constants() -> None:
    assert config.USE_MOCK == (not config.OPENAI_API_KEY)
    assert config.BASE_DIR == BASE_DIR
    assert config.DB_PATH == config.DATA_DIR / "npc_memory.db"
    assert config.NPCS_PATH == config.DATA_DIR / "npcs.json"
    assert config.STORY_GRAPH_PATH == config.DATA_DIR / "story_graph.json"
    assert config.KNOWLEDGE_PATH == config.DATA_DIR / "knowledge.json"
    assert config.VECTOR_DIM == (64 if config.USE_MOCK else 1536)
    assert config.MAX_HISTORY_TURNS >= 1
    assert config.SESSION_TIMEOUT >= 1
    assert config.LOG_LEVEL
    assert config.DATA_DIR.is_dir()


def test_settings_defaults_reflect_constants() -> None:
    cfg = Settings()
    assert cfg.openai_api_key == config.OPENAI_API_KEY
    assert cfg.openai_model == config.OPENAI_MODEL
    assert cfg.openai_base_url == config.OPENAI_BASE_URL
    assert cfg.resolved_database_path == config.DB_PATH
    assert cfg.resolved_npcs_path == config.NPCS_PATH
    assert cfg.resolved_story_graph_path == config.STORY_GRAPH_PATH
    assert cfg.resolved_knowledge_path == config.KNOWLEDGE_PATH
    assert cfg.vector_dim == config.VECTOR_DIM
    assert cfg.max_history_turns == config.MAX_HISTORY_TURNS
    assert cfg.session_timeout == config.SESSION_TIMEOUT
    assert cfg.llm_mock_mode == config.USE_MOCK


def test_invalid_values_rejected() -> None:
    with pytest.raises(ValidationError):
        Settings(port=0)
    with pytest.raises(ValidationError):
        Settings(log_level="verbose")
    with pytest.raises(ValidationError):
        Settings(max_history_turns=0)
    with pytest.raises(ValidationError):
        Settings(vector_dim=0)


def test_mock_mode_derived_from_key() -> None:
    assert Settings(openai_api_key="").llm_mock_mode is True
    assert Settings(openai_api_key="sk-test").llm_mock_mode is False


def test_llm_api_url_from_base_url() -> None:
    cfg = Settings(openai_base_url="https://api.openai.com/v1/")
    assert cfg.llm_api_url == "https://api.openai.com/v1/chat/completions"


def test_resolved_paths_follow_data_dir(tmp_path) -> None:
    cfg = Settings(data_dir=tmp_path)
    assert cfg.resolved_database_path == tmp_path / "npc_memory.db"
    assert cfg.resolved_npcs_path == tmp_path / "npcs.json"
    assert cfg.resolved_story_graph_path == tmp_path / "story_graph.json"
    assert cfg.resolved_knowledge_path == tmp_path / "knowledge.json"
