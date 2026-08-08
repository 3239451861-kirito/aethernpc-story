"""配置中心：自动检测 Mock 模式，创建必要目录"""

import os
import logging
from pathlib import Path
from dotenv import load_dotenv

# 加载 .env
_ENV_PATH = Path(__file__).resolve().parent.parent / ".env"
load_dotenv(_ENV_PATH)

# OpenAI 配置
OPENAI_API_KEY: str = os.getenv("OPENAI_API_KEY", "").strip()
OPENAI_MODEL: str = os.getenv("OPENAI_MODEL", "gpt-4o-mini").strip()
OPENAI_BASE_URL: str = os.getenv("OPENAI_BASE_URL", "https://api.openai.com/v1").strip().rstrip("/")

# 运行模式
USE_MOCK: bool = not OPENAI_API_KEY

# 路径配置
BASE_DIR: Path = Path(__file__).resolve().parent.parent
DATA_DIR: Path = BASE_DIR / "data"
DB_PATH: Path = DATA_DIR / "npc_memory.db"
NPCS_PATH: Path = DATA_DIR / "npcs.json"
STORY_GRAPH_PATH: Path = DATA_DIR / "story_graph.json"
KNOWLEDGE_PATH: Path = DATA_DIR / "knowledge.json"

# 自动创建目录
DATA_DIR.mkdir(parents=True, exist_ok=True)

# 应用参数
MAX_HISTORY_TURNS: int = int(os.getenv("MAX_HISTORY_TURNS", "10"))
SESSION_TIMEOUT: int = int(os.getenv("SESSION_TIMEOUT", "3600"))
VECTOR_DIM: int = 64 if USE_MOCK else 1536
LOG_LEVEL: str = os.getenv("LOG_LEVEL", "INFO")

# 日志配置
logging.basicConfig(
    level=getattr(logging, LOG_LEVEL.upper(), logging.INFO),
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)


# ---- 运行时配置对象（依赖注入 / 测试隔离，默认值全部来自上方常量）----

from pydantic import BaseModel, ConfigDict, Field


class Settings(BaseModel):
    """运行时配置：默认值来自配置中心常量，额外参数均经 Pydantic 校验。"""

    model_config = ConfigDict(extra="ignore")

    app_name: str = Field(default="AetherNPC", min_length=1, max_length=64)
    host: str = "127.0.0.1"
    port: int = Field(default=8000, ge=1, le=65535)
    debug: bool = False

    openai_api_key: str = OPENAI_API_KEY
    openai_model: str = OPENAI_MODEL
    openai_base_url: str = OPENAI_BASE_URL
    llm_timeout_seconds: float = Field(default=15.0, gt=0, le=300)
    llm_max_tokens: int = Field(default=512, ge=16, le=4096)
    llm_fallback_to_mock: bool = True

    max_history_turns: int = Field(default=MAX_HISTORY_TURNS, ge=1, le=200)
    session_timeout: int = Field(default=SESSION_TIMEOUT, ge=1, le=2_592_000)
    log_level: str = Field(default=LOG_LEVEL.upper(), pattern=r"^(DEBUG|INFO|WARNING|ERROR|CRITICAL)$")
    vector_dim: int = Field(default=VECTOR_DIM, ge=8, le=4096)

    rag_top_k: int = Field(default=3, ge=1, le=20)
    cors_origins: list[str] = ["*"]

    data_dir: Path = DATA_DIR
    static_dir: Path = BASE_DIR / "static"
    database_path: Path | None = None
    npcs_file: Path | None = None
    story_graph_file: Path | None = None
    knowledge_file: Path | None = None

    @property
    def resolved_database_path(self) -> Path:
        return self.database_path or self.data_dir / "npc_memory.db"

    @property
    def resolved_npcs_path(self) -> Path:
        return self.npcs_file or self.data_dir / "npcs.json"

    @property
    def resolved_story_graph_path(self) -> Path:
        return self.story_graph_file or self.data_dir / "story_graph.json"

    @property
    def resolved_knowledge_path(self) -> Path:
        return self.knowledge_file or self.data_dir / "knowledge.json"

    @property
    def llm_api_url(self) -> str:
        return self.openai_base_url.rstrip("/") + "/chat/completions"

    @property
    def llm_mock_mode(self) -> bool:
        return not bool(self.openai_api_key.strip())

    @property
    def max_history_messages(self) -> int:
        return self.max_history_turns * 2

    @classmethod
    def load(cls) -> "Settings":
        return cls()


def setup_logging(level: str | None = None) -> None:
    """统一日志格式：%(asctime)s [%(levelname)s] %(name)s: %(message)s"""
    normalized = (level or LOG_LEVEL).upper()
    logging.basicConfig(
        level=getattr(logging, normalized, logging.INFO),
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
        force=True,
    )
