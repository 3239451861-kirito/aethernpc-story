"""Pydantic v2 数据模型：严格遵循数据模型精确定义。"""

from __future__ import annotations

from enum import Enum
from typing import Any

from pydantic import BaseModel, ConfigDict, Field, field_validator


class NPCProfile(BaseModel):
    model_config = ConfigDict(extra="forbid", str_strip_whitespace=True)

    npc_id: str = Field(..., min_length=1, max_length=64, pattern=r"^[a-z_]+$", description="NPC唯一标识")
    name: str = Field(..., min_length=1, max_length=128)
    personality: str = Field(..., min_length=10, max_length=512)
    background: str = Field(..., min_length=10, max_length=1024)
    dialogue_style: str = Field(..., min_length=5, max_length=256)
    voice_tone: str = Field(default="neutral", pattern=r"^(neutral|happy|sad|angry|suspicious|mysterious|grumpy)$")
    knowledge_domains: list[str] = Field(default_factory=list, max_length=10)
    initial_relationship: int = Field(default=0, ge=-100, le=100)

    @field_validator("knowledge_domains")
    @classmethod
    def _clean_domains(cls, values: list[str]) -> list[str]:
        return [value.strip() for value in values if value.strip()]


class StoryChoice(BaseModel):
    model_config = ConfigDict(extra="forbid", str_strip_whitespace=True)

    choice_id: str = Field(..., min_length=1, max_length=64, pattern=r"^[a-z_0-9]+$")
    text: str = Field(..., min_length=1, max_length=256)
    preconditions: dict[str, Any] = Field(default_factory=dict, description="前置条件，如{'memory.has_gold': true, 'relationship.gate_guard': 10}")
    next_node: str | None = Field(default=None, description="目标节点ID，None表示终局")
    effects: dict[str, Any] = Field(default_factory=dict, description="选择后的效果")


class StoryNode(BaseModel):
    model_config = ConfigDict(extra="forbid", str_strip_whitespace=True)

    node_id: str = Field(..., min_length=1, max_length=64)
    description: str = Field(..., min_length=1, max_length=2048)
    choices: list[StoryChoice] = Field(default_factory=list, max_length=20)
    is_end: bool = Field(default=False, description="是否为终局节点")


class StoryGraph(BaseModel):
    model_config = ConfigDict(extra="forbid")

    start_node: str = Field(..., min_length=1)
    nodes: dict[str, StoryNode] = Field(..., min_length=1, description="node_id -> StoryNode 映射")


class DialogueRequest(BaseModel):
    model_config = ConfigDict(extra="forbid", str_strip_whitespace=True)

    session_id: str = Field(..., min_length=1, max_length=128)
    npc_id: str = Field(..., min_length=1, max_length=64)
    message: str = Field(..., min_length=1, max_length=2048)
    choice_id: str | None = Field(default=None, max_length=64)


class NPCResponse(BaseModel):
    model_config = ConfigDict(extra="forbid", str_strip_whitespace=True)

    dialogue: str = Field(..., min_length=1, max_length=2048, description="NPC回复文本")
    emotion: str = Field(default="neutral", pattern=r"^(neutral|happy|sad|angry|surprised|suspicious|fearful)$")
    action: str | None = Field(default=None, max_length=256, description="动作/表情")
    selected_choice_id: str | None = Field(default=None, description="模型建议的选项ID")
    memory_updates: list[str] = Field(default_factory=list, max_length=20)
    cited_knowledge_ids: list[str] = Field(default_factory=list, max_length=10)
    confidence: float = Field(default=1.0, ge=0.0, le=1.0)
    suggested_choices: list[dict[str, str]] = Field(default_factory=list, description="[{choice_id, text}]")

    @field_validator("suggested_choices")
    @classmethod
    def _validate_suggested_choices(cls, values: list[dict[str, str]]) -> list[dict[str, str]]:
        cleaned: list[dict[str, str]] = []
        for item in values:
            if set(item) != {"choice_id", "text"}:
                raise ValueError("suggested_choices 每项必须包含 choice_id 与 text")
            cleaned.append({"choice_id": item["choice_id"].strip(), "text": item["text"].strip()})
        return cleaned


class SessionState(BaseModel):
    model_config = ConfigDict(extra="forbid")

    session_id: str
    player_id: str
    current_node: str
    history: list[dict[str, Any]] = Field(default_factory=list)
    relationship_values: dict[str, int] = Field(default_factory=dict)
    memory: dict[str, Any] = Field(default_factory=dict)
    created_at: float = Field(..., description="创建时间戳")
    last_active: float = Field(..., description="最后活跃时间戳")
    covered_nodes: set[str] = Field(default_factory=set, exclude=True)
    covered_edges: set[tuple[str, str, str]] = Field(default_factory=set, exclude=True)


class KnowledgeDoc(BaseModel):
    model_config = ConfigDict(extra="forbid", str_strip_whitespace=True)

    doc_id: str = Field(..., min_length=1, max_length=64)
    content: str = Field(..., min_length=1, max_length=10000)
    metadata: dict[str, Any] = Field(default_factory=dict)
    embedding: list[float] | None = Field(default=None)


class TestRunRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    __test__ = False  # 防止 pytest 将其误收集为测试类

    target_coverage: float = Field(default=0.8, ge=0.0, le=1.0)
    max_steps: int = Field(default=200, ge=1, le=10000)
    invoke_dialogue_model: bool = Field(default=False)


class TestRunResult(BaseModel):
    model_config = ConfigDict(extra="forbid")

    __test__ = False  # 防止 pytest 将其误收集为测试类

    node_coverage: float = Field(..., ge=0.0, le=1.0)
    edge_coverage: float = Field(..., ge=0.0, le=1.0)
    total_nodes: int = Field(..., ge=0)
    covered_nodes: int = Field(..., ge=0)
    total_edges: int = Field(..., ge=0)
    covered_edges: int = Field(..., ge=0)
    unreachable_nodes: list[str] = Field(default_factory=list)
    dead_ends: list[str] = Field(default_factory=list)
    invalid_jumps: list[dict[str, Any]] = Field(default_factory=list)
    steps_taken: int = Field(..., ge=0)
    reached_target: bool
    path_trace: list[str] = Field(default_factory=list)


# ---- 内部辅助模型（非规格模型，用于 LLM 输入与系统内部）----


class MessageRole(str, Enum):
    SYSTEM = "system"
    USER = "user"
    ASSISTANT = "assistant"


class ChatMessage(BaseModel):
    model_config = ConfigDict(str_strip_whitespace=True, extra="forbid")

    role: MessageRole
    content: str = Field(min_length=1, max_length=4000)

    def to_openai_dict(self) -> dict[str, str]:
        return {"role": self.role.value, "content": self.content}


class HealthResponse(BaseModel):
    model_config = ConfigDict(extra="forbid")

    status: str
    app: str
    version: str
    mock_mode: bool
    timestamp: str


class CreateSessionRequest(BaseModel):
    model_config = ConfigDict(extra="forbid", str_strip_whitespace=True)

    player_id: str = Field(..., min_length=1, max_length=64)


class SessionSummary(BaseModel):
    model_config = ConfigDict(extra="forbid")

    session_id: str
    message_count: int = Field(ge=0)
    last_message: str | None = None
    updated_at: str = ""
