"""Prompt 模板引擎：构建符合角色人设的 System Prompt"""

from __future__ import annotations

import re
from typing import Any

from app.schemas import NPCProfile, StoryChoice, StoryNode

_TOKEN_RE = re.compile(r"\{\{\s*(\w+)\s*\}\}")


class PromptTemplate:
    """基于 {{key}} 占位符的轻量模板（用于辅助 Prompt）。"""

    def __init__(self, template: str, name: str = "") -> None:
        self.template = template
        self.name = name

    def render(self, **kwargs: Any) -> str:
        missing = set(_TOKEN_RE.findall(self.template)) - set(kwargs)
        if missing:
            raise ValueError(f"Prompt 缺少参数: {', '.join(sorted(missing))}")
        return _TOKEN_RE.sub(lambda match: str(kwargs[match.group(1)]), self.template)


CHOICE_SELECTION_PROMPT = PromptTemplate(
    name="choice_selection",
    template=(
        "你是剧情遍历 Agent。当前节点描述：{{description}}\n"
        "可用选项：\n{{choices_text}}\n"
        "请只回复一个选项编号（choice_id），不要解释。"
    ),
)


def render_rag_results(results: list[dict[str, Any]]) -> str:
    """将 RAG 检索结果（dict 列表）渲染为上下文文本。"""
    if not results:
        return ""
    lines = []
    for item in results:
        metadata = item.get("metadata") or {}
        title = metadata.get("title", item.get("doc_id", ""))
        lines.append(f"- 《{title}》{item.get('content', '')}")
    return "\n".join(lines)


def render_choices_text(choices: list[StoryChoice]) -> str:
    """将可用剧情选项渲染为文本列表。"""
    if not choices:
        return "（当前无可用选项）"
    return "\n".join(f"- [{choice.choice_id}] {choice.text}" for choice in choices)


def build_npc_system_prompt(
    npc: NPCProfile,
    story_ctx: str,
    knowledge_ctx: str,
    history_summary: str,
    available_choices: list[StoryChoice],
) -> str:
    """
    构建 System Prompt。

    要求：
    1. 角色设定段：name, personality, background, dialogue_style, voice_tone
    2. 世界观知识段：knowledge_ctx（可为空）
    3. 当前剧情段：story_ctx
    4. 对话历史段：history_summary（可为空）
    5. 可用选项段：列出 available_choices 的 choice_id 和 text
    6. 输出规则段（JSON 格式强制约束）
    """
    choices_text = "\n".join([f"  [{c.choice_id}] {c.text}" for c in available_choices])
    return f"""你是一位名为「{npc.name}」的 NPC，正在一个奇幻 RPG 游戏中与玩家对话。

【角色设定】
- 性格：{npc.personality}
- 背景：{npc.background}
- 说话风格：{npc.dialogue_style}
- 语气基调：{npc.voice_tone}

【世界观知识】
{knowledge_ctx or "（无）"}

【当前剧情】
{story_ctx}

【对话历史摘要】
{history_summary or "（无）"}

【可用选项】
{choices_text or "（当前无可用选项）"}

【输出规则 - 严格遵守】
1. 你必须以 JSON 格式输出，包含以下字段：
   - "dialogue": 你的回复文本（符合角色性格，100字以内）
   - "emotion": 情绪标签（只能是 neutral/happy/sad/angry/surprised/suspicious/fearful）
   - "action": 可选的动作描述（可为 null）
   - "selected_choice_id": 如果玩家的话对应某个剧情选项，填写该选项 ID；否则为 null
   - "memory_updates": 需要记录的新记忆（字符串列表，可为空）
   - "cited_knowledge_ids": 引用的知识文档 ID 列表（可为空）
   - "confidence": 你对当前回复的置信度（0.0~1.0）
   - "suggested_choices": 当前节点可用选项列表，格式 [{{"choice_id": "...", "text": "..."}}]

2. selected_choice_id 只能从【可用选项】中选择，绝对不能编造不存在的选项。
3. 不要跳出角色，不要提及你是 AI 或语言模型。
4. 如果玩家的话与【世界观知识】矛盾，请委婉指出或表现出困惑。
"""


def build_history_summary(history: list[dict[str, Any]], max_turns: int = 5) -> str:
    """
    取最近 max_turns 轮对话，格式：
    玩家: xxx
    NPC: xxx

    边界：history 为空返回 "（无对话历史）"
    """
    if not history:
        return "（无对话历史）"
    recent = history[-max_turns * 2 :]
    lines: list[str] = []
    for item in recent:
        role = item.get("role")
        content = str(item.get("content", "")).strip()
        if not content:
            continue
        if role == "user":
            lines.append(f"玩家: {content}")
        elif role in ("assistant", "npc"):
            lines.append(f"NPC: {content}")
    return "\n".join(lines) if lines else "（无对话历史）"


def compose_choice_prompt(node: StoryNode, choices: list[StoryChoice]) -> str:
    """为 Agent 遍历生成选项选择提示词。"""
    return CHOICE_SELECTION_PROMPT.render(
        description=node.description,
        choices_text=render_choices_text(choices),
    )
