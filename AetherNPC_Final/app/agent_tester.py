"""AI Agent：自动遍历剧情图，输出覆盖率报告"""

import random
import logging
from typing import Any

from app.schemas import TestRunRequest, TestRunResult, StoryChoice
from app.story_engine import get_story_engine
from app.memory import get_memory_manager
from app.services.dialogue_service import get_dialogue_service

logger = logging.getLogger(__name__)


class AgentTester:
    def __init__(self) -> None:
        self.story = get_story_engine()
        self.memory = get_memory_manager()
        self.service = get_dialogue_service()

    async def run(self, req: TestRunRequest) -> TestRunResult:
        """
        自动遍历剧情图。

        算法：
        1. all_nodes = set(story.get_all_nodes())
        2. all_edges = set(story.get_all_edges())
        3. covered_nodes = set(), covered_edges = set(), dead_ends = [], invalid_jumps = [], path_trace = []
        4. 创建测试会话：session_id = service.create_session("agent-tester")
        5. current_node = story.get_start_node()
        6. covered_nodes.add(current_node)
        7. path_trace.append(current_node)
        """
        all_nodes = set(self.story.get_all_nodes())
        all_edges = set(self.story.get_all_edges())
        covered_nodes: set[str] = set()
        covered_edges: set[tuple[str, str, str]] = set()
        dead_ends: list[str] = []
        invalid_jumps: list[dict[str, Any]] = []
        path_trace: list[str] = []
        session_id = self.service.create_session("agent-tester")
        session = self.memory.get_session(session_id)
        # 服务端初始化各 NPC 初始好感度，保证关系型前置条件可达
        for npc in self.service.npcs.values():
            session.relationship_values[npc.npc_id] = npc.initial_relationship
        current_node = self.story.get_start_node()
        covered_nodes.add(current_node)
        path_trace.append(current_node)
        steps = 0
        node_cov = 0.0
        edge_cov = 0.0

        while steps < req.max_steps:
            session.covered_nodes = covered_nodes
            session.covered_edges = covered_edges
            node = self.story.get_node(current_node)
            if not node or not node.choices:
                # 死胡同
                if current_node not in dead_ends:
                    dead_ends.append(current_node)
                backtrack = self._find_backtrack(session_id, covered_edges, all_edges)
                if backtrack:
                    current_node = backtrack
                    path_trace.append(f"BACKTRACK->{current_node}")
                    steps += 1
                    continue
                break

            # 启发式评分
            scored_choices: list[tuple[StoryChoice, int]] = []
            for choice in node.choices:
                next_n = choice.next_node
                edge = (current_node, choice.choice_id, next_n)
                score = 0
                if next_n and next_n not in covered_nodes:
                    score = 3  # 新节点最高优先级
                elif edge not in covered_edges:
                    score = 2  # 新边次高
                else:
                    score = 1  # 已覆盖
                scored_choices.append((choice, score))

            # 按 score 降序，同分随机
            scored_choices.sort(key=lambda item: (-item[1], random.random()))

            # 当前节点没有未覆盖的新节点/新边时，直接回溯到有未覆盖边的节点，
            # 避免沿已覆盖边空转耗尽 max_steps。
            if not any(score >= 2 for _, score in scored_choices):
                backtrack = self._find_backtrack(session_id, covered_edges, all_edges)
                if backtrack:
                    current_node = backtrack
                    path_trace.append(f"BACKTRACK->{current_node}")
                    steps += 1
                    continue
                break

            # 尝试选择，如果非法则尝试下一个
            selected_choice: StoryChoice | None = None
            for choice, _ in scored_choices:
                state = self.memory.get_session(session_id)
                is_valid, next_node, err = self.story.validate_choice(
                    current_node,
                    choice.choice_id,
                    {
                        "memory": state.memory,
                        "relationship_values": state.relationship_values,
                    },
                )
                if is_valid:
                    selected_choice = choice
                    break
                invalid_jumps.append(
                    {
                        "from_node": current_node,
                        "choice_id": choice.choice_id,
                        "error": err,
                        "step": steps,
                    }
                )

            if not selected_choice:
                break  # 所有选项都非法

            # 推进（应用 effects，保证后续前置条件可达）
            state = self.memory.get_session(session_id)
            self.story.apply_effects(
                {
                    "memory": state.memory,
                    "relationship_values": state.relationship_values,
                },
                selected_choice.effects,
            )
            edge = (current_node, selected_choice.choice_id, selected_choice.next_node)
            covered_edges.add(edge)
            if selected_choice.next_node:
                covered_nodes.add(selected_choice.next_node)
                self.memory.set_current_node(session_id, selected_choice.next_node)
                current_node = selected_choice.next_node
                path_trace.append(current_node)

            steps += 1

            # 检查覆盖率
            node_cov = len(covered_nodes) / len(all_nodes) if all_nodes else 1.0
            edge_cov = len(covered_edges) / len(all_edges) if all_edges else 1.0
            if node_cov >= req.target_coverage and edge_cov >= req.target_coverage:
                break

        unreachable = list(all_nodes - covered_nodes)
        return TestRunResult(
            node_coverage=round(node_cov, 4),
            edge_coverage=round(edge_cov, 4),
            total_nodes=len(all_nodes),
            covered_nodes=len(covered_nodes),
            total_edges=len(all_edges),
            covered_edges=len(covered_edges),
            unreachable_nodes=unreachable,
            dead_ends=dead_ends,
            invalid_jumps=invalid_jumps,
            steps_taken=steps,
            reached_target=(
                node_cov >= req.target_coverage and edge_cov >= req.target_coverage
            ),
            path_trace=path_trace,
        )

    def _find_backtrack(
        self,
        session_id: str,
        covered_edges: set[tuple[str, str, str]],
        all_edges: set[tuple[str, str, str]],
    ) -> str | None:
        """
        在已访问节点中找一个还有未覆盖边的节点。
        遍历 session.covered_nodes，对每个节点检查其 choices，
        如果存在边 (node_id, choice_id, next_node) 不在 covered_edges 中且在 all_edges 中，
        且该选项在当前会话状态下校验通过（避免被前置条件锁死的边反复回溯），返回该 node_id。
        """
        session = self.memory.get_session(session_id)
        if session is None:
            return None
        session_state = {
            "memory": session.memory,
            "relationship_values": session.relationship_values,
        }
        for node_id in session.covered_nodes:
            node = self.story.get_node(node_id)
            if node is None:
                continue
            for choice in node.choices:
                edge = (node_id, choice.choice_id, choice.next_node)
                if edge in all_edges and edge not in covered_edges:
                    is_valid, _, _ = self.story.validate_choice(
                        node_id,
                        choice.choice_id,
                        session_state,
                    )
                    if not is_valid:
                        continue
                    return node_id
        return None


# 全局单例
_agent_tester: AgentTester | None = None


def get_agent_tester() -> AgentTester:
    global _agent_tester
    if _agent_tester is None:
        _agent_tester = AgentTester()
    return _agent_tester
