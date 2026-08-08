"""业务编排：对话流程的完整处理"""

import json
import logging
from typing import Any

from app import config
from app.schemas import DialogueRequest, NPCProfile, NPCResponse
from app.llm_client import get_llm_client
from app.memory import get_memory_manager
from app.story_engine import get_story_engine
from app.rag import get_rag_store
from app.prompts import build_npc_system_prompt

logger = logging.getLogger(__name__)

DEFAULT_NPCS: list[NPCProfile] = [
    NPCProfile(
        npc_id="gate_guard",
        name="城门守卫巴顿",
        personality="严肃、忠诚、略带多疑。对陌生人保持警惕，但对证明自己价值的人会给予尊重。",
        background="在银月城守卫了十五年城门，见过太多阴谋和伪装。",
        dialogue_style="简短有力，常使用命令式语句，偶尔引用军规",
        voice_tone="grumpy",
        knowledge_domains=["地理", "事件"],
        initial_relationship=0,
    ),
    NPCProfile(
        npc_id="tavern_owner",
        name="酒馆老板玛格丽特",
        personality="热情、八卦、精明。喜欢听冒险故事，也会用信息换取金币。",
        background="经营「醉龙酒馆」二十年，城里没有她不知道的消息。",
        dialogue_style="话多且快，喜欢用比喻，经常插话",
        voice_tone="happy",
        knowledge_domains=["组织", "事件"],
        initial_relationship=10,
    ),
    NPCProfile(
        npc_id="mysterious_stranger",
        name="兜帽旅人",
        personality="神秘、谨慎、话中有话。似乎知道很多不该知道的事情。",
        background="来历不明，总在关键时刻出现。",
        dialogue_style="隐喻多，反问多，说话留半句",
        voice_tone="mysterious",
        knowledge_domains=["神器", "组织"],
        initial_relationship=-5,
    ),
    NPCProfile(
        npc_id="mayor",
        name="城主艾德里克",
        personality="沉稳威严，城府极深，把银月城的利益看得比任何人的性命都重。",
        background="世代掌管银月城，年轻时曾是远征黑森林的骑士，深知封印的秘密。",
        dialogue_style="言辞克制，多用命令与承诺，偶尔透露出疲惫。",
        voice_tone="neutral",
        knowledge_domains=["事件", "组织"],
        initial_relationship=0,
    ),
    NPCProfile(
        npc_id="librarian",
        name="图书馆管理员伊莲",
        personality="安静博学，对历史充满敬畏，说话带着羊皮纸般的耐心。",
        background="在月神图书馆整理典籍三十年，通晓第一纪元至今的全部编年史。",
        dialogue_style="引经据典，语速缓慢，喜欢用卷轴上的原话回答。",
        voice_tone="mysterious",
        knowledge_domains=["历史", "神器"],
        initial_relationship=10,
    ),
    NPCProfile(
        npc_id="ranger",
        name="守林人卡恩",
        personality="沉默寡言，警惕敏锐，认准的人可以托付性命，陌生人休想踏入林界一步。",
        background="第三代守林人，曾在黑森林里独自追踪教团伏击者整整七天。",
        dialogue_style="短句，夹杂哨声与手语，几乎不用形容词。",
        voice_tone="grumpy",
        knowledge_domains=["地理", "事件"],
        initial_relationship=0,
    ),
    NPCProfile(
        npc_id="priestess",
        name="月神殿祭司莉亚娜",
        personality="温柔虔诚，相信月光能净化一切，但内心也藏着对封印将破的恐惧。",
        background="自幼在月神殿长大，主持过三十七次满月祈福，见过圣物图鉴真迹。",
        dialogue_style="语气柔和，爱用月相比喻，祈祷时声音像低吟。",
        voice_tone="neutral",
        knowledge_domains=["传说", "历史"],
        initial_relationship=10,
    ),
    NPCProfile(
        npc_id="merchant",
        name="翡翠商人奥斯汀",
        personality="精明圆滑，笑容永远挂在脸上，账本和真心都锁在铁箱里。",
        background="翡翠商队大掌柜，商队失踪后日夜在酒馆打探消息，怕下一个是自己。",
        dialogue_style="快言快语，满嘴行情，砍价时像连珠炮。",
        voice_tone="happy",
        knowledge_domains=["事件", "矿物"],
        initial_relationship=5,
    ),
    NPCProfile(
        npc_id="alchemist",
        name="药剂师诺拉",
        personality="痴迷月长石的药性，实验笔记写满怪诞猜想，常把活人当试验品般打量。",
        background="曾在王都学院进修，因研究禁忌配方被除名，回到银月城开了间药剂铺。",
        dialogue_style="自言自语多过回答，术语密集，偶尔突然冒出一句真相。",
        voice_tone="mysterious",
        knowledge_domains=["矿物", "传说"],
        initial_relationship=0,
    ),
    NPCProfile(
        npc_id="captain",
        name="城卫队长薇拉",
        personality="雷厉风行，纪律至上，对老兵格外宽容，对偷渡客毫不留情。",
        background="十五年前从边境调任银月城，是唯一亲眼见过教团献祭的幸存军官。",
        dialogue_style="命令式短句，音量高，训话能穿透三条街。",
        voice_tone="grumpy",
        knowledge_domains=["组织", "事件"],
        initial_relationship=0,
    ),
    NPCProfile(
        npc_id="beggar",
        name="老汤姆",
        personality="装疯卖傻，实则耳朵比谁都灵，银月城的小道消息有一半从他这里漏出去。",
        background="曾是星光商队向导，商队失踪后失去一切，靠行乞为生，也借机盯着教团。",
        dialogue_style="颠三倒四，喜欢用谜语和反问，偶尔蹦出一句惊人之语。",
        voice_tone="sad",
        knowledge_domains=["事件", "传说"],
        initial_relationship=-5,
    ),
    NPCProfile(
        npc_id="cult_informant",
        name="塞拉",
        personality="惊弓之鸟，眼神躲闪，说三句藏两句，但每句都是真的。",
        background="影子教团前执事，目睹同伴被虚空之力吞噬后叛逃，如今躲在银月城阴影里。",
        dialogue_style="低声细语，频繁看四周，用暗语指代教团。",
        voice_tone="suspicious",
        knowledge_domains=["组织", "神器"],
        initial_relationship=-10,
    ),
    NPCProfile(
        npc_id="miner",
        name="矿工巴雷特",
        personality="粗犷豪爽，满手老茧，不信神不信鬼，只信铁镐和月长石的光芒。",
        background="在银月矿坑干了二十年，最清楚月长石被谁买走、又运去了哪里。",
        dialogue_style="嗓门大，爱说矿上的黑话，骂骂咧咧却热心肠。",
        voice_tone="grumpy",
        knowledge_domains=["矿物", "地理"],
        initial_relationship=5,
    ),
]


class DialogueService:
    def __init__(self) -> None:
        self.llm = get_llm_client()
        self.memory = get_memory_manager()
        self.story = get_story_engine()
        self.rag = get_rag_store()
        self.npcs: dict[str, NPCProfile] = {}
        self._load_npcs()

    def _load_npcs(self) -> None:
        """
        从 config.NPCS_PATH 加载 NPC 配置。
        JSON 结构：{"npcs": [NPCProfile, ...]}
        用 NPCProfile.model_validate() 逐条校验。
        空文件或格式错误 → raise ValueError
        """
        try:
            raw = config.NPCS_PATH.read_text(encoding="utf-8")
        except Exception as exc:
            raise ValueError(f"NPC 配置读取失败: {config.NPCS_PATH}: {exc}") from exc
        try:
            data = json.loads(raw)
        except Exception as exc:
            raise ValueError(f"NPC 配置文件格式错误: {config.NPCS_PATH}: {exc}") from exc
        if not isinstance(data, dict) or not data.get("npcs"):
            raise ValueError(f"NPC 配置文件缺少 npcs 列表: {config.NPCS_PATH}")
        self.npcs = {
            npc.npc_id: npc
            for npc in (NPCProfile.model_validate(item) for item in data["npcs"])
        }

    def create_session(self, player_id: str) -> str:
        """调用 memory.create_session(player_id, story.get_start_node())"""
        return self.memory.create_session(player_id, self.story.get_start_node())

    async def process_dialogue(self, req: DialogueRequest) -> dict[str, Any]:
        """
        完整对话处理流程（这是核心函数，必须零 bug）：

        Step 1: 校验
        - session = memory.get_session(req.session_id)，None → return {"error": "会话不存在或已过期", "code": 404}
        - npc = self.npcs.get(req.npc_id)，None → return {"error": f"NPC {req.npc_id} 不存在", "code": 404}

        Step 2: 获取剧情上下文
        - current_node_id = session.current_node
        - current_node = story.get_node(current_node_id)
        - available_choices = story.get_available_choices(current_node_id)

        Step 3: RAG 检索
        - domain_filter = npc.knowledge_domains[0] if npc.knowledge_domains else None
        - docs = await rag.search_by_text(req.message.strip(), top_k=3, domain_filter=domain_filter)
        - knowledge_ctx = "\\n".join([f"[{d['doc_id']}] {d['content']}" for d in docs])
        - cited_doc_ids = [d["doc_id"] for d in docs]

        Step 4: 构建 Prompt
        - history_summary = memory.get_history_summary(req.session_id)
        - story_ctx = f"当前节点：{current_node_id}\\n{current_node.description if current_node else ''}"
        - system_prompt = build_npc_system_prompt(npc, story_ctx, knowledge_ctx, history_summary, available_choices)

        Step 5: 调用 LLM
        - npc_response = await llm.chat_completion(system_prompt, req.message.strip())

        Step 6: 【关键】剧情状态机校验
        - 服务端提示（req.choice_id）优先，其次模型建议（npc_response.selected_choice_id）
        - validate_choice 通过 → 应用 effects；不通过 → 记录 warning，模型建议时置信度减半

        Step 7: 更新记忆
        - for mem in npc_response.memory_updates: memory.add_memory(req.session_id, mem, True)

        Step 8: 推进剧情
        - if next_node_id: memory.set_current_node(req.session_id, next_node_id)

        Step 9: 更新关系值（按情感映射）

        Step 10: 记录历史（user/npc 两条）

        Step 11: 组装响应（suggested_choices 为空时用 available_choices 兜底）
        """
        # Step 1: 校验
        session = self.memory.get_session(req.session_id)
        if session is None:
            return {"error": "会话不存在或已过期", "code": 404}
        npc = self.npcs.get(req.npc_id)
        if npc is None:
            return {"error": f"NPC {req.npc_id} 不存在", "code": 404}
        # 服务端初始化 NPC 初始好感度（NPCProfile.initial_relationship）
        session.relationship_values.setdefault(npc.npc_id, npc.initial_relationship)

        # Step 2: 获取剧情上下文
        current_node_id = session.current_node
        current_node = self.story.get_node(current_node_id)
        available_choices = self.story.get_available_choices(current_node_id)

        # Step 2.5: 服务端提示（玩家点击的选项）优先校验并应用，
        # 让 LLM/Mock 知道玩家刚刚做了什么，从而给出与剧情匹配的回复
        validated_choice: str | None = None
        next_node_id: str | None = None
        applied_choice_id: str | None = None
        applied_choice_text: str | None = None
        applied_node_id: str | None = None
        if req.choice_id:
            applied_node_id = current_node_id
            is_valid, next_node, err = self.story.validate_choice(
                current_node_id,
                req.choice_id,
                {
                    "memory": session.memory,
                    "relationship_values": session.relationship_values,
                },
            )
            if is_valid:
                validated_choice = req.choice_id
                next_node_id = next_node
                choice_obj = next(
                    (
                        choice
                        for choice in available_choices
                        if choice.choice_id == req.choice_id
                    ),
                    None,
                )
                if choice_obj is not None:
                    applied_choice_id = choice_obj.choice_id
                    applied_choice_text = choice_obj.text
                    if choice_obj.effects:
                        self.story.apply_effects(
                            {
                                "memory": session.memory,
                                "relationship_values": session.relationship_values,
                            },
                            choice_obj.effects,
                        )
                if next_node_id:
                    current_node_id = next_node_id
                    current_node = self.story.get_node(current_node_id)
                    available_choices = self.story.get_available_choices(current_node_id)
            else:
                logger.warning("玩家选择的选项无效: %s, 错误: %s", req.choice_id, err)

        # Step 3: RAG 检索
        domain_filter = npc.knowledge_domains[0] if npc.knowledge_domains else None
        docs = await self.rag.search_by_text(
            req.message.strip(),
            top_k=3,
            domain_filter=domain_filter,
        )
        knowledge_ctx = "\n".join([f"[{d['doc_id']}] {d['content']}" for d in docs])
        cited_doc_ids = [d["doc_id"] for d in docs]

        # Step 4: 构建 Prompt
        history_summary = self.memory.get_history_summary(req.session_id)
        story_ctx = (
            f"当前节点：{current_node_id}\n"
            f"{current_node.description if current_node else ''}"
        )
        if applied_choice_id:
            story_ctx += (
                f"\n【刚刚发生的动作】玩家在节点 {applied_node_id} 选择了选项"
                f"「{applied_choice_id}」：{applied_choice_text}"
            )
        system_prompt = build_npc_system_prompt(
            npc,
            story_ctx,
            knowledge_ctx,
            history_summary,
            available_choices,
        )

        # Step 5: 调用 LLM
        npc_response = await self.llm.chat_completion(
            system_prompt,
            req.message.strip(),
        )

        # Step 6: 【关键】剧情状态机校验（模型建议；玩家提示已在 Step 2.5 处理）
        if npc_response.selected_choice_id and validated_choice is None:
            candidate = npc_response.selected_choice_id
            is_valid, next_node, err = self.story.validate_choice(
                current_node_id,
                candidate,
                {
                    "memory": session.memory,
                    "relationship_values": session.relationship_values,
                },
            )
            if is_valid:
                validated_choice = candidate
                next_node_id = next_node
                choice_obj = next(
                    (choice for choice in available_choices if choice.choice_id == candidate),
                    None,
                )
                if choice_obj and choice_obj.effects:
                    self.story.apply_effects(
                        {
                            "memory": session.memory,
                            "relationship_values": session.relationship_values,
                        },
                        choice_obj.effects,
                    )
            else:
                logger.warning("LLM 建议了非法选项: %s, 错误: %s", candidate, err)
                npc_response.confidence = max(0.0, npc_response.confidence * 0.5)

        # Step 7: 更新记忆
        for mem in npc_response.memory_updates:
            self.memory.add_memory(req.session_id, mem, True)

        # Step 8: 推进剧情
        if next_node_id:
            self.memory.set_current_node(req.session_id, next_node_id)

        # Step 9: 更新关系值
        emotion_map = {
            "happy": 5,
            "neutral": 0,
            "suspicious": -2,
            "angry": -10,
            "sad": -3,
            "surprised": 1,
            "fearful": -1,
        }
        delta = emotion_map.get(npc_response.emotion, 0)
        if delta != 0:
            self.memory.update_relationship(req.session_id, req.npc_id, delta)

        # Step 10: 记录历史
        self.memory.append_history(
            req.session_id,
            "user",
            req.message,
            {"npc_id": req.npc_id, "choice_id": req.choice_id},
        )
        self.memory.append_history(
            req.session_id,
            "npc",
            npc_response.dialogue,
            {
                "npc_id": req.npc_id,
                "emotion": npc_response.emotion,
                "action": npc_response.action,
                "validated_choice": validated_choice,
            },
        )

        # Step 11: 组装响应
        if not npc_response.suggested_choices and available_choices:
            npc_response.suggested_choices = [
                {"choice_id": choice.choice_id, "text": choice.text}
                for choice in available_choices
            ]

        return {
            "session_id": req.session_id,
            "npc_id": req.npc_id,
            "response": npc_response.model_dump(),
            "current_node": next_node_id or current_node_id,
            "relationship": self.memory.get_relationship(req.session_id, req.npc_id),
            "validated_choice": validated_choice,
            "cited_knowledge": cited_doc_ids,
        }

    def get_session_state(self, session_id: str) -> dict[str, Any] | None:
        """
        获取会话状态快照。
        返回可 JSON 序列化的 dict，包含 session_id, player_id, current_node,
        relationship_values, memory, history_count, covered_nodes, covered_edges。
        covered_nodes 和 covered_edges 从 session 对象读取（它们被 exclude=True，需要手动加入）。
        """
        session = self.memory.get_session(session_id)
        if session is None:
            return None
        return {
            "session_id": session.session_id,
            "player_id": session.player_id,
            "current_node": session.current_node,
            "relationship_values": dict(session.relationship_values),
            "memory": dict(session.memory),
            "history_count": len(session.history),
            "covered_nodes": sorted(session.covered_nodes),
            "covered_edges": sorted(session.covered_edges),
        }

    def delete_session(self, session_id: str) -> bool:
        """删除会话（供 API 删除接口使用）。"""
        return self.memory.delete_session(session_id)


# 全局单例
_dialogue_service: DialogueService | None = None


def get_dialogue_service() -> DialogueService:
    global _dialogue_service
    if _dialogue_service is None:
        _dialogue_service = DialogueService()
    return _dialogue_service
