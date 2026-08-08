"""确定性剧情状态机：模型只给建议，服务端拥有最终决策权"""

import json
import logging
from pathlib import Path
from typing import Any

from app import config
from app.schemas import StoryGraph, StoryNode, StoryChoice

logger = logging.getLogger(__name__)

DEFAULT_STORY_GRAPH = StoryGraph(
    start_node="city_gate",
    nodes={
        "city_gate": StoryNode(
            node_id="city_gate",
            description="银月城城门巍峨，月长石城墙泛着微光。城门守卫巴顿正打量着每一个进城的人。",
            choices=[
                StoryChoice(
                    choice_id="ask_guard",
                    text="上前向守卫询问情况",
                    next_node="guard_dialogue",
                    effects={"relationship.gate_guard": 2},
                ),
                StoryChoice(
                    choice_id="bribe_guard",
                    text="塞给守卫一袋金币",
                    next_node="guard_bribed",
                    preconditions={"memory.has_gold": True},
                    effects={"memory.bribed": True, "relationship.gate_guard": -5},
                ),
                StoryChoice(
                    choice_id="sneak_past",
                    text="趁守卫不注意偷偷溜进城",
                    next_node="caught_sneaking",
                    effects={"memory.sneaking": True},
                ),
                StoryChoice(
                    choice_id="show_badge",
                    text="出示城主的徽章",
                    next_node="city_plaza",
                    preconditions={"relationship.gate_guard": 10},
                    effects={"memory.badge_shown": True},
                ),
            ],
        ),
        "guard_dialogue": StoryNode(
            node_id="guard_dialogue",
            description="巴顿语气生硬，但回答得很有条理。他提到最近商队失踪的消息。",
            choices=[
                StoryChoice(
                    choice_id="ask_caravan",
                    text="打听失踪的商队",
                    next_node="caravan_info",
                    effects={"memory.caravan_info": True, "relationship.gate_guard": 3},
                ),
                StoryChoice(
                    choice_id="ask_danger",
                    text="询问黑森林的危险",
                    next_node="forest_warning",
                    effects={"memory.forest_danger": True, "relationship.gate_guard": 3},
                ),
                StoryChoice(
                    choice_id="salute",
                    text="行一个标准的军礼后进城",
                    next_node="city_plaza",
                    effects={"relationship.gate_guard": 10},
                ),
            ],
        ),
        "guard_bribed": StoryNode(
            node_id="guard_bribed",
            description="你塞出一袋金币，巴顿沉默片刻，侧身让开了一条路。",
            choices=[
                StoryChoice(
                    choice_id="enter",
                    text="快步进城",
                    next_node="city_plaza",
                    effects={"memory.bribed": True},
                ),
                StoryChoice(
                    choice_id="ask_secret",
                    text="追问守卫知道的秘密",
                    next_node="guard_secret",
                    effects={"memory.guard_secret": True},
                ),
            ],
        ),
        "caught_sneaking": StoryNode(
            node_id="caught_sneaking",
            description="你刚翻过墙头，就被巡逻的守卫一把按住。",
            choices=[
                StoryChoice(
                    choice_id="apologize",
                    text="道歉并说明来意",
                    next_node="city_plaza",
                    effects={"memory.warned": True, "relationship.gate_guard": -3},
                ),
                StoryChoice(
                    choice_id="fight",
                    text="挣脱后动手反抗",
                    next_node="guard_fight",
                    effects={"memory.fought_guard": True, "relationship.gate_guard": -10},
                ),
            ],
        ),
        "caravan_info": StoryNode(
            node_id="caravan_info",
            description="巴顿告诉你，失踪的三支商队最后都消失在黑森林方向。",
            choices=[
                StoryChoice(
                    choice_id="go_tavern",
                    text="先去醉龙酒馆打听消息",
                    next_node="tavern",
                ),
                StoryChoice(
                    choice_id="go_forest",
                    text="直接前往黑森林",
                    next_node="forest",
                    effects={"memory.headed_forest": True},
                ),
            ],
        ),
        "forest_warning": StoryNode(
            node_id="forest_warning",
            description="巴顿警告你，黑森林常年被黑色雾气笼罩，连猎户都不愿深入。",
            choices=[
                StoryChoice(
                    choice_id="go_tavern",
                    text="先去醉龙酒馆打听消息",
                    next_node="tavern",
                ),
                StoryChoice(
                    choice_id="go_forest",
                    text="仍然前往黑森林",
                    next_node="forest",
                    effects={"memory.headed_forest": True},
                ),
            ],
        ),
        "guard_secret": StoryNode(
            node_id="guard_secret",
            description="巴顿压低声音：影子教团的人最近常在黑森林边缘出没。",
            choices=[
                StoryChoice(
                    choice_id="go_townhall",
                    text="去市政厅汇报",
                    next_node="townhall",
                    effects={"memory.cult_hint": True},
                ),
                StoryChoice(
                    choice_id="go_tavern",
                    text="去酒馆打听影子教团",
                    next_node="tavern",
                    effects={"memory.cult_hint": True},
                ),
            ],
        ),
        "guard_fight": StoryNode(
            node_id="guard_fight",
            description="你与守卫扭打在一起，惊动了半个城门。",
            choices=[
                StoryChoice(
                    choice_id="surrender",
                    text="束手就擒",
                    next_node="city_plaza",
                    effects={"memory.warned": True, "relationship.gate_guard": -5},
                ),
                StoryChoice(
                    choice_id="win",
                    text="放倒守卫后冲进城内",
                    next_node="city_plaza",
                    effects={"memory.beat_guard": True, "relationship.gate_guard": -8},
                ),
            ],
        ),
        "city_plaza": StoryNode(
            node_id="city_plaza",
            description="银月城中央广场人来人往，醉龙酒馆的招牌在风中摇晃。",
            choices=[
                StoryChoice(
                    choice_id="go_tavern",
                    text="走进醉龙酒馆",
                    next_node="tavern",
                ),
                StoryChoice(
                    choice_id="go_townhall",
                    text="前往市政厅",
                    next_node="townhall",
                ),
                StoryChoice(
                    choice_id="go_forest",
                    text="出城前往黑森林",
                    next_node="forest",
                    effects={"memory.headed_forest": True},
                ),
                StoryChoice(
                    choice_id="visit_library",
                    text="前往月神图书馆查阅古籍",
                    next_node="library",
                    effects={"memory.curious_lore": True},
                ),
                StoryChoice(
                    choice_id="visit_temple",
                    text="前往月神殿祈福",
                    next_node="temple",
                    effects={"memory.curious_faith": True},
                ),
                StoryChoice(
                    choice_id="visit_mine",
                    text="前往城郊的银月矿坑",
                    next_node="mine",
                    effects={"memory.curious_ore": True},
                ),
            ],
        ),
        "temple": StoryNode(
            node_id="temple",
            description="月神殿穹顶洒下银白月光，祭司莉亚娜正在为信徒们低声祈福。",
            choices=[
                StoryChoice(
                    choice_id="pray_blessing",
                    text="参加满月祈福仪式",
                    next_node="temple_ritual",
                    effects={"memory.blessed": True, "relationship.priestess": 3},
                ),
                StoryChoice(
                    choice_id="ask_priestess",
                    text="向莉亚娜询问圣物的传说",
                    next_node="temple_lore",
                    effects={"memory.temple_lore": True},
                ),
            ],
        ),
        "temple_ritual": StoryNode(
            node_id="temple_ritual",
            description="月光穿过彩窗，在你掌心凝成一点银辉。莉亚娜为你轻声祝福。",
            choices=[
                StoryChoice(
                    choice_id="join_ritual",
                    text="带着祝福踏上黑森林之路",
                    next_node="forest",
                    effects={"memory.headed_forest": True, "memory.blessed": True},
                ),
                StoryChoice(
                    choice_id="bless_blade",
                    text="请莉亚娜为你的武器赐福",
                    next_node="forest",
                    effects={"memory.blessed_blade": True},
                ),
            ],
        ),
        "temple_lore": StoryNode(
            node_id="temple_lore",
            description="莉亚娜低声讲述第一纪元：圣物是月神赐给七贤者的礼物，也是最后的希望。",
            choices=[
                StoryChoice(
                    choice_id="learn_oath",
                    text="追问圣物的下落",
                    next_node="cult_trail",
                    effects={"memory.seals_known": True},
                ),
                StoryChoice(
                    choice_id="leave_temple",
                    text="谢过祭司，去酒馆打听消息",
                    next_node="tavern",
                ),
            ],
        ),
        "mine": StoryNode(
            node_id="mine",
            description="银月矿坑入口灯火昏暗，矿工们正把闪着银光的月长石原矿装上板车。",
            choices=[
                StoryChoice(
                    choice_id="inspect_ore",
                    text="查看矿场的发货账本",
                    next_node="mine_ledger",
                    effects={"memory.ore_inspected": True},
                ),
                StoryChoice(
                    choice_id="talk_miners",
                    text="和矿工们套近乎",
                    next_node="mine_rumors",
                    effects={"memory.miner_rumors": True},
                ),
            ],
        ),
        "mine_ledger": StoryNode(
            node_id="mine_ledger",
            description="发货单上，大批月长石被运往黑森林方向，收货人署名是一个月牙印记。",
            choices=[
                StoryChoice(
                    choice_id="follow_shipment",
                    text="带着账本线索去市政厅",
                    next_node="townhall",
                    effects={"memory.shipment_followed": True},
                ),
                StoryChoice(
                    choice_id="trace_cult",
                    text="顺着月牙印记追查教团",
                    next_node="cult_trail",
                    effects={"memory.cult_lead": True},
                ),
            ],
        ),
        "mine_rumors": StoryNode(
            node_id="mine_rumors",
            description="矿工们压低声音：穿黑袍的人每月都来收购月长石，出手阔绰，从不还价。",
            choices=[
                StoryChoice(
                    choice_id="buy_ore",
                    text="花金币买下一块上等原矿",
                    next_node="mine_ledger",
                    preconditions={"memory.has_gold": True},
                    effects={"memory.ore_bought": True},
                ),
                StoryChoice(
                    choice_id="ask_direction",
                    text="询问黑森林的方向后出发",
                    next_node="forest",
                    effects={"memory.headed_forest": True},
                ),
            ],
        ),
        "library": StoryNode(
            node_id="library",
            description="月神图书馆穹顶高悬，管理员伊莲正在整理一摞羊皮卷轴。",
            choices=[
                StoryChoice(
                    choice_id="read_history",
                    text="借阅第一纪元的历史卷轴",
                    next_node="library_lore",
                    effects={"memory.first_era": True},
                ),
                StoryChoice(
                    choice_id="find_map",
                    text="寻找上古废墟的地图",
                    next_node="library_map",
                    effects={"memory.artifact_map": True},
                ),
            ],
        ),
        "library_lore": StoryNode(
            node_id="library_lore",
            description="卷轴上记载着第一纪元的故事：七位贤者铸造圣物，封印虚空之主。",
            choices=[
                StoryChoice(
                    choice_id="ask_seals",
                    text="追问七件圣物的下落",
                    next_node="cult_trail",
                    effects={"memory.seals_known": True},
                ),
                StoryChoice(
                    choice_id="leave",
                    text="谢过伊莲，去酒馆喝一杯",
                    next_node="tavern",
                ),
            ],
        ),
        "library_map": StoryNode(
            node_id="library_map",
            description="你在舆图架上找到一张泛黄的地图，标记着黑森林深处的上古废墟。",
            choices=[
                StoryChoice(
                    choice_id="follow_map",
                    text="按地图前往废墟",
                    next_node="ruins",
                    effects={"memory.headed_forest": True},
                ),
                StoryChoice(
                    choice_id="report_mayor",
                    text="把地图带去市政厅",
                    next_node="townhall",
                    effects={"memory.map_reported": True},
                ),
            ],
        ),
        "tavern": StoryNode(
            node_id="tavern",
            description="醉龙酒馆里人声鼎沸，老板玛格丽特擦着杯子冲你笑。",
            choices=[
                StoryChoice(
                    choice_id="buy_info",
                    text="花金币买一条消息",
                    next_node="tavern_info",
                    preconditions={"memory.has_gold": True},
                    effects={"memory.tavern_info": True, "relationship.tavern_owner": 3},
                ),
                StoryChoice(
                    choice_id="ask_rumors",
                    text="旁敲侧击打听谣言",
                    next_node="tavern_rumors",
                    preconditions={"relationship.tavern_owner": 10},
                    effects={"memory.rumors": True, "relationship.tavern_owner": 5},
                ),
                StoryChoice(
                    choice_id="drink",
                    text="先来一大杯麦酒",
                    next_node="tavern_drunk",
                    effects={"memory.drunk": True},
                ),
            ],
        ),
        "tavern_info": StoryNode(
            node_id="tavern_info",
            description="玛格丽特凑近你，压低声音说出她知道的线索。",
            choices=[
                StoryChoice(
                    choice_id="go_townhall",
                    text="去市政厅确认",
                    next_node="townhall",
                ),
                StoryChoice(
                    choice_id="go_forest",
                    text="按线索前往黑森林",
                    next_node="forest",
                    effects={"memory.headed_forest": True},
                ),
            ],
        ),
        "tavern_rumors": StoryNode(
            node_id="tavern_rumors",
            description="酒客们的闲谈里，反复出现「影子教团」和「黑森林」这两个词。",
            choices=[
                StoryChoice(
                    choice_id="go_townhall",
                    text="去市政厅汇报",
                    next_node="townhall",
                ),
                StoryChoice(
                    choice_id="go_forest",
                    text="前往黑森林查探",
                    next_node="forest",
                    effects={"memory.headed_forest": True},
                ),
            ],
        ),
        "tavern_drunk": StoryNode(
            node_id="tavern_drunk",
            description="麦酒后劲十足，你的视线开始模糊。",
            choices=[
                StoryChoice(
                    choice_id="sleep_off",
                    text="趴在桌上睡一觉",
                    next_node="tavern_rumors",
                    effects={"memory.sober": True},
                ),
                StoryChoice(
                    choice_id="order_coffee",
                    text="点一杯醒酒咖啡",
                    next_node="tavern_info",
                    effects={"memory.sober": True},
                ),
            ],
        ),
        "townhall": StoryNode(
            node_id="townhall",
            description="市政厅大门敞开着，书记官们抱着一摞卷宗来回奔走。",
            choices=[
                StoryChoice(
                    choice_id="meet_mayor",
                    text="求见城主",
                    next_node="mayor_office",
                ),
                StoryChoice(
                    choice_id="check_records",
                    text="溜进档案室查记录",
                    next_node="town_records",
                ),
            ],
        ),
        "mayor_office": StoryNode(
            node_id="mayor_office",
            description="城主听完你的来意，眉头紧锁。他需要有人去黑森林查清商队失踪的真相。",
            choices=[
                StoryChoice(
                    choice_id="accept_mission",
                    text="接下调查任务",
                    next_node="forest_camp",
                    effects={"memory.mission": True},
                ),
                StoryChoice(
                    choice_id="decline",
                    text="婉拒任务，先去查档案",
                    next_node="town_records",
                    effects={"memory.declined": True},
                ),
            ],
        ),
        "town_records": StoryNode(
            node_id="town_records",
            description="档案室的卷宗堆到天花板，灰尘在阳光里飞舞。",
            choices=[
                StoryChoice(
                    choice_id="find_ledger",
                    text="翻找可疑的交易账本",
                    next_node="ledger_clue",
                    effects={"memory.ledger": True},
                ),
                StoryChoice(
                    choice_id="leave",
                    text="离开市政厅去黑森林",
                    next_node="forest",
                    effects={"memory.headed_forest": True},
                ),
            ],
        ),
        "ledger_clue": StoryNode(
            node_id="ledger_clue",
            description="你在账本里发现一笔可疑的月长石交易，买家署名只有一个月牙标记。",
            choices=[
                StoryChoice(
                    choice_id="go_forest",
                    text="顺着线索前往黑森林",
                    next_node="forest_camp",
                    effects={"memory.headed_forest": True},
                ),
                StoryChoice(
                    choice_id="investigate_cult",
                    text="追查月牙标记的来历",
                    next_node="cult_trail",
                    effects={"memory.cult_lead": True},
                ),
            ],
        ),
        "forest": StoryNode(
            node_id="forest",
            description="黑森林边缘雾气翻涌，树干像沉默的哨兵。",
            choices=[
                StoryChoice(
                    choice_id="enter_deep",
                    text="深入黑森林",
                    next_node="forest_camp",
                    effects={"memory.headed_forest": True},
                ),
                StoryChoice(
                    choice_id="follow_river",
                    text="沿河流寻找营地痕迹",
                    next_node="cult_trail",
                    effects={"memory.river_path": True},
                ),
                StoryChoice(
                    choice_id="visit_ranger",
                    text="拜访守林人的小屋",
                    next_node="ranger_hut",
                    effects={"memory.ranger_hut": True},
                ),
            ],
        ),
        "ranger_hut": StoryNode(
            node_id="ranger_hut",
            description="守林人卡恩正擦拭猎弓，屋前挂着几串风干的兽骨，炉火噼啪作响。",
            choices=[
                StoryChoice(
                    choice_id="hire_guide",
                    text="花金币雇卡恩带路",
                    next_node="forest_camp",
                    preconditions={"memory.has_gold": True},
                    effects={"memory.guide_hired": True, "relationship.ranger": 5},
                ),
                StoryChoice(
                    choice_id="ask_tracks",
                    text="请他辨认地上的黑袍脚印",
                    next_node="cult_trail",
                    effects={"memory.ranger_tracks": True},
                ),
            ],
        ),
        "forest_camp": StoryNode(
            node_id="forest_camp",
            description="森林深处有一片被踩平的营地，地上散落着翡翠商队的货物残骸。",
            choices=[
                StoryChoice(
                    choice_id="investigate",
                    text="搜查营地与脚印",
                    next_node="ruins",
                    effects={"memory.camp_found": True},
                ),
                StoryChoice(
                    choice_id="search_tracks",
                    text="追踪拖曳痕迹",
                    next_node="cult_trail",
                    effects={"memory.tracks": True},
                ),
            ],
        ),
        "cult_trail": StoryNode(
            node_id="cult_trail",
            description="你发现一串通往废墟的黑袍脚印，脚印旁落着一枚月牙徽章。",
            choices=[
                StoryChoice(
                    choice_id="follow_trail",
                    text="沿脚印前往废墟",
                    next_node="ruins",
                    effects={"memory.trail": True},
                ),
                StoryChoice(
                    choice_id="study_map",
                    text="对照地图确认废墟入口",
                    next_node="ruins",
                    effects={"memory.map_studied": True},
                ),
            ],
        ),
        "ruins": StoryNode(
            node_id="ruins",
            description="上古废墟的入口处，浮雕上刻着七件圣物的图案。",
            choices=[
                StoryChoice(
                    choice_id="search_altar",
                    text="潜入祭坛房间",
                    next_node="altar_room",
                    effects={"memory.altar": True},
                ),
                StoryChoice(
                    choice_id="ambush_cult",
                    text="伏击落单的黑袍人",
                    next_node="cult_ambush",
                    effects={"memory.ambushed": True},
                ),
            ],
        ),
        "cult_ambush": StoryNode(
            node_id="cult_ambush",
            description="黑袍人从阴影中扑出，将你团团围住。",
            choices=[
                StoryChoice(
                    choice_id="break_free",
                    text="奋力突围冲进祭坛",
                    next_node="altar_room",
                    effects={"memory.escaped": True},
                ),
                StoryChoice(
                    choice_id="captured",
                    text="被俘后假意投靠",
                    next_node="ending_betrayal",
                    effects={"memory.choice": "betrayal"},
                ),
            ],
        ),
        "altar_room": StoryNode(
            node_id="altar_room",
            description="祭坛中央悬浮着虚空之钥，一群黑袍人正在进行献祭仪式。",
            choices=[
                StoryChoice(
                    choice_id="seal",
                    text="夺走虚空之钥并加固封印",
                    next_node="ending_hero",
                    effects={"memory.choice": "hero"},
                ),
                StoryChoice(
                    choice_id="betray",
                    text="加入教团，背叛银月城",
                    next_node="ending_betrayal",
                    effects={"memory.choice": "betrayal"},
                ),
                StoryChoice(
                    choice_id="rescue",
                    text="先救出被囚的商队成员",
                    next_node="ending_rescuer",
                    effects={"memory.choice": "rescuer"},
                ),
            ],
        ),
        "ending_hero": StoryNode(
            node_id="ending_hero",
            description="你取回虚空之钥并加固封印，银月城从此恢复宁静。你成为传说中的英雄。",
            is_end=True,
        ),
        "ending_betrayal": StoryNode(
            node_id="ending_betrayal",
            description="你选择与教团同流合污，银月城陷入长夜。历史会记住你的背叛。",
            is_end=True,
        ),
        "ending_rescuer": StoryNode(
            node_id="ending_rescuer",
            description="你救出被囚的商队成员并销毁虚空之钥，让月光重新照亮黑森林。",
            is_end=True,
        ),
    },
)


class StoryEngine:
    def __init__(self) -> None:
        self.story_graph: StoryGraph | None = None
        self._load()

    def _load(self) -> None:
        """
        从 config.STORY_GRAPH_PATH 加载 JSON。
        1. 文件不存在 → raise FileNotFoundError
        2. 用 StoryGraph.model_validate() 反序列化
        3. 校验 start_node 必须在 nodes 中
        4. 校验所有 choice.next_node 必须在 nodes 中（为 None 除外）
        5. 校验所有 node_id 和 nodes 的 key 一致
        """
        path = config.STORY_GRAPH_PATH
        try:
            raw = path.read_text(encoding="utf-8")
        except FileNotFoundError as exc:
            raise FileNotFoundError(f"剧情图文件不存在: {path}") from exc
        except Exception as exc:
            raise IOError(f"剧情图读取失败: {path}: {exc}") from exc
        data = json.loads(raw)
        graph = StoryGraph.model_validate(data)
        if graph.start_node not in graph.nodes:
            raise ValueError(f"start_node 不在节点映射中: {graph.start_node}")
        for node_id, node in graph.nodes.items():
            if node.node_id != node_id:
                raise ValueError(f"node_id 与节点 key 不一致: {node_id} != {node.node_id}")
            for choice in node.choices:
                if choice.next_node is not None and choice.next_node not in graph.nodes:
                    raise ValueError(f"目标节点不存在: {choice.next_node}")
        self.story_graph = graph

    def get_node(self, node_id: str) -> StoryNode | None:
        """获取节点，不存在返回 None"""
        if self.story_graph is None:
            return None
        return self.story_graph.nodes.get(node_id)

    def get_available_choices(self, node_id: str) -> list[StoryChoice]:
        """获取当前节点可用选项，节点不存在返回 []"""
        node = self.get_node(node_id)
        if node is None:
            return []
        return list(node.choices)

    def validate_choice(
        self,
        node_id: str,
        choice_id: str,
        session_state: dict[str, Any],
    ) -> tuple[bool, str | None, str]:
        """
        校验选项合法性。返回 (is_valid, next_node_id, error_msg)。

        校验流程：
        1. node = get_node(node_id)，不存在 → (False, None, "节点不存在: {node_id}")
        2. 在 node.choices 中查找 choice_id，不存在 → (False, None, "选项 {choice_id} 不属于节点 {node_id}")
        3. 检查 preconditions：
           - key 以 "memory." 开头：session_state["memory"].get(key[7:]) == expected_value
           - key 以 "relationship." 开头：session_state["relationship_values"].get(npc_id, 0) >= expected_value
           - 其他 key：直接 session_state.get(key) == expected_value
           - 任一条件不满足 → (False, None, "前置条件不满足: {key} 需要 {expected}")
        4. next_node 不为 None 但不在 story_graph.nodes 中 → (False, None, "目标节点不存在: {next_node}")
        5. 全部通过 → (True, target_choice.next_node, "")
        """
        node = self.get_node(node_id)
        if node is None:
            return False, None, f"节点不存在: {node_id}"
        target_choice = next(
            (choice for choice in node.choices if choice.choice_id == choice_id),
            None,
        )
        if target_choice is None:
            return False, None, f"选项 {choice_id} 不属于节点 {node_id}"
        for key, expected in target_choice.preconditions.items():
            if key.startswith("memory."):
                memory = session_state.get("memory") or {}
                if memory.get(key[len("memory."):]) != expected:
                    return False, None, f"前置条件不满足: {key} 需要 {expected}"
            elif key.startswith("relationship."):
                npc_id = key[len("relationship."):]
                relationships = session_state.get("relationship_values") or {}
                actual = relationships.get(npc_id, 0)
                if not (isinstance(actual, (int, float)) and actual >= expected):
                    return False, None, f"前置条件不满足: {key} 需要 {expected}"
            else:
                if session_state.get(key) != expected:
                    return False, None, f"前置条件不满足: {key} 需要 {expected}"
        if (
            target_choice.next_node is not None
            and self.story_graph is not None
            and target_choice.next_node not in self.story_graph.nodes
        ):
            return False, None, f"目标节点不存在: {target_choice.next_node}"
        return True, target_choice.next_node, ""

    def apply_effects(self, session_state: dict[str, Any], effects: dict[str, Any]) -> None:
        """
        应用选择后的效果。
        - key 以 "memory." 开头：session_state["memory"][key[7:]] = value
        - key 以 "relationship." 开头：session_state["relationship_values"][npc_id] += value（限制 [-100,100]）
        """
        memory = session_state.setdefault("memory", {})
        relationships = session_state.setdefault("relationship_values", {})
        for key, value in effects.items():
            if key.startswith("memory."):
                memory[key[len("memory."):]] = value
            elif key.startswith("relationship."):
                npc_id = key[len("relationship."):]
                current = relationships.get(npc_id, 0)
                relationships[npc_id] = max(-100, min(100, current + value))
            else:
                session_state[key] = value

    def get_all_nodes(self) -> list[str]:
        """返回所有节点 ID 列表"""
        if self.story_graph is None:
            return []
        return list(self.story_graph.nodes)

    def get_all_edges(self) -> list[tuple[str, str, str]]:
        """
        返回所有边 (from_node_id, choice_id, to_node_id)。
        只返回 next_node 不为 None 的 choice。
        """
        if self.story_graph is None:
            return []
        edges: list[tuple[str, str, str]] = []
        for node_id, node in self.story_graph.nodes.items():
            for choice in node.choices:
                if choice.next_node is not None:
                    edges.append((node_id, choice.choice_id, choice.next_node))
        return edges

    def get_start_node(self) -> str:
        """返回起始节点 ID"""
        return self.story_graph.start_node

    def find_dead_ends(self) -> list[str]:
        """返回所有终局节点（is_end=True 或 choices 为空）"""
        if self.story_graph is None:
            return []
        return [
            node_id
            for node_id, node in self.story_graph.nodes.items()
            if node.is_end or not node.choices
        ]

    def find_unreachable_from(self, start: str) -> list[str]:
        """
        从 start 节点 BFS 遍历。
        返回所有从 start 出发不可达的节点 ID。
        边界：start 不存在返回所有节点。
        """
        if self.story_graph is None:
            return []
        nodes = self.story_graph.nodes
        if start not in nodes:
            return list(nodes)
        reachable: set[str] = set()
        queue = [start]
        while queue:
            node_id = queue.pop(0)
            if node_id in reachable:
                continue
            reachable.add(node_id)
            for choice in nodes[node_id].choices:
                if (
                    choice.next_node is not None
                    and choice.next_node in nodes
                    and choice.next_node not in reachable
                ):
                    queue.append(choice.next_node)
        return [node_id for node_id in nodes if node_id not in reachable]

    def validate_graph_integrity(self) -> list[str]:
        """
        校验剧情图完整性。
        检查项：
        1. 所有 next_node 引用必须存在
        2. 不能有孤立节点（至少被一条边指向或作为 start_node）
        3. 返回错误信息列表，空列表表示无错误
        """
        if self.story_graph is None:
            return ["剧情图未加载"]
        errors: list[str] = []
        nodes = self.story_graph.nodes
        incoming: set[str] = set()
        for node_id, node in nodes.items():
            for choice in node.choices:
                if choice.next_node is not None:
                    if choice.next_node not in nodes:
                        errors.append(f"边 {node_id} -> {choice.next_node} 的目标节点不存在")
                    else:
                        incoming.add(choice.next_node)
        for node_id in nodes:
            if node_id != self.story_graph.start_node and node_id not in incoming:
                errors.append(f"孤立节点: {node_id}")
        return errors


# 全局单例
_story_engine: StoryEngine | None = None


def get_story_engine() -> StoryEngine:
    global _story_engine
    if _story_engine is None:
        _story_engine = StoryEngine()
    return _story_engine
