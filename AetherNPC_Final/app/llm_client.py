"""LLM 客户端：OpenAI API + Mock 降级"""

import hashlib
import json
import logging
import random
import re
from typing import Any

import httpx
import numpy as np

from app import config
from app.schemas import NPCResponse

logger = logging.getLogger(__name__)


def _parse_suggested_choices(system_prompt: str) -> list[dict[str, str]]:
    """从 System Prompt 的【可用选项】段解析选项列表。"""
    choices: list[dict[str, str]] = []
    in_section = False
    for line in system_prompt.splitlines():
        stripped = line.strip()
        if stripped.startswith("【可用选项】"):
            in_section = True
            continue
        if in_section:
            if stripped.startswith("【"):
                break
            match = re.match(r"\[([a-z0-9_]+)\]\s*(.+)", stripped)
            if match:
                choices.append({"choice_id": match.group(1), "text": match.group(2).strip()})
    return choices


def _extract_npc_name(system_prompt: str) -> str:
    """从 System Prompt 中解析 NPC 名称。"""
    match = re.search(r"(?:你是|名为)「(.+?)」", system_prompt)
    return match.group(1).strip() if match else "NPC"


def _parse_mock_context(system_prompt: str) -> tuple[str | None, str | None, str | None]:
    """从 System Prompt 中解析当前节点与玩家刚选择的选项。"""
    node_id = None
    choice_id = None
    choice_text = None
    match = re.search(r"当前节点：(\S+)", system_prompt)
    if match:
        node_id = match.group(1).strip()
    match = re.search(r"玩家在节点 (\S+) 选择了选项「([a-z0-9_]+)」：(.+)", system_prompt)
    if match:
        node_id = match.group(1).strip()
        choice_id = match.group(2)
        choice_text = match.group(3).strip()
    return node_id, choice_id, choice_text


# 主回复表：覆盖全部剧情选项的生动文案
# (dialogue, emotion, action, confidence, cited_knowledge_ids)
CHOICE_REPLIES: dict[tuple[str, str], tuple[str, str, str, float, list[str]]] = {
    ("city_gate", "ask_guard"): (
        "巴顿把长矛往地上一顿，上下打量你：「外乡人？报上名来，说明来意。最近商队失踪的事闹得紧，别给老子添乱。」",
        "neutral", "顿矛打量", 0.9, [],
    ),
    ("city_gate", "bribe_guard"): (
        "你悄悄把一袋金币塞进巴顿手里。他掂了掂，沉默片刻，侧身让开一条缝：「……进去吧。今天你没见过我，我也没见过你。」",
        "suspicious", "侧身让路", 0.9, [],
    ),
    ("city_gate", "sneak_past"): (
        "你贴着墙根摸向城门，眼看就要成功——脚下枯枝咔嚓一响。「站住！干什么的？」巡逻的守卫呼啦一声围了上来，按住了你的肩膀。",
        "surprised", "被按在墙上", 0.9, [],
    ),
    ("city_gate", "show_badge"): (
        "你亮出城主徽章，银光在月光下一闪。巴顿瞳孔一缩，立刻立正敬礼：「大人恕罪！请进——城主正在市政厅等您。」",
        "neutral", "立正敬礼", 0.95, [],
    ),
    ("guard_dialogue", "ask_caravan"): (
        "巴顿压低声音，像在说什么军机：「翡翠、铁盾、星光，三支商队全栽在黑森林。星光商队三天前失踪，最后见他们的是守林人卡恩。」",
        "suspicious", "压低声音", 0.9, ["knowledge_004"],
    ),
    ("guard_dialogue", "ask_danger"): (
        "巴顿皱眉，拇指划过矛刃：「黑森林那雾气，进去十个人，能出来七个都算烧高香。连猎户都不愿靠近，你最好也别。」",
        "suspicious", "皱眉摩挲矛刃", 0.9, ["knowledge_005"],
    ),
    ("guard_dialogue", "salute"): (
        "你脚跟一碰，行了个标准的军礼。巴顿眼神松动，回礼放行：「是条汉子。进城吧——记住，别让城里的人难做。」",
        "neutral", "回礼放行", 0.9, [],
    ),
    ("guard_bribed", "enter"): (
        "你快步闪进城门，消失在人群中。身后的巴顿像什么都没发生过，继续面无表情地站岗。",
        "neutral", "快步进城", 0.9, [],
    ),
    ("guard_bribed", "ask_secret"): (
        "巴顿四下张望，声音压到几乎听不见：「影子教团的人，最近总在黑森林边缘出没。这话我只说一次，别让我后悔。」",
        "suspicious", "警惕地张望", 0.9, ["knowledge_002"],
    ),
    ("caught_sneaking", "apologize"): (
        "你连忙道歉，亮出守林人卡恩的信物，守卫将信将疑地松开你：「再有下次，直接送地牢。走吧。」",
        "neutral", "松开你的衣领", 0.85, [],
    ),
    ("caught_sneaking", "fight"): (
        "你猛一挣身，和守卫扭打在一起。城门口顿时乱作一团，示警的号角声刺破夜空，更多守卫冲了过来。",
        "angry", "扭打挣脱", 0.85, [],
    ),
    ("caravan_info", "go_tavern"): (
        "你谢过巴顿，转身走向醉龙酒馆。据说那里的老板玛格丽特，连老鼠洞里的秘密都能挖出来。",
        "neutral", "转身走向酒馆", 0.9, [],
    ),
    ("caravan_info", "go_forest"): (
        "你决定直奔黑森林。城门外，雾气已经在林线边缘翻涌，像一只等候多时的巨兽。",
        "neutral", "望向林线", 0.9, [],
    ),
    ("forest_warning", "go_tavern"): (
        "你决定先听听酒馆里的消息。临别时巴顿补了一句：「要是见到穿黑袍的，绕着走。」",
        "neutral", "点点头", 0.9, [],
    ),
    ("forest_warning", "go_forest"): (
        "你谢过警告，还是迈步走向黑森林。身后传来巴顿低低的一声叹息。",
        "neutral", "迈步向前", 0.9, [],
    ),
    ("guard_secret", "go_townhall"): (
        "你揣着这个秘密赶往市政厅。月长石铺就的台阶在月光下泛着冷光，像在提醒你：知道太多，未必是好事。",
        "neutral", "快步离开", 0.9, [],
    ),
    ("guard_secret", "go_tavern"): (
        "你走进醉龙酒馆，打算探探影子教团的风声。炉火映着酒客们的脸，每个人都在假装什么都不知道。",
        "neutral", "推门而入", 0.9, [],
    ),
    ("guard_fight", "surrender"): (
        "你被按在地上，反剪双手押进城。巴顿冷冷道：「地牢里清醒一夜，明天再好好说话。」",
        "angry", "被押进城", 0.85, [],
    ),
    ("guard_fight", "win"): (
        "你放倒两名守卫，夺路冲进城内。身后号角长鸣，整座城门都为你醒了过来。",
        "angry", "夺路冲进城内", 0.85, [],
    ),
    ("city_plaza", "go_tavern"): (
        "你推开醉龙酒馆的木门，喧闹声和酒香扑面而来。玛格丽特擦着杯子冲你笑：「新面孔？坐，姐姐请第一杯。」",
        "happy", "推门而入", 0.9, [],
    ),
    ("city_plaza", "go_townhall"): (
        "你穿过广场走向市政厅，书记官们抱着一摞卷宗匆匆擦肩而过。有人在低声议论商队的事，又立刻噤声。",
        "neutral", "穿过广场", 0.9, [],
    ),
    ("city_plaza", "go_forest"): (
        "你出城向东，暮色里黑森林的轮廓渐渐逼近。风从林间钻出来，带着一股说不清道不明的腥甜。",
        "neutral", "出城向东", 0.9, [],
    ),
    ("city_plaza", "visit_library"): (
        "你推开月神图书馆的铜门，管理员伊莲推了推眼镜：「欢迎。这里的每一卷羊皮纸，都比这座城市更老。」",
        "neutral", "推了推眼镜", 0.9, [],
    ),
    ("city_plaza", "visit_temple"): (
        "你沿着月长石台阶走向月神殿，银白的月光在穹顶流淌。祭司莉亚娜回头看你：「旅人，来听月亮说话吗？」",
        "neutral", "抬头望向穹顶", 0.9, [],
    ),
    ("city_plaza", "visit_mine"): (
        "你来到城郊矿坑，空气里飘着矿石的尘土味。矿工们正把月长石原矿装上板车，有人朝你吹了声口哨。",
        "neutral", "走进矿场", 0.9, [],
    ),
    ("temple", "pray_blessing"): (
        "你在莉亚娜面前跪下，月光透过彩窗落进掌心。她低声道：「愿月神看清你的来路，也照亮你的归途。」",
        "neutral", "低声祈福", 0.95, ["knowledge_010"],
    ),
    ("temple", "ask_priestess"): (
        "莉亚娜望着祭坛上的圣物浮雕，声音轻得像叹息：「第一纪元，七贤者以月神之名铸下七件圣物——可如今，只剩传说了。」",
        "suspicious", "凝视浮雕", 0.9, ["knowledge_007", "knowledge_008"],
    ),
    ("temple_ritual", "join_ritual"): (
        "仪式结束，你掌心的银辉凝成一点印记。莉亚娜郑重道：「带着月光上路吧，黑森林里的黑暗，怕光。」",
        "happy", "掌心泛起银辉", 0.95, [],
    ),
    ("temple_ritual", "bless_blade"): (
        "莉亚娜将月长石粉末撒上你的武器，银光沿着刃口游走。「愿它斩得断黑暗。」她轻声说。",
        "neutral", "为武器赐福", 0.95, ["knowledge_006"],
    ),
    ("temple_lore", "learn_oath"): (
        "莉亚娜压低声音：「虚空之钥就在黑森林的废墟里，教团想用它打开虚空之门。其余圣物散落大陆，我这里有它们的图鉴。」",
        "suspicious", "压低声音", 0.95, ["knowledge_008", "knowledge_003"],
    ),
    ("temple_lore", "leave_temple"): (
        "你谢过莉亚娜，走出月神殿。她望着你的背影低语：「愿月神指引你，别让月光熄灭。」",
        "neutral", "合掌告别", 0.9, [],
    ),
    ("mine", "inspect_ore"): (
        "你溜进货仓，翻开积灰的发货账本。大批月长石被运往黑森林，收货人署名只有一个月牙印记。",
        "suspicious", "翻动账本", 0.9, ["knowledge_006"],
    ),
    ("mine", "talk_miners"): (
        "你递出半壶酒，矿工们立刻打开了话匣子。「穿黑袍的每月都来收矿石，出手阔绰，从不还价——邪门得很。」",
        "neutral", "递出酒壶", 0.9, ["knowledge_011"],
    ),
    ("mine_ledger", "follow_shipment"): (
        "你带上账本线索赶往市政厅。月牙印记像一枚烙印，烫在每个失踪的名字下面。",
        "neutral", "收好账本", 0.9, [],
    ),
    ("mine_ledger", "trace_cult"): (
        "你顺着月牙印记的方向追去，发现它与影子教团的徽记一模一样——线索在黑森林深处等着你。",
        "suspicious", "对照徽记", 0.9, ["knowledge_002"],
    ),
    ("mine_rumors", "buy_ore"): (
        "你掏出一袋金币，买下一块上等月长石原矿。矿工巴雷特压着嗓子：「兄弟，跟着这矿走，你什么都知道了。」",
        "suspicious", "掂量原矿", 0.9, ["knowledge_006"],
    ),
    ("mine_rumors", "ask_direction"): (
        "矿工指了个方向：「黑森林在东边，雾气最浓的地方。记住，进了林子别回头。」",
        "neutral", "指向东方", 0.9, ["knowledge_005"],
    ),
    ("library", "read_history"): (
        "伊莲递来一卷泛黄的羊皮纸：「第一纪元，月神赐福，七位贤者铸造圣物封印虚空之主——那是银月城传说的开端，也是末日的伏笔。」",
        "neutral", "递来卷轴", 0.95, ["knowledge_007"],
    ),
    ("library", "find_map"): (
        "你在舆图架下层翻出一张泛黄的地图，标注着黑森林深处的上古废墟，还有一条隐秘的通道。伊莲皱眉：「这张图……不该在这里。」",
        "suspicious", "展开地图", 0.9, ["knowledge_009"],
    ),
    ("library_lore", "ask_seals"): (
        "你追问圣物下落，伊莲的声音压得极低：「虚空之钥就在废墟里，教团想用它开门。其余六件散落大陆——但传说，集齐七件才能彻底终结虚空。」",
        "suspicious", "压低声音", 0.95, ["knowledge_008", "knowledge_003"],
    ),
    ("library_lore", "leave"): (
        "你谢过伊莲，合上卷轴。她忽然补了一句：「月光会记住你今晚读过的每一个字。」",
        "neutral", "合上卷轴", 0.9, [],
    ),
    ("library_map", "follow_map"): (
        "你收好地图，穿过黑森林的浓雾。古老的废墟轮廓在树影后显现，地图上的标记与现实严丝合缝。",
        "neutral", "收好地图", 0.9, ["knowledge_009"],
    ),
    ("library_map", "report_mayor"): (
        "你带着地图赶到市政厅。城主盯着标注的废墟入口，指尖轻轻叩着桌面：「看来，有人比我们先一步找到了它。」",
        "neutral", "呈上地图", 0.9, [],
    ),
    ("tavern", "buy_info"): (
        "玛格丽特收下金币，声音压过炉火的噼啪声：「月长石交易出了问题——账本上有个月牙标记，和影子教团脱不了干系。这话出了门我就不认。」",
        "suspicious", "压低声音", 0.9, ["knowledge_006", "knowledge_002"],
    ),
    ("tavern", "ask_rumors"): (
        "酒客们醉醺醺地议论着：「黑森林里闹鬼」「有人看见黑袍人」「星光商队根本没走出林子」……玛格丽特朝你使了个眼色，努了努嘴。",
        "neutral", "使了个眼色", 0.9, ["knowledge_004"],
    ),
    ("tavern", "drink"): (
        "一大杯麦酒下肚，暖意顺着喉咙烧到胃里。邻桌的低语钻进耳朵：「废墟、祭坛、七件圣物……钥匙就在那儿。」",
        "happy", "畅饮麦酒", 0.85, [],
    ),
    ("tavern_info", "go_townhall"): (
        "你带着酒馆的情报赶往市政厅，脚步不由自主地快了起来。城里的灯火一盏盏熄了，只有月光还醒着。",
        "neutral", "快步离开", 0.9, [],
    ),
    ("tavern_info", "go_forest"): (
        "你按玛格丽特的线索前往黑森林，腰间的水壶随脚步叮当作响。她追到门口喊：「活着回来！」",
        "neutral", "动身出发", 0.9, [],
    ),
    ("tavern_rumors", "go_townhall"): (
        "你把零碎的谣言拼成一条线，快步走向市政厅。老汤姆蹲在墙角冲你喊：「顺着月牙走，别回头！」",
        "neutral", "整理思绪", 0.9, [],
    ),
    ("tavern_rumors", "go_forest"): (
        "你决定顺着谣言去黑森林一探究竟。玛格丽特往你手里塞了一块干粮：「带上，路远。」",
        "neutral", "走向城门", 0.9, [],
    ),
    ("tavern_drunk", "sleep_off"): (
        "你在桌上睡了一觉，醒来时酒馆只剩烛火。桌上压着一张纸条，是玛格丽特的字迹：「线索在黑森林，小心影子教团。醒了就走吧。」",
        "neutral", "揉着额头醒来", 0.85, ["knowledge_002"],
    ),
    ("tavern_drunk", "order_coffee"): (
        "一杯醒酒咖啡下肚，你清醒了不少。玛格丽特顺势坐下，低声说起她的情报，眼神里没有半点玩笑。",
        "neutral", "啜饮咖啡", 0.9, [],
    ),
    ("townhall", "meet_mayor"): (
        "你被领进城主办公室。城主放下鹅毛笔，锐利的目光像能看穿人心：「商队、教团、月长石——说说你知道的，别漏一个字。」",
        "neutral", "审视着你", 0.9, [],
    ),
    ("townhall", "check_records"): (
        "你溜进档案室，灰尘呛得你直打喷嚏。成堆的卷宗里，总有一页记着不该记的东西。",
        "neutral", "翻找卷宗", 0.9, [],
    ),
    ("mayor_office", "accept_mission"): (
        "城主点头，从抽屉里取出一封信：「商队和月长石交易都要查清。带上这封信，必要时可以调动城门卫——但别让我失望。」",
        "neutral", "递出一封信", 0.95, [],
    ),
    ("mayor_office", "decline"): (
        "你婉拒任务，城主挑眉：「那至少去档案室看看账本。线索不会等人，教团更不会。」",
        "neutral", "挑了挑眉", 0.9, [],
    ),
    ("town_records", "find_ledger"): (
        "你在积灰的账本里翻到一笔可疑的月长石交易，买家的署名只有一个月牙标记。墨迹很新，像是有人刚补上的。",
        "suspicious", "翻动账本", 0.9, ["knowledge_006"],
    ),
    ("town_records", "leave"): (
        "你放下卷宗，离开市政厅。门外的月光很亮，亮得有些反常——像是整座城都在屏息。",
        "neutral", "放下卷宗", 0.9, [],
    ),
    ("ledger_clue", "go_forest"): (
        "你收起账本线索，动身前往黑森林。月牙标记像一枚冷眼，钉在你的记忆里。",
        "neutral", "收起账本", 0.9, [],
    ),
    ("ledger_clue", "investigate_cult"): (
        "你顺着月牙标记追查，发现它与影子教团的徽记一模一样。风从黑森林方向吹来，带着焚香与铁锈的气味。",
        "suspicious", "对照徽记", 0.9, ["knowledge_002"],
    ),
    ("forest", "enter_deep"): (
        "你拨开浓雾踏入森林，脚下是松软的腐叶。远处传来低沉的嗡鸣，像某种古老的东西正在苏醒。",
        "neutral", "拨开浓雾", 0.9, [],
    ),
    ("forest", "follow_river"): (
        "你沿河流前进，水声掩盖了脚步声。岸边的泥地上留着拖曳的痕迹——那不是野兽留下的。",
        "suspicious", "沿河前进", 0.9, [],
    ),
    ("forest", "visit_ranger"): (
        "你走向林边的守林人小屋。卡恩抬眼看你，猎弓的弦在月光下绷成一条银线：「进林子？先说说你的来意。」",
        "neutral", "打量来客", 0.9, [],
    ),
    ("ranger_hut", "hire_guide"): (
        "你掏出一袋金币放在桌上。卡恩掂了掂，别上猎刀：「成交。进林子里，听我的，别自作聪明。」",
        "neutral", "别上猎刀", 0.9, [],
    ),
    ("ranger_hut", "ask_tracks"): (
        "卡恩蹲下身，指尖划过黑袍脚印：「三个成年人，负重不轻，往废墟方向去了。脚印很新——你来得正是时候。」",
        "suspicious", "蹲下查看脚印", 0.9, ["knowledge_011"],
    ),
    ("forest_camp", "investigate"): (
        "你搜遍营地，找到翡翠商队的货物残骸，还有一路拖向废墟的痕迹。火堆还是温的——他们没走远。",
        "suspicious", "搜查营地", 0.9, ["knowledge_004"],
    ),
    ("forest_camp", "search_tracks"): (
        "你循着拖曳痕迹前进，在泥地里发现一串黑袍脚印，尽头通向废墟。一枚月牙徽章静静躺在脚印旁。",
        "suspicious", "蹲下查看", 0.9, [],
    ),
    ("cult_trail", "follow_trail"): (
        "你沿黑袍脚印来到废墟入口，浮雕上的七件圣物图案在月光下泛着冷光，像七双注视的眼睛。",
        "suspicious", "沿脚印前进", 0.9, ["knowledge_003"],
    ),
    ("cult_trail", "study_map"): (
        "你对照地图确认了废墟的入口——这是上古祭坛的遗迹。风从门内涌出，带着焚香与低吟。",
        "neutral", "对照地图", 0.9, ["knowledge_009"],
    ),
    ("ruins", "search_altar"): (
        "你从侧廊潜入，屏住呼吸。祭坛中央悬浮着虚空之钥，黑袍人正围着它低声吟唱，献祭的石台还残留着暗红的痕迹。",
        "fearful", "屏息潜入", 0.9, ["knowledge_003"],
    ),
    ("ruins", "ambush_cult"): (
        "你伏击了一个落单的黑袍人，勒住他的脖子逼问祭坛位置。他临死前低语：「虚空之主……即将降临……」",
        "fearful", "勒住黑袍人", 0.9, ["knowledge_002"],
    ),
    ("cult_ambush", "break_free"): (
        "你奋力突围，撞开祭坛大门。黑袍人在身后嘶吼着追来，月光在你脚下铺成一条逃生之路。",
        "fearful", "奋力突围", 0.9, [],
    ),
    ("cult_ambush", "captured"): (
        "你被缚住双手押到祭坛前。黑袍祭司冷笑：「又多一个祭品——虚空之主会感谢你的血肉。」",
        "fearful", "被押到祭坛", 0.85, [],
    ),
    ("altar_room", "seal"): (
        "你一把夺过虚空之钥，注入月长石的净化之力。封印轰然亮起，黑袍人哀嚎着退散，月光重新灌满整座祭坛。",
        "happy", "夺过虚空之钥", 0.95, ["knowledge_003", "knowledge_006"],
    ),
    ("altar_room", "betray"): (
        "你转身接过黑袍，站在祭司身侧。虚空之钥被送入祭坛，天空骤然暗下来——银月城的月光，正在一寸寸熄灭。",
        "sad", "接过黑袍", 0.95, [],
    ),
    ("altar_room", "rescue"): (
        "你砍断囚笼的锁链，救出瑟瑟发抖的商队幸存者，随后将虚空之钥掷入祭坛火焰。炸裂的圣光吞噬了教团，也点燃了整片黑森林的夜空。",
        "happy", "斩断锁链", 0.95, [],
    ),
}

# 基础剧情选项回复（保底）：(dialogue, emotion, action, confidence, cited_knowledge_ids)
CHOICE_REPLIES_BASE: dict[tuple[str, str], tuple[str, str, str, float, list[str]]] = {
    ("city_gate", "ask_guard"): (
        "巴顿上下打量你一番：「外乡人？报上名来，说明来意。最近商队失踪的事闹得紧，别添乱。」",
        "neutral", "上下打量", 0.9, [],
    ),
    ("city_gate", "bribe_guard"): (
        "你递出金币，巴顿的目光在钱袋上停了一瞬，最终侧身让开：「……进去吧。今天你没见过我。」",
        "suspicious", "侧身让开", 0.9, [],
    ),
    ("city_gate", "sneak_past"): (
        "「站住！干什么的？」巡逻的守卫立刻围了上来，一把按住你的肩膀。",
        "surprised", "被一把按住", 0.9, [],
    ),
    ("city_gate", "show_badge"): (
        "巴顿看见徽章，立刻立正敬礼：「大人！请进，城主正在市政厅等您。」",
        "neutral", "立正敬礼", 0.95, [],
    ),
    ("guard_dialogue", "ask_caravan"): (
        "巴顿压低声音：「翡翠、铁盾、星光三支商队都栽在黑森林。星光商队三天前失踪，最后见他们的是守林人。」",
        "suspicious", "压低声音", 0.9, ["knowledge_004"],
    ),
    ("guard_dialogue", "ask_danger"): (
        "巴顿皱眉：「黑森林雾气常年不散，进去十个人，能出来七个都算运气。连猎户都不愿靠近。」",
        "suspicious", "皱眉", 0.9, ["knowledge_005"],
    ),
    ("guard_dialogue", "salute"): (
        "你行了一个标准军礼，巴顿眼神松动，回礼放行：「是条汉子。进城吧。」",
        "neutral", "回礼放行", 0.9, [],
    ),
    ("guard_bribed", "enter"): (
        "你快步走进城门，身后的巴顿像什么都没发生过一样继续站岗。",
        "neutral", "快步进城", 0.9, [],
    ),
    ("guard_bribed", "ask_secret"): (
        "巴顿四下张望，声音压到最低：「影子教团的人，最近总在黑森林边缘活动。别告诉别人是我说的。」",
        "suspicious", "四下张望", 0.9, ["knowledge_002"],
    ),
    ("caught_sneaking", "apologize"): (
        "你连忙道歉并说明来意，守卫将信将疑地松开你：「再有下次，直接送地牢。」",
        "neutral", "松开你", 0.85, [],
    ),
    ("caught_sneaking", "fight"): (
        "你挣脱后与守卫扭打起来，城门顿时乱作一团，更多守卫冲了过来。",
        "angry", "扭打在一起", 0.85, [],
    ),
    ("caravan_info", "go_tavern"): (
        "你谢过巴顿，转身走向醉龙酒馆——那里的老板玛格丽特什么都知道。",
        "neutral", "转身走向酒馆", 0.9, [],
    ),
    ("caravan_info", "go_forest"): (
        "你决定直奔黑森林。城门外，雾气已经在林线边缘翻涌。",
        "neutral", "望向远方", 0.9, [],
    ),
    ("forest_warning", "go_tavern"): (
        "你决定先听听酒馆里的消息再做打算。",
        "neutral", "点点头", 0.9, [],
    ),
    ("forest_warning", "go_forest"): (
        "你谢过警告，还是走向黑森林。身后传来巴顿的叹息。",
        "neutral", "迈步向前", 0.9, [],
    ),
    ("guard_secret", "go_townhall"): (
        "你赶往市政厅，要把影子教团的消息报告给城主。",
        "neutral", "快步离开", 0.9, [],
    ),
    ("guard_secret", "go_tavern"): (
        "你走进醉龙酒馆，想探探影子教团的风声。",
        "neutral", "推门而入", 0.9, [],
    ),
    ("guard_fight", "surrender"): (
        "你被守卫按在地上，押着进了城。巴顿冷冷道：「地牢里清醒一夜，明天再说话。」",
        "angry", "被押进城", 0.85, [],
    ),
    ("guard_fight", "win"): (
        "你放倒两名守卫冲进城内，身后响起示警的号角声。",
        "angry", "冲进城内", 0.85, [],
    ),
    ("city_plaza", "go_tavern"): (
        "你推开醉龙酒馆的木门，喧闹声扑面而来，玛格丽特笑着招呼你坐下。",
        "happy", "推门而入", 0.9, [],
    ),
    ("city_plaza", "go_townhall"): (
        "你穿过广场走向市政厅，台阶上的书记官正抱着一摞卷宗匆匆走过。",
        "neutral", "穿过广场", 0.9, [],
    ),
    ("city_plaza", "go_forest"): (
        "你出城向东，黑森林的轮廓在暮色中越来越近。",
        "neutral", "出城向东", 0.9, [],
    ),
    ("city_plaza", "visit_library"): (
        "你推开月神图书馆的铜门，管理员伊莲抬头推了推眼镜：「欢迎，冒险者。这里藏着银月城千年的记忆。」",
        "neutral", "推了推眼镜", 0.9, [],
    ),
    ("library", "read_history"): (
        "伊莲递来一卷泛黄的羊皮纸：「第一纪元，月神赐福，七位贤者铸造圣物封印虚空之主——那是银月城传说的开端。」",
        "neutral", "递来卷轴", 0.95, ["knowledge_007"],
    ),
    ("library", "find_map"): (
        "你在舆图架下层翻出一张泛黄的地图，标注着黑森林深处的上古废墟与一条隐秘通道。",
        "neutral", "展开地图", 0.9, ["knowledge_009"],
    ),
    ("library_lore", "ask_seals"): (
        "你追问圣物下落，伊莲压低声音：「虚空之钥就在黑森林的废墟里，教团想用它打开虚空之门。其余圣物分散在大陆各处。」",
        "suspicious", "压低声音", 0.95, ["knowledge_008", "knowledge_003"],
    ),
    ("library_lore", "leave"): (
        "你谢过伊莲，合上卷轴，走向醉龙酒馆。她望着你的背影低语：「愿月神指引你。」",
        "neutral", "合上卷轴", 0.9, [],
    ),
    ("library_map", "follow_map"): (
        "你收好地图，按标记穿过黑森林的浓雾，上古废墟的轮廓逐渐显现。",
        "neutral", "收好地图", 0.9, ["knowledge_009"],
    ),
    ("library_map", "report_mayor"): (
        "你带着地图赶到市政厅，城主凝视着标注的废墟入口，眉头紧锁。",
        "neutral", "呈上地图", 0.9, [],
    ),
    ("tavern", "buy_info"): (
        "玛格丽特收下金币，压低声音：「月长石交易出了问题，账本上有个月牙标记，和影子教团脱不了干系。」",
        "suspicious", "压低声音", 0.9, ["knowledge_006", "knowledge_002"],
    ),
    ("tavern", "ask_rumors"): (
        "酒客们醉醺醺地议论着：「黑森林里闹鬼」「有人看见黑袍人」「星光商队根本没走出林子」……玛格丽特朝你使了个眼色。",
        "neutral", "使了个眼色", 0.9, ["knowledge_004"],
    ),
    ("tavern", "drink"): (
        "一大杯麦酒下肚，暖意上涌，你听清了邻桌的低语：「废墟、祭坛、七件圣物……」",
        "happy", "畅饮麦酒", 0.85, [],
    ),
    ("tavern_info", "go_townhall"): (
        "你带着酒馆的情报赶往市政厅。",
        "neutral", "快步离开", 0.9, [],
    ),
    ("tavern_info", "go_forest"): (
        "你按玛格丽特的线索前往黑森林，腰间的水壶随着脚步叮当作响。",
        "neutral", "动身出发", 0.9, [],
    ),
    ("tavern_rumors", "go_townhall"): (
        "你把听到的谣言整理一番，快步走向市政厅。",
        "neutral", "整理思绪", 0.9, [],
    ),
    ("tavern_rumors", "go_forest"): (
        "你决定顺着谣言去黑森林一探究竟。",
        "neutral", "走向城门", 0.9, [],
    ),
    ("tavern_drunk", "sleep_off"): (
        "你在桌上睡了一觉，醒来时酒馆只剩烛火。玛格丽特留下纸条：「线索在黑森林，小心影子教团。」",
        "neutral", "揉着额头醒来", 0.85, ["knowledge_002"],
    ),
    ("tavern_drunk", "order_coffee"): (
        "一杯醒酒咖啡下肚，你清醒了不少，玛格丽特顺势低声说起她的情报。",
        "neutral", "啜饮咖啡", 0.9, [],
    ),
    ("townhall", "meet_mayor"): (
        "你被领进城主办公室。城主放下鹅毛笔，锐利的目光审视着你：「说说你知道的。」",
        "neutral", "审视着你", 0.9, [],
    ),
    ("townhall", "check_records"): (
        "你溜进档案室，灰尘呛得你直打喷嚏。卷宗里或许藏着什么。",
        "neutral", "翻找卷宗", 0.9, [],
    ),
    ("mayor_office", "accept_mission"): (
        "城主点头：「很好。商队和月长石交易都要查清。带上这封信，必要时可以调动城门卫。」",
        "neutral", "递出一封信", 0.95, [],
    ),
    ("mayor_office", "decline"): (
        "你婉拒任务，城主挑眉：「那至少去档案室看看账本，别浪费我的时间。」",
        "neutral", "挑了挑眉", 0.9, [],
    ),
    ("town_records", "find_ledger"): (
        "你在积灰的账本里翻到一笔可疑的月长石交易，买家的署名只有一个月牙标记。",
        "suspicious", "翻动账本", 0.9, ["knowledge_006"],
    ),
    ("town_records", "leave"): (
        "你放下卷宗离开市政厅，决定直接去黑森林。",
        "neutral", "放下卷宗", 0.9, [],
    ),
    ("ledger_clue", "go_forest"): (
        "你收起账本线索，动身前往黑森林。",
        "neutral", "收起账本", 0.9, [],
    ),
    ("ledger_clue", "investigate_cult"): (
        "你顺着月牙标记追查，发现它与影子教团的徽记一模一样。",
        "suspicious", "对照徽记", 0.9, ["knowledge_002"],
    ),
    ("forest", "enter_deep"): (
        "你拨开浓雾踏入森林，脚下是松软的腐叶，远处传来低沉的嗡鸣。",
        "neutral", "拨开浓雾", 0.9, [],
    ),
    ("forest", "follow_river"): (
        "你沿河流前进，水声掩盖了脚步声。岸边的泥地上留着拖曳的痕迹。",
        "neutral", "沿河前进", 0.9, [],
    ),
    ("forest_camp", "investigate"): (
        "你搜遍营地，找到翡翠商队的货物残骸和拖向废墟的痕迹。",
        "suspicious", "搜查营地", 0.9, ["knowledge_004"],
    ),
    ("forest_camp", "search_tracks"): (
        "你循着拖曳痕迹前进，发现一串黑袍脚印，尽头通向废墟。",
        "suspicious", "蹲下查看", 0.9, [],
    ),
    ("cult_trail", "follow_trail"): (
        "你沿黑袍脚印来到废墟入口，浮雕上的七件圣物图案在月光下泛着冷光。",
        "suspicious", "沿脚印前进", 0.9, ["knowledge_003"],
    ),
    ("cult_trail", "study_map"): (
        "你对照地图确认了废墟的入口——这是上古祭坛的遗迹。",
        "neutral", "对照地图", 0.9, [],
    ),
    ("ruins", "search_altar"): (
        "你从侧廊潜入，祭坛中央悬浮着虚空之钥，黑袍人正在进行献祭。",
        "fearful", "屏息潜入", 0.9, ["knowledge_003"],
    ),
    ("ruins", "ambush_cult"): (
        "你伏击了一个落单的黑袍人，逼问出祭坛的位置。他临死前低语：「虚空之主……降临……」",
        "fearful", "勒住黑袍人", 0.9, ["knowledge_002"],
    ),
    ("cult_ambush", "break_free"): (
        "你奋力突围，冲进祭坛大厅。黑袍人在你身后嘶吼着追来。",
        "fearful", "奋力突围", 0.9, [],
    ),
    ("cult_ambush", "captured"): (
        "你被缚住双手押到祭坛前。黑袍祭司冷笑：「多一个祭品，正好。」",
        "fearful", "被押到祭坛", 0.85, [],
    ),
    ("altar_room", "seal"): (
        "你一把夺过虚空之钥，注入月长石的净化之力。封印重新亮起，黑袍人哀嚎着退散。",
        "happy", "夺过虚空之钥", 0.95, ["knowledge_003", "knowledge_006"],
    ),
    ("altar_room", "betray"): (
        "你转身加入教团，接过黑袍。虚空之钥被送入祭坛，银月城的天色暗了下来。",
        "sad", "接过黑袍", 0.95, [],
    ),
    ("altar_room", "rescue"): (
        "你砍断囚笼的锁链救出商队幸存者，随后将虚空之钥投入祭坛火焰，炸裂的圣光吞噬了教团。",
        "happy", "斩断锁链", 0.95, [],
    ),
}

# 终局节点收尾对话
END_REPLIES: dict[str, tuple[str, str, str, float, list[str]]] = {
    "ending_hero": (
        "故事结束了——你取回虚空之钥并加固封印，银月城从此恢复宁静。多年后，人们把这段故事称作「月光纪元」。",
        "happy", "望向星空", 1.0, [],
    ),
    "ending_betrayal": (
        "故事结束了——你与教团同流合污，银月城陷入长夜。月长石的光辉一天天黯淡，而你在黑暗里听见虚空之主的低笑。",
        "sad", "垂下眼帘", 1.0, [],
    ),
    "ending_rescuer": (
        "故事结束了——你救出商队幸存者并销毁虚空之钥，月光重新照亮黑森林。幸存者们说，那晚的月亮比任何时候都亮。",
        "happy", "松了口气", 1.0, [],
    ),
}


def _mock_embedding(text: str) -> list[float]:
    """SHA-256 种子生成 VECTOR_DIM 维标准正态向量并 L2 归一化。"""
    digest = hashlib.sha256(text.encode("utf-8")).digest()
    seed = int.from_bytes(digest[:8], "little")
    rng = np.random.default_rng(seed)
    vector = rng.standard_normal(config.VECTOR_DIM)
    norm = float(np.linalg.norm(vector))
    if norm == 0.0:
        return [0.0] * config.VECTOR_DIM
    return (vector / norm).tolist()


class LLMClient:
    def __init__(self) -> None:
        self.api_key = config.OPENAI_API_KEY
        self.model = config.OPENAI_MODEL
        self.base_url = config.OPENAI_BASE_URL
        self.use_mock = config.USE_MOCK
        self._client: httpx.AsyncClient | None = None
        if not self.use_mock:
            self._client = httpx.AsyncClient(timeout=httpx.Timeout(60.0, connect=10.0))

    async def close(self) -> None:
        if self._client:
            await self._client.aclose()

    async def chat_completion(
        self,
        system_prompt: str,
        user_message: str,
        temperature: float = 0.7,
    ) -> NPCResponse:
        """
        调用大模型或返回 Mock 回复。

        Mock 模式逻辑：
        1. user_message 转小写后检查关键词：
           - 含 "进城" 或 "调查" 或 "失踪" → dialogue="城门守卫打量了你一眼...",
             emotion="suspicious", selected_choice_id="ask_guard",
             memory_updates=["玩家对失踪商队感兴趣"], cited_knowledge_ids=["knowledge_001"], confidence=0.9
           - 含 "谢谢" 或 "感谢" → dialogue="『不必谢我...』", emotion="neutral", confidence=0.8
           - 其他 → dialogue="『嗯……我不太确定你在说什么。』", emotion=random.choice([...]), confidence=0.5
        2. suggested_choices 从 system_prompt 中解析可用选项（如果 prompt 里有）

        真实模式逻辑：
        1. POST {base_url}/chat/completions
        2. payload 包含 model, messages, temperature, response_format={"type": "json_object"}, max_tokens=800
        3. 解析 choices[0].message.content 为 JSON
        4. 用 NPCResponse.model_validate() 反序列化
        5. 任何异常 → 打印 warning → 降级到 Mock
        """
        if self.use_mock:
            return self._mock_response(system_prompt, user_message)
        try:
            payload = {
                "model": self.model,
                "messages": [
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_message},
                ],
                "temperature": temperature,
                "response_format": {"type": "json_object"},
                "max_tokens": 800,
            }
            headers: dict[str, str] = {}
            if self.api_key:
                headers["Authorization"] = f"Bearer {self.api_key}"
            response = await self._client.post(
                f"{self.base_url}/chat/completions",
                json=payload,
                headers=headers,
            )
            response.raise_for_status()
            data = response.json()
            content = data["choices"][0]["message"]["content"]
            parsed = json.loads(content)
            return NPCResponse.model_validate(parsed)
        except Exception as exc:
            logger.warning("LLM 请求失败（%s），降级到 Mock 回复", exc)
            return self._mock_response(system_prompt, user_message)

    def _mock_response(self, system_prompt: str, user_message: str) -> NPCResponse:
        lowered = user_message.lower()
        suggested_choices = _parse_suggested_choices(system_prompt)
        npc_name = _extract_npc_name(system_prompt)
        node_id, chosen_id, chosen_text = _parse_mock_context(system_prompt)

        # 玩家刚选择了某个剧情选项：返回与该节点/选项匹配的剧情回复
        if chosen_id:
            reply = CHOICE_REPLIES.get((node_id, chosen_id)) or CHOICE_REPLIES_BASE.get(
                (node_id, chosen_id)
            )
            if reply is not None:
                dialogue, emotion, action, confidence, cited = reply
                return NPCResponse(
                    dialogue=dialogue,
                    emotion=emotion,
                    action=action,
                    confidence=confidence,
                    cited_knowledge_ids=cited,
                    suggested_choices=suggested_choices,
                )
            return NPCResponse(
                dialogue=f"{npc_name}点点头：「你选择了「{chosen_text}」。我们继续。」",
                emotion="neutral",
                action="点了点头",
                confidence=0.85,
                suggested_choices=suggested_choices,
            )

        # 到达终局节点：返回与结局相符的收尾对话
        if node_id in END_REPLIES:
            dialogue, emotion, action, confidence, cited = END_REPLIES[node_id]
            return NPCResponse(
                dialogue=dialogue,
                emotion=emotion,
                action=action,
                confidence=confidence,
                cited_knowledge_ids=cited,
                suggested_choices=suggested_choices,
            )

        if any(keyword in lowered for keyword in ("进城", "调查", "失踪")):
            return NPCResponse(
                dialogue="城门守卫打量了你一眼：『最近商队失踪得蹊跷，要查的话，先去找守卫长吧。』",
                emotion="suspicious",
                action="警惕地打量着来客",
                selected_choice_id="ask_guard",
                memory_updates=["玩家对失踪商队感兴趣"],
                cited_knowledge_ids=["knowledge_001"],
                confidence=0.9,
                suggested_choices=suggested_choices,
            )
        if any(keyword in lowered for keyword in ("谢谢", "感谢")):
            return NPCResponse(
                dialogue="『不必谢我……你多保重。』",
                emotion="neutral",
                action="摆了摆手",
                confidence=0.8,
                suggested_choices=suggested_choices,
            )
        if any(keyword in lowered for keyword in ("你好", "您好", "hello", "hi", "嗨")):
            return NPCResponse(
                dialogue=f"{npc_name}微微颔首：「你好，旅行者。欢迎来到银月城，愿月光指引你。」",
                emotion="neutral",
                action="微微颔首",
                confidence=0.9,
                suggested_choices=suggested_choices,
            )
        if any(keyword in user_message for keyword in ("你是谁", "你叫什么", "名字")):
            return NPCResponse(
                dialogue=(
                    f"{npc_name}挺直腰板：「我是{npc_name}。看你面生，是外地来的冒险者吧？"
                    "城门最近可不太平。」"
                ),
                emotion="neutral",
                action="打量着你",
                confidence=0.9,
                suggested_choices=suggested_choices,
            )
        if any(keyword in user_message for keyword in ("银月城", "世界", "历史", "知识")):
            return NPCResponse(
                dialogue=(
                    f"{npc_name}指了指城墙上泛着银光的砖石：「银月城建于第一纪元，"
                    "城墙由月长石砌成，满月之夜会发出柔和的银光。」"
                ),
                emotion="neutral",
                action="指向城墙",
                cited_knowledge_ids=["knowledge_001"],
                confidence=0.9,
                suggested_choices=suggested_choices,
            )
        if any(keyword in user_message for keyword in ("黑森林", "森林")):
            return NPCResponse(
                dialogue=(
                    f"{npc_name}压低声音：「黑森林常年被黑色雾气笼罩，最近三支商队"
                    "都在那边失踪了。要查的话，得做好万全准备。」"
                ),
                emotion="suspicious",
                action="压低声音",
                cited_knowledge_ids=["knowledge_005"],
                confidence=0.85,
                suggested_choices=suggested_choices,
            )
        if any(keyword in user_message for keyword in ("任务", "剧情", "商队", "下一步", "怎么办", "线索")):
            return NPCResponse(
                dialogue=(
                    f"{npc_name}点点头：「商队失踪的线索指向黑森林。你可以先向守卫打听细节，"
                    "或者去醉龙酒馆听些消息。」"
                ),
                emotion="neutral",
                action="点了点头",
                cited_knowledge_ids=["knowledge_004"],
                confidence=0.85,
                suggested_choices=suggested_choices,
            )
        emotion = random.choice(["neutral", "suspicious", "surprised", "sad"])
        return NPCResponse(
            dialogue="『嗯……我不太确定你在说什么。』",
            emotion=emotion,
            action="歪了歪头",
            confidence=0.5,
            suggested_choices=suggested_choices,
        )

    async def get_embedding(self, text: str) -> list[float]:
        """
        获取文本向量。

        Mock 模式：
        1. SHA-256(text.encode("utf-8")) 取前 8 字节作为 numpy 种子
        2. np.random.default_rng(seed) 生成 config.VECTOR_DIM 个标准正态分布数
        3. L2 归一化（范数为 0 时返回全 0 向量）
        4. 返回 Python list[float]

        真实模式：
        1. POST /embeddings, model="text-embedding-3-small", input=text
        2. 解析 data[0].embedding
        3. 异常 → 降级 Mock
        """
        if self.use_mock:
            return _mock_embedding(text)
        try:
            payload = {"model": "text-embedding-3-small", "input": text}
            headers: dict[str, str] = {}
            if self.api_key:
                headers["Authorization"] = f"Bearer {self.api_key}"
            response = await self._client.post(
                f"{self.base_url}/embeddings",
                json=payload,
                headers=headers,
            )
            response.raise_for_status()
            data = response.json()
            return [float(value) for value in data["data"][0]["embedding"]]
        except Exception as exc:
            logger.warning("Embedding 请求失败（%s），降级到 Mock 向量", exc)
            return _mock_embedding(text)


# 全局单例
_llm_client: LLMClient | None = None


def get_llm_client() -> LLMClient:
    global _llm_client
    if _llm_client is None:
        _llm_client = LLMClient()
    return _llm_client
