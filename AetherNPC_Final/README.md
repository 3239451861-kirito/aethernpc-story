# AetherNPC —— AIGC 智能 NPC 对话系统

> 生成可直接运行、pytest 全绿、零 warning、带完整前端 Demo 的生产级项目。

AetherNPC 是一个基于 **FastAPI + Pydantic v2 + SQLite + WebSocket** 的智能 NPC 对话系统。系统采用分层架构、单例模式与依赖注入，核心原则是：**大模型只给建议，服务端拥有最终决策权**。

完整的银月城世界观设定见 [WORLDVIEW.md](WORLDVIEW.md)：第一纪元、月长石、七件圣物、影子教团、黑森林上古废墟，以及英雄/背叛/救赎三个结局。

项目当前包含 **13 位 NPC**、**36 个剧情节点**、**76 个剧情选项**，并配有城门、图书馆、月神殿、矿区、守林人五条可探索支线。

## 特性

- **分层架构**：配置中心 / 数据模型 / Prompt 模板 / LLM 客户端 / RAG 检索 / 会话记忆 / 剧情状态机 / 业务编排 / API 入口
- **剧情选项系统**：剧情图由节点与选项构成，支持前置条件（`memory.*` / `relationship.*`）与效果（记忆、好感度变更）
- **会话记忆**：`MemoryManager` 管理上下文、关系值、历史与覆盖统计，线程安全、支持 `SESSION_TIMEOUT` 超时清理
- **LLM Mock 降级**：`OPENAI_API_KEY` 留空即启用 Mock；`chat_completion` 统一返回 `NPCResponse`，真实 API 调用失败自动降级，服务永不崩溃
- **RAG 向量检索**：SQLite 持久化知识库 + 余弦相似度检索，Mock 模式生成 `VECTOR_DIM` 维确定性向量，支持 `domain_filter`（metadata.category）过滤
- **AI Agent 遍历测试**：DFS 回溯自动遍历剧情图，输出节点/边覆盖率、不可达节点、死胡同与非法跳转
- **WebSocket 前端 Demo**：零外部依赖单文件页面，支持剧情选项点击交互

## 快速开始

```powershell
# 1. 创建虚拟环境并安装依赖
python -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r requirements.txt

# 2. 初始化数据（NPC 人设 / 剧情图 / 知识库）
.\.venv\Scripts\python.exe -m scripts.seed_knowledge --force

# 3. 运行测试（全绿、零 warning、覆盖率 ≥80%）
.\.venv\Scripts\python.exe -m pytest

# 4. 启动服务
.\.venv\Scripts\python.exe -m uvicorn app.main:app --host 127.0.0.1 --port 8000
```

打开 <http://127.0.0.1:8000/static/demo.html> 即可体验 WebSocket 实时对话 Demo；API 文档位于 <http://127.0.0.1:8000/docs>。

## 配置

复制 `.env.example` 为 `.env` 并按需修改：

```env
# OpenAI 配置（留空则启用 Mock 模式）
OPENAI_API_KEY=
OPENAI_MODEL=gpt-4o-mini
OPENAI_BASE_URL=https://api.openai.com/v1

# 应用配置
MAX_HISTORY_TURNS=10
SESSION_TIMEOUT=3600
LOG_LEVEL=INFO
```

`OPENAI_API_KEY` 留空时系统使用确定性 Mock 回复；配置 Key 后走真实 OpenAI 兼容接口，调用失败仍会自动降级到 Mock。

## API 一览

| 方法 | 路径 | 说明 |
| --- | --- | --- |
| GET | `/health` | 健康检查 |
| GET | `/npcs` | 获取 NPC 配置（`{"npcs": [...]}`） |
| GET | `/story/graph` | 获取剧情图 |
| POST | `/sessions` | 创建会话（`player_id` 查询参数，默认 `player-001`） |
| POST | `/dialogue` | 发起对话（返回信封式结果） |
| GET | `/sessions/{session_id}` | 获取会话状态快照 |
| POST | `/tests/run` | 运行剧情图 Agent 遍历测试 |
| WS | `/ws/{session_id}` | WebSocket 实时对话 |
| GET | `/static/demo.html` | 前端 Demo 页面 |

### POST /api/dialogue 示例

```json
{
  "session_id": "POST /api/sessions 返回的 session_id",
  "npc_id": "aether_watcher",
  "message": "我接受委托",
  "choice_id": "accept_quest"
}
```

`choice_id` 可选，作为服务端提示；模型建议（`selected_choice_id`）同样会被校验。服务端校验前置条件、应用效果（记忆/好感度）并推进剧情，返回信封：

```json
{
  "session_id": "xxx",
  "npc_id": "aether_watcher",
  "response": {
    "dialogue": "……",
    "emotion": "neutral",
    "action": "轻轻点了点头",
    "confidence": 0.9,
    "suggested_choices": [{"choice_id": "visit_forge", "text": "前往铁匠铺做准备"}]
  },
  "current_node": "call_to_action",
  "relationship": 15,
  "validated_choice": "accept_quest",
  "cited_knowledge": ["starfall_land"]
}
```

### WebSocket 协议

连接 `/ws/{session_id}` 后发送：

```json
{"npc_id": "aether_watcher", "message": "你好"}
```

发送 `{"action": "close"}` 或断开连接即可退出；接收与 `POST /dialogue` 相同的信封，校验失败时返回 `{"type": "error", "error": "..."}`。

## 剧情图格式（data/story_graph.json）

```json
{
  "start_node": "arrival",
  "nodes": {
    "arrival": {
      "node_id": "arrival",
      "description": "你抵达浮空岛艾瑟拉……",
      "choices": [
        {
          "choice_id": "accept_quest",
          "text": "接受守望者的委托",
          "preconditions": {},
          "next_node": "call_to_action",
          "effects": {"memory.quest_accepted": true, "relationship.aether_watcher": 5}
        }
      ],
      "is_end": false
    }
  }
}
```

前置条件支持精确匹配（如 `"memory.has_weapon": true`）与数值范围（如 `"relationship.aether_watcher": {"min": 15}`）；效果中 `memory.*` 写入会话记忆，`relationship.*` 按整数增量更新好感度。

## 目录结构

```text
AetherNPC/
├── app/
│   ├── config.py           # 配置中心（python-dotenv + Pydantic）
│   ├── schemas.py          # 数据模型精确定义
│   ├── prompts.py          # Prompt 模板引擎
│   ├── llm_client.py       # LLM 客户端 + Mock 降级
│   ├── rag.py              # RAG 向量检索
│   ├── memory.py           # MemoryManager 会话记忆（内存缓存 + 线程安全）
│   ├── story_engine.py     # 确定性剧情状态机（选项校验/效果/图完整性）
│   ├── services/           # 对话服务（process_dialogue 单例）
│   ├── agent_tester.py     # AI Agent 启发式剧情遍历测试
│   └── main.py             # FastAPI 入口
├── scripts/seed_knowledge.py
├── tests/                  # pytest 测试（全绿、零 warning、覆盖率 ≥80%）
├── data/                   # NPC / 剧情图 / 知识库
├── static/demo.html        # WebSocket 前端 Demo
├── requirements.txt
├── pytest.ini
└── README.md
```

## 硬约束落实清单

- Python 3.10+，全部使用 `|` 联合类型，无 `Optional` / `Union`
- Pydantic v2 API（`model_dump` / `model_validate`），所有模型均含 `model_config`
- 所有 I/O 均 `try/except Exception`，无裸 `except`，异常至少记录 `logging.warning`
- SQLite 连接、httpx client、文件句柄全部显式关闭或使用 context manager
- 全局内存状态均使用 `threading.Lock()` / `asyncio.Lock()` 保护
- 外部输入全部经过 Pydantic 校验，字符串自动 `strip()`，SQL 仅参数化查询
- 统一 `logging.getLogger(__name__)` 与日志格式
- 文件读写指定 `encoding="utf-8"`，JSON 写入指定 `ensure_ascii=False`
- 路径全部基于 `Path(__file__).resolve().parent` 相对定位，无硬编码绝对路径
- 任何外部 API 调用失败自动降级到 Mock，服务不崩溃

## 测试

```powershell
.\.venv\Scripts\python.exe -m pytest
```

`pytest.ini` 启用 `asyncio_mode = auto`、覆盖率统计与 `--cov-fail-under=80` 门槛；项目以“零 warning”为目标，任何依赖升级若引入 warning 都应在提交前处理。
