# AetherNPC —— AIGC 智能 NPC 对话系统 / AIGC-Powered NPC Dialogue Engine

![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)
![Python](https://img.shields.io/badge/Python-3.10%2B-blue.svg)
![FastAPI](https://img.shields.io/badge/FastAPI-0.110%2B-009688.svg)
![WebSocket](https://img.shields.io/badge/WebSocket-Realtime-4A90D9.svg)
![Release](https://img.shields.io/badge/Release-v0.1.0-blue.svg)

> 面向 RPG / 开放世界游戏的智能 NPC 对话与分支剧情引擎。核心原则：**大模型只给建议，服务端拥有最终决策权。**
>
> An intelligent NPC dialogue and branching-story engine for RPG games, built on **FastAPI + Pydantic v2 + SQLite + WebSocket**. The LLM suggests — the server decides.

## 项目简介 / About

AetherNPC 是一个生产级、可测试的 AIGC 智能 NPC 对话系统：服务端以确定性剧情状态机（剧情图）控制分支走向，LLM 负责生成符合人设的对话回复，RAG 负责从知识库检索设定，会话记忆追踪玩家关系与历史。项目内置 13 位 NPC、36 个剧情节点、76 个剧情选项，以及五条可探索支线（城门、图书馆、月神殿、矿区、守林人）。

AetherNPC is a production-ready framework for LLM-driven NPC conversations. A deterministic story state machine controls branching outcomes; the LLM generates in-character dialogue; a SQLite-backed RAG retrieves world-lore; session memory tracks relationships and history. Ships with 13 NPCs, 36 story nodes, 76 choices, and 5 explorable side quests.

## 核心特性 / Features

- **确定性剧情状态机** / Deterministic branching story engine with preconditions (`memory.*` / `relationship.*`) and effects
- **RAG 知识检索** / SQLite-backed knowledge retrieval with cosine similarity and domain filters
- **会话记忆** / Thread-safe session memory with relationship values and `SESSION_TIMEOUT` cleanup
- **LLM Mock 降级** / Automatic mock fallback — leave `OPENAI_API_KEY` empty to run fully offline, and the service never crashes on API failures
- **AI Agent 遍历测试** / DFS agent traverses the story graph and reports coverage, unreachable nodes and illegal jumps
- **WebSocket 前端 Demo** / Zero-dependency single-file front-end with real-time interactive dialogue
- **测试保障** / pytest suite with `asyncio_mode = auto` and ≥80% coverage gate, zero-warning target

## 快速开始 / Quick Start

代码位于 [`AetherNPC_Final/`](AetherNPC_Final/)，详细的中文文档（配置、API 一览、WebSocket 协议、剧情图格式）见 [AetherNPC_Final/README.md](AetherNPC_Final/README.md)。

```powershell
cd AetherNPC_Final
python -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
.\.venv\Scripts\python.exe -m scripts.seed_knowledge --force
.\.venv\Scripts\python.exe -m pytest
.\.venv\Scripts\python.exe -m uvicorn app.main:app --host 127.0.0.1 --port 8000
```

打开 <http://127.0.0.1:8000/static/demo.html> 体验 WebSocket 实时对话 Demo；API 文档位于 <http://127.0.0.1:8000/docs>。

## 目录结构 / Structure

```text
aethernpc-story/
├── LICENSE                 # MIT
├── README.md               # 本文件
└── AetherNPC_Final/        # 项目主体（FastAPI 应用 / 剧情数据 / 测试 / Demo）
    ├── app/                # 分层架构：配置 / 模型 / Prompt / LLM / RAG / 记忆 / 剧情引擎 / 服务 / API
    ├── data/               # NPC 人设、剧情图、知识库
    ├── scripts/            # 数据初始化脚本
    ├── static/demo.html    # WebSocket 前端 Demo
    ├── tests/              # pytest 测试套件
    └── README.md           # 详细中文文档
```

## 许可 / License

本项目基于 [MIT License](LICENSE) 开源。
