"""知识库初始化脚本：生成 data/ 下的 NPC/剧情图 JSON，并将知识文档写入 SQLite 向量库。"""

from __future__ import annotations

import argparse
import asyncio
import json
import logging
import sys
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent
if str(BASE_DIR) not in sys.path:
    sys.path.insert(0, str(BASE_DIR))

from app.config import setup_logging  # noqa: E402
from app.rag import DEFAULT_KNOWLEDGE, RAGStore  # noqa: E402
from app.services.dialogue_service import DEFAULT_NPCS  # noqa: E402
from app.story_engine import DEFAULT_STORY_GRAPH  # noqa: E402

setup_logging("INFO")
logger = logging.getLogger(__name__)


def _write_json(path: Path, payload: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as handle:
        json.dump(payload, handle, ensure_ascii=False, indent=2)


async def _seed_knowledge_db(store: RAGStore) -> None:
    for doc in DEFAULT_KNOWLEDGE:
        await store.add_document(doc.doc_id, doc.content, doc.metadata)


def seed(force: bool = False) -> int:
    data_dir = BASE_DIR / "data"
    targets = {
        "npcs.json": {"npcs": [npc.model_dump() for npc in DEFAULT_NPCS]},
        "story_graph.json": {
            "start_node": DEFAULT_STORY_GRAPH.start_node,
            "nodes": {
                node_id: node.model_dump()
                for node_id, node in DEFAULT_STORY_GRAPH.nodes.items()
            },
        },
        "knowledge.json": {"documents": [doc.model_dump() for doc in DEFAULT_KNOWLEDGE]},
    }
    for name, payload in targets.items():
        path = data_dir / name
        if path.exists() and not force:
            logger.info("已存在，跳过: %s", path)
            continue
        try:
            _write_json(path, payload)
            logger.info("已生成: %s", path)
        except Exception as exc:
            logger.error("生成 %s 失败: %s", path, exc)
            return 1
    try:
        store = RAGStore(db_path=data_dir / "npc_memory.db")
        for doc in store.list_all():
            store.delete_document(doc["doc_id"])
        asyncio.run(_seed_knowledge_db(store))
        logger.info("知识向量库已写入: %s", store.db_path)
    except Exception as exc:
        logger.error("知识向量库写入失败: %s", exc)
        return 1
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(description="初始化 AetherNPC 数据目录")
    parser.add_argument("--force", action="store_true", help="强制覆盖已有文件")
    args = parser.parse_args()
    return seed(force=args.force)


if __name__ == "__main__":
    raise SystemExit(main())
