"""RAG：SQLite 持久化 + 余弦相似度向量检索"""

import sqlite3
import json
import logging
from pathlib import Path
from typing import Any

import numpy as np

from app import config
from app.llm_client import get_llm_client
from app.schemas import KnowledgeDoc

logger = logging.getLogger(__name__)

DEFAULT_KNOWLEDGE: list[KnowledgeDoc] = [
    KnowledgeDoc(
        doc_id="knowledge_001",
        content="银月城是大陆上最古老的城市之一，建于第一纪元。城墙由月长石砌成，在满月之夜会发出柔和的银光。",
        metadata={"category": "地理", "region": "银月城"},
    ),
    KnowledgeDoc(
        doc_id="knowledge_002",
        content="影子教团是一个崇拜虚空之主的秘密组织，成员身穿黑袍，相信通过活人献祭可以打开通往虚空维度的大门。",
        metadata={"category": "组织", "danger_level": "极高"},
    ),
    KnowledgeDoc(
        doc_id="knowledge_003",
        content="虚空之钥是一件上古神器，据说是封印虚空之主的七件圣物之一。它的下落已经失传了数百年。",
        metadata={"category": "神器", "era": "上古时代"},
    ),
    KnowledgeDoc(
        doc_id="knowledge_004",
        content="失踪的商队包括翡翠商队（12人）、铁盾商队（8人）、星光商队（15人）。最后一批星光商队在三天前失踪于黑森林。",
        metadata={"category": "事件", "status": "调查中"},
    ),
    KnowledgeDoc(
        doc_id="knowledge_005",
        content="黑森林位于银月城以东，常年被黑色雾气笼罩。传说森林深处有一座上古废墟，里面埋藏着不为人知的秘密。",
        metadata={"category": "地理", "danger_level": "高"},
    ),
    KnowledgeDoc(
        doc_id="knowledge_006",
        content="月长石是一种稀有的魔法矿石，能够吸收月光并释放净化之力。银月城的城墙就是用月长石建造的。",
        metadata={"category": "矿物", "magic": True},
    ),
    KnowledgeDoc(
        doc_id="knowledge_007",
        content="第一纪元是月神赐福的时代，七位贤者铸造了七件圣物，用以封印虚空之主，大陆自此进入长久的安宁。",
        metadata={"category": "历史", "era": "第一纪元"},
    ),
    KnowledgeDoc(
        doc_id="knowledge_008",
        content="七件圣物分别是虚空之钥、晨曦之镜、潮汐之笛、烬灭之锤、星语之冠、影缚之环与月辉之盾，如今分散在大陆各处。",
        metadata={"category": "神器", "era": "第一纪元"},
    ),
    KnowledgeDoc(
        doc_id="knowledge_009",
        content="黑森林深处的上古废墟是七贤者封印虚空之主的祭坛遗迹，浮雕上记录着当年的战斗，也警示着封印的脆弱。",
        metadata={"category": "遗迹", "region": "黑森林"},
    ),
    KnowledgeDoc(
        doc_id="knowledge_010",
        content="银月城的月神信仰源远流长，满月之夜居民会聚集在月长石祭坛前祈祷净化，月长石也因此被视为月神的眼泪。",
        metadata={"category": "传说", "region": "银月城"},
    ),
    KnowledgeDoc(
        doc_id="knowledge_011",
        content="守林人是世代守护黑森林边界的猎人团体，熟悉林中的每一处陷阱与道路，也是商队失踪案最早的目击者。",
        metadata={"category": "组织", "region": "黑森林"},
    ),
    KnowledgeDoc(
        doc_id="knowledge_012",
        content="醉龙酒馆是银月城消息最灵通的地方，老板玛格丽特掌握着商队、教团与月长石交易的种种风声。",
        metadata={"category": "组织", "region": "银月城"},
    ),
    KnowledgeDoc(
        doc_id="knowledge_013",
        content="月神殿是银月城最古老的建筑，祭司世代守护月神祭坛，满月祈福能安抚人心，也能为出征者赐下月光祝福。",
        metadata={"category": "传说", "region": "银月城"},
    ),
    KnowledgeDoc(
        doc_id="knowledge_014",
        content="银月矿坑出产大陆最纯净的月长石，可近来大量矿石被运往黑森林方向，收货方只留下一个月牙印记。",
        metadata={"category": "矿物", "region": "银月城"},
    ),
]


def cosine_similarity(a: list[float], b: list[float]) -> float:
    """
    计算余弦相似度。
    边界：a 或 b 为空列表，或全零向量 → 返回 0.0
    """
    if not a or not b:
        return 0.0
    a_vec = np.array(a, dtype=np.float32)
    b_vec = np.array(b, dtype=np.float32)
    dot = float(np.dot(a_vec, b_vec))
    norm_a = float(np.linalg.norm(a_vec))
    norm_b = float(np.linalg.norm(b_vec))
    if norm_a == 0.0 or norm_b == 0.0:
        return 0.0
    return dot / (norm_a * norm_b)


class RAGStore:
    def __init__(self, db_path: Path | None = None) -> None:
        self.db_path = Path(db_path) if db_path else config.DB_PATH
        self._init_db()

    def _init_db(self) -> None:
        """建表 knowledge(doc_id TEXT PRIMARY KEY, content TEXT, metadata TEXT, embedding TEXT)"""
        try:
            self.db_path.parent.mkdir(parents=True, exist_ok=True)
            connection = sqlite3.connect(str(self.db_path), check_same_thread=False)
            try:
                with connection:
                    connection.execute(
                        "CREATE TABLE IF NOT EXISTS knowledge ("
                        " doc_id TEXT PRIMARY KEY,"
                        " content TEXT NOT NULL,"
                        " metadata TEXT NOT NULL,"
                        " embedding TEXT NOT NULL)"
                    )
            finally:
                connection.close()
        except Exception as exc:
            logger.warning("RAG 数据库初始化失败: %s", exc)

    def _connect(self) -> sqlite3.Connection:
        connection = sqlite3.connect(str(self.db_path), check_same_thread=False)
        connection.row_factory = sqlite3.Row
        return connection

    async def add_document(
        self,
        doc_id: str,
        content: str,
        metadata: dict[str, Any] | None = None,
    ) -> None:
        """
        添加文档。
        1. 调用 get_llm_client().get_embedding(content) 获取向量
        2. INSERT OR REPLACE INTO knowledge VALUES (?, ?, ?, ?)
        3. 参数化查询，禁止字符串拼接 SQL
        """
        embedding = await get_llm_client().get_embedding(content)
        metadata = metadata or {}
        try:
            connection = self._connect()
            try:
                with connection:
                    connection.execute(
                        "INSERT OR REPLACE INTO knowledge (doc_id, content, metadata, embedding)"
                        " VALUES (?, ?, ?, ?)",
                        (
                            doc_id,
                            content,
                            json.dumps(metadata, ensure_ascii=False),
                            json.dumps(embedding),
                        ),
                    )
            finally:
                connection.close()
        except Exception as exc:
            logger.warning("知识文档写入失败: %s", exc)

    def search_by_embedding(
        self,
        query_embedding: list[float],
        top_k: int = 3,
        domain_filter: str | None = None,
    ) -> list[dict[str, Any]]:
        """
        向量检索。
        1. SELECT * FROM knowledge
        2. 逐条解析 embedding JSON，计算 cosine_similarity
        3. 如果 domain_filter 不为 None，只保留 metadata['category'] == domain_filter 的文档
        4. 按 score 降序，返回前 top_k 个
        5. 每条结果格式：{"doc_id": str, "content": str, "metadata": dict, "score": float}
        """
        try:
            connection = self._connect()
            try:
                with connection:
                    rows = connection.execute("SELECT * FROM knowledge").fetchall()
            finally:
                connection.close()
        except Exception as exc:
            logger.warning("向量检索失败: %s", exc)
            return []
        results: list[dict[str, Any]] = []
        for row in rows:
            try:
                metadata = json.loads(row["metadata"])
                embedding = json.loads(row["embedding"])
            except Exception as exc:
                logger.warning("知识文档数据损坏，已跳过 %s: %s", row["doc_id"], exc)
                continue
            if domain_filter is not None and metadata.get("category") != domain_filter:
                continue
            score = cosine_similarity(query_embedding, embedding)
            results.append(
                {
                    "doc_id": row["doc_id"],
                    "content": row["content"],
                    "metadata": metadata,
                    "score": score,
                }
            )
        results.sort(key=lambda item: item["score"], reverse=True)
        return results[:top_k]

    async def search_by_text(
        self,
        query: str,
        top_k: int = 3,
        domain_filter: str | None = None,
    ) -> list[dict[str, Any]]:
        """先 get_embedding(query)，再 search_by_embedding"""
        query_embedding = await get_llm_client().get_embedding(query)
        return self.search_by_embedding(
            query_embedding,
            top_k=top_k,
            domain_filter=domain_filter,
        )

    def get_document(self, doc_id: str) -> dict[str, Any] | None:
        """SELECT 单条，返回 dict 或 None"""
        try:
            connection = self._connect()
            try:
                with connection:
                    row = connection.execute(
                        "SELECT doc_id, content, metadata FROM knowledge WHERE doc_id = ?",
                        (doc_id,),
                    ).fetchone()
            finally:
                connection.close()
        except Exception as exc:
            logger.warning("知识文档读取失败: %s", exc)
            return None
        if row is None:
            return None
        try:
            metadata = json.loads(row["metadata"])
        except Exception:
            metadata = {}
        return {"doc_id": row["doc_id"], "content": row["content"], "metadata": metadata}

    def list_all(self) -> list[dict[str, Any]]:
        """返回所有文档"""
        try:
            connection = self._connect()
            try:
                with connection:
                    rows = connection.execute(
                        "SELECT doc_id, content, metadata FROM knowledge ORDER BY doc_id"
                    ).fetchall()
            finally:
                connection.close()
        except Exception as exc:
            logger.warning("知识文档列表读取失败: %s", exc)
            return []
        results: list[dict[str, Any]] = []
        for row in rows:
            try:
                metadata = json.loads(row["metadata"])
            except Exception:
                metadata = {}
            results.append(
                {"doc_id": row["doc_id"], "content": row["content"], "metadata": metadata}
            )
        return results

    def delete_document(self, doc_id: str) -> bool:
        """删除文档，返回是否成功"""
        try:
            connection = self._connect()
            try:
                with connection:
                    cursor = connection.execute(
                        "DELETE FROM knowledge WHERE doc_id = ?",
                        (doc_id,),
                    )
                return cursor.rowcount > 0
            finally:
                connection.close()
        except Exception as exc:
            logger.warning("知识文档删除失败: %s", exc)
            return False


# 全局单例
_rag_store: RAGStore | None = None


def get_rag_store() -> RAGStore:
    global _rag_store
    if _rag_store is None:
        _rag_store = RAGStore()
    return _rag_store
