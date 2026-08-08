"""RAG 知识库测试：SQLite 持久化 + 余弦相似度向量检索。"""

from __future__ import annotations

import asyncio
import math

import pytest

import app.config as config
from app.llm_client import get_llm_client
from app.rag import RAGStore, cosine_similarity, get_rag_store

SAMPLE_KNOWLEDGE = [
    {
        "doc_id": "knowledge_001",
        "content": "银月城是大陆上最古老的城市之一，建于第一纪元。",
        "metadata": {"category": "地理", "region": "银月城"},
    },
    {
        "doc_id": "knowledge_002",
        "content": "影子教团是一个崇拜虚空之主的秘密组织。",
        "metadata": {"category": "组织", "danger_level": "极高"},
    },
    {
        "doc_id": "knowledge_003",
        "content": "虚空之钥是一件上古神器，据说是封印虚空之主的七件圣物之一。",
        "metadata": {"category": "神器", "era": "上古时代"},
    },
    {
        "doc_id": "knowledge_004",
        "content": "最后一批星光商队在三天前失踪于黑森林。",
        "metadata": {"category": "事件", "status": "调查中"},
    },
    {
        "doc_id": "knowledge_005",
        "content": "黑森林位于银月城以东，常年被黑色雾气笼罩。",
        "metadata": {"category": "地理", "danger_level": "高"},
    },
    {
        "doc_id": "knowledge_006",
        "content": "月长石是一种稀有的魔法矿石，能够吸收月光并释放净化之力。",
        "metadata": {"category": "矿物", "magic": True},
    },
]


@pytest.fixture
def rag(tmp_path, monkeypatch) -> RAGStore:
    monkeypatch.setattr(config, "DB_PATH", tmp_path / "test_rag.db")
    store = RAGStore()
    for doc in SAMPLE_KNOWLEDGE:
        asyncio.run(store.add_document(doc["doc_id"], doc["content"], doc["metadata"]))
    return store


def test_cosine_similarity_identical() -> None:
    assert cosine_similarity([1.0, 0.0], [1.0, 0.0]) == pytest.approx(1.0)


def test_cosine_similarity_orthogonal() -> None:
    assert cosine_similarity([1.0, 0.0], [0.0, 1.0]) == pytest.approx(0.0)


def test_cosine_similarity_empty() -> None:
    assert cosine_similarity([], [1.0, 0.0]) == 0.0
    assert cosine_similarity([0.0, 0.0], [1.0, 0.0]) == 0.0


def test_mock_embedding_deterministic() -> None:
    client = get_llm_client()
    first = asyncio.run(client.get_embedding("黑森林在哪里"))
    second = asyncio.run(client.get_embedding("黑森林在哪里"))
    assert first == second
    assert len(first) == config.VECTOR_DIM
    norm = math.sqrt(sum(value * value for value in first))
    assert norm == pytest.approx(1.0)


def test_add_and_search(rag: RAGStore) -> None:
    results = asyncio.run(rag.search_by_text("黑森林在哪里", top_k=3))
    assert len(results) == 3
    for item in results:
        assert set(item) == {"doc_id", "content", "metadata", "score"}
    assert {item["doc_id"] for item in results} <= {
        doc["doc_id"] for doc in SAMPLE_KNOWLEDGE
    }


def test_search_domain_filter(rag: RAGStore) -> None:
    results = asyncio.run(rag.search_by_text("黑森林", top_k=5, domain_filter="地理"))
    assert results
    assert all(item["metadata"]["category"] == "地理" for item in results)


def test_get_document(rag: RAGStore) -> None:
    doc = rag.get_document("knowledge_001")
    assert doc is not None
    assert doc["doc_id"] == "knowledge_001"
    assert "银月城" in doc["content"]
    assert rag.get_document("missing") is None


def test_delete_document(rag: RAGStore) -> None:
    assert rag.delete_document("knowledge_006") is True
    assert rag.get_document("knowledge_006") is None
    assert rag.delete_document("knowledge_006") is False


def test_list_all_and_persistence(rag: RAGStore) -> None:
    ids = {item["doc_id"] for item in rag.list_all()}
    assert ids == {doc["doc_id"] for doc in SAMPLE_KNOWLEDGE}
    second = RAGStore()
    assert second.get_document("knowledge_001") is not None


def test_singleton(tmp_path, monkeypatch) -> None:
    import app.rag as rag_module

    monkeypatch.setattr(config, "DB_PATH", tmp_path / "singleton.db")
    rag_module._rag_store = None
    first = get_rag_store()
    second = get_rag_store()
    assert first is second
    assert first.db_path == tmp_path / "singleton.db"
