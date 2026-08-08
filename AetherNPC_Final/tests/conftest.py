import pytest
from fastapi.testclient import TestClient
from app.main import app


@pytest.fixture
def client():
    return TestClient(app)


@pytest.fixture
def service():
    from app.services.dialogue_service import get_dialogue_service

    return get_dialogue_service()


@pytest.fixture(autouse=True)
def reset_singletons():
    """每个测试前重置单例状态"""
    import app.llm_client as llm_module
    import app.memory as memory_module
    import app.rag as rag_module
    import app.story_engine as story_module
    import app.services.dialogue_service as service_module
    import app.agent_tester as tester_module

    modules = (
        llm_module,
        memory_module,
        rag_module,
        story_module,
        service_module,
        tester_module,
    )
    attrs = (
        "_llm_client",
        "_memory_manager",
        "_rag_store",
        "_story_engine",
        "_dialogue_service",
        "_agent_tester",
    )
    for module in modules:
        for attr in attrs:
            if hasattr(module, attr):
                setattr(module, attr, None)
    yield
    client = llm_module._llm_client
    if client is not None and not client.use_mock:
        import asyncio

        asyncio.run(client.close())
    for module in modules:
        for attr in attrs:
            if hasattr(module, attr):
                setattr(module, attr, None)
