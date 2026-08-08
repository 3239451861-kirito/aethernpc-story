"""FastAPI 接口测试：HTTP、WebSocket、静态资源与 Agent 遍历测试。"""

from __future__ import annotations

from fastapi.testclient import TestClient


def _new_session(client: TestClient) -> str:
    response = client.post("/sessions", params={"player_id": "player"})
    assert response.status_code == 200
    return response.json()["session_id"]


def test_health(client: TestClient) -> None:
    response = client.get("/health")
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "ok"
    assert data["mock_mode"] is True
    assert data["version"] == "2.0.0"


def test_create_session(client: TestClient) -> None:
    response = client.post("/sessions", params={"player_id": "alice"})
    assert response.status_code == 200
    data = response.json()
    assert data["player_id"] == "alice"
    assert data["session_id"]
    default = client.post("/sessions").json()
    assert default["player_id"] == "player-001"
    assert default["session_id"]


def test_dialogue_endpoint(client: TestClient) -> None:
    session_id = _new_session(client)
    response = client.post(
        "/dialogue",
        json={
            "session_id": session_id,
            "npc_id": "gate_guard",
            "message": "谢谢",
        },
    )
    assert response.status_code == 200
    data = response.json()
    assert data["response"]["dialogue"]
    assert data["current_node"] == "city_gate"
    assert data["relationship"] == 0


def test_dialogue_missing_session_404(client: TestClient) -> None:
    response = client.post(
        "/dialogue",
        json={"session_id": "nope", "npc_id": "gate_guard", "message": "你好"},
    )
    assert response.status_code == 404
    assert response.json()["detail"] == "会话不存在或已过期"


def test_dialogue_unknown_npc_404(client: TestClient) -> None:
    session_id = _new_session(client)
    response = client.post(
        "/dialogue",
        json={"session_id": session_id, "npc_id": "ghost", "message": "你好"},
    )
    assert response.status_code == 404


def test_dialogue_invalid_input_422(client: TestClient) -> None:
    session_id = _new_session(client)
    response = client.post(
        "/dialogue",
        json={"session_id": session_id, "npc_id": "gate_guard", "message": "   "},
    )
    assert response.status_code == 422


def test_dialogue_choice_flow(client: TestClient) -> None:
    session_id = _new_session(client)
    response = client.post(
        "/dialogue",
        json={
            "session_id": session_id,
            "npc_id": "gate_guard",
            "message": "谢谢",
            "choice_id": "ask_guard",
        },
    )
    assert response.status_code == 200
    data = response.json()
    assert data["validated_choice"] == "ask_guard"
    assert data["current_node"] == "guard_dialogue"
    state = client.get("/sessions/" + session_id).json()
    assert state["current_node"] == "guard_dialogue"
    assert state["relationship_values"]["gate_guard"] == 2


def test_get_session_and_404(client: TestClient) -> None:
    session_id = _new_session(client)
    state = client.get("/sessions/" + session_id).json()
    assert state["current_node"] == "city_gate"
    assert state["history_count"] == 0
    assert "covered_nodes" in state
    assert client.get("/sessions/not-exist").status_code == 404


def test_list_npcs(client: TestClient) -> None:
    response = client.get("/npcs")
    assert response.status_code == 200
    data = response.json()
    assert {npc["npc_id"] for npc in data["npcs"]} == {
        "gate_guard",
        "tavern_owner",
        "mysterious_stranger",
        "mayor",
        "librarian",
        "ranger",
        "priestess",
        "merchant",
        "alchemist",
        "captain",
        "beggar",
        "cult_informant",
        "miner",
    }
    assert len(data["npcs"]) == 13


def test_story_graph(client: TestClient) -> None:
    response = client.get("/story/graph")
    assert response.status_code == 200
    data = response.json()
    assert data["start_node"] == "city_gate"
    assert len(data["nodes"]) == 36
    assert "city_gate" in data["nodes"]


def test_run_tests(client: TestClient) -> None:
    response = client.post("/tests/run", json={})
    assert response.status_code == 200
    report = response.json()
    assert report["total_nodes"] == 36
    assert report["total_edges"] == 76
    assert 29 <= report["covered_nodes"] <= 34
    assert 61 <= report["covered_edges"] <= 67
    assert report["edge_coverage"] >= 0.8
    assert {"guard_bribed", "guard_secret"} <= set(report["unreachable_nodes"])
    assert report["reached_target"] is True
    assert isinstance(report["invalid_jumps"], list)
    for item in report["invalid_jumps"]:
        assert {"from_node", "choice_id", "error", "step"} <= set(item)
    assert any(item.startswith("BACKTRACK->") for item in report["path_trace"])


def test_run_tests_max_steps(client: TestClient) -> None:
    report = client.post("/tests/run", json={"max_steps": 1}).json()
    assert report["steps_taken"] == 1
    assert report["reached_target"] is False


def test_websocket_dialogue(client: TestClient) -> None:
    session_id = _new_session(client)
    with client.websocket_connect("/ws/" + session_id) as websocket:
        websocket.send_json({"npc_id": "gate_guard", "message": "谢谢"})
        data = websocket.receive_json()
        assert data["response"]["dialogue"]
        assert data["current_node"] == "city_gate"
        websocket.send_json(
            {"npc_id": "gate_guard", "message": "谢谢", "choice_id": "ask_guard"}
        )
        data = websocket.receive_json()
        assert data["validated_choice"] == "ask_guard"
        websocket.send_json({"action": "close"})


def test_websocket_missing_session_error(client: TestClient) -> None:
    with client.websocket_connect("/ws/not-exist") as websocket:
        websocket.send_json({"npc_id": "gate_guard", "message": "你好"})
        data = websocket.receive_json()
        assert data["type"] == "error"


def test_websocket_invalid_message_returns_error(client: TestClient) -> None:
    with client.websocket_connect("/ws/test-session") as websocket:
        websocket.send_text("{不是合法JSON")
        data = websocket.receive_json()
        assert data["type"] == "error"


def test_static_demo_served(client: TestClient) -> None:
    response = client.get("/static/demo.html")
    assert response.status_code == 200
    assert "AetherNPC" in response.text
