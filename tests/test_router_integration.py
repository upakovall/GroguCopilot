"""Integration tests for mounting decoupled router into a 3rd-party FastAPI app."""

import json
from fastapi import FastAPI
from starlette.testclient import TestClient
from grogu_copilot import create_copilot_router, MCPRegistry, ViewContext, UIComponent


def test_independent_fastapi_app_integration():
    # 1. Create independent 3rd-party FastAPI application
    third_party_app = FastAPI(title="My Independent Third Party App")
    
    # 2. Inject MCPRegistry
    registry = MCPRegistry()
    
    # 3. Mount Copilot Router
    copilot_router = create_copilot_router(
        registry=registry,
        llm_backend="mock",
        endpoint_path="/ws/copilot"
    )
    third_party_app.include_router(copilot_router)

    # 4. Connect via TestClient
    client = TestClient(third_party_app)
    
    # Test health
    health_resp = client.get("/copilot/health")
    assert health_resp.status_code == 200
    assert health_resp.json()["status"] == "healthy"

    # Test WebSocket session
    with client.websocket_connect("/ws/copilot") as ws:
        # Handshake
        init_raw = ws.receive_text()
        init_data = json.loads(init_raw)
        assert init_data["type"] == "SESSION_INIT"

        # Send ViewContext
        ctx = ViewContext(
            screen_id="user_profile",
            title="User Profile Screen",
            components=[
                UIComponent(
                    id="email_notifications_toggle",
                    type="switch",
                    label="Email Notifications",
                    value=True,
                    allowed_actions=["toggle"]
                )
            ]
        )
        ws.send_text(json.dumps({
            "type": "VIEW_CONTEXT_UPDATE",
            "view_context": ctx.model_dump()
        }))

        # Send text prompt
        ws.send_text(json.dumps({
            "type": "TEXT_PROMPT",
            "text": "Disable email notifications"
        }))

        # Collect responses
        messages = []
        for _ in range(5):
            msg = json.loads(ws.receive_text())
            messages.append(msg)
            if msg["type"] == "AUDIO_STREAM_END":
                break

        msg_types = [m["type"] for m in messages]
        assert "AGENT_RESPONSE" in msg_types
        assert "AUDIO_RESPONSE" in msg_types

        agent_msg = next(m for m in messages if m["type"] == "AGENT_RESPONSE")
        actions = agent_msg["agent_response"]["actions"]
        assert len(actions) == 1
        assert actions[0]["target_id"] == "email_notifications_toggle"
        assert actions[0]["payload"]["state"] is False
