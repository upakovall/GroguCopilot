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


def test_continuous_voice_multiturn_interaction():
    """Verify continuous voice mode: multiple consecutive voice turns with clean STT buffer resets."""
    app = FastAPI(title="Continuous Voice App")
    registry = MCPRegistry()
    copilot_router = create_copilot_router(
        registry=registry,
        llm_backend="mock",
        endpoint_path="/ws/copilot"
    )
    app.include_router(copilot_router)
    client = TestClient(app)

    with client.websocket_connect("/ws/copilot") as ws:
        # 1. Handshake
        init_data = json.loads(ws.receive_text())
        assert init_data["type"] == "SESSION_INIT"

        # 2. ViewContext Update
        ctx = ViewContext(
            screen_id="checkout",
            title="Checkout Screen",
            components=[
                UIComponent(
                    id="submit_order_btn",
                    type="button",
                    label="Place Order",
                    allowed_actions=["click"]
                )
            ]
        )
        ws.send_text(json.dumps({
            "type": "VIEW_CONTEXT_UPDATE",
            "view_context": ctx.model_dump()
        }))

        # TURN 1: Stream binary PCM audio chunks, then AUDIO_END
        dummy_pcm_chunk_1 = b"\x00\x00" * 800  # 1600 bytes = 800 16-bit PCM samples
        dummy_pcm_chunk_2 = b"\x01\x00" * 800
        ws.send_bytes(dummy_pcm_chunk_1)
        ws.send_bytes(dummy_pcm_chunk_2)

        ws.send_text(json.dumps({"type": "AUDIO_END"}))

        # Collect Turn 1 responses until AUDIO_STREAM_END
        turn_1_types = []
        while True:
            msg = json.loads(ws.receive_text())
            turn_1_types.append(msg["type"])
            if msg["type"] == "AUDIO_STREAM_END":
                break

        assert "TRANSCRIPTION" in turn_1_types
        assert "AGENT_RESPONSE" in turn_1_types
        assert "AUDIO_RESPONSE" in turn_1_types
        assert "AUDIO_STREAM_END" in turn_1_types

        # TURN 2: Stream consecutive binary PCM audio turn without disconnecting
        dummy_pcm_chunk_3 = b"\x02\x00" * 800
        ws.send_bytes(dummy_pcm_chunk_3)
        ws.send_text(json.dumps({"type": "AUDIO_END"}))

        # Collect Turn 2 responses until AUDIO_STREAM_END
        turn_2_types = []
        while True:
            msg = json.loads(ws.receive_text())
            turn_2_types.append(msg["type"])
            if msg["type"] == "AUDIO_STREAM_END":
                break

        assert "TRANSCRIPTION" in turn_2_types
        assert "AGENT_RESPONSE" in turn_2_types
        assert "AUDIO_RESPONSE" in turn_2_types
        assert "AUDIO_STREAM_END" in turn_2_types

