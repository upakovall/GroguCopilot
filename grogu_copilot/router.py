"""FastAPI APIRouter Factory for Voice AI Copilot.

Enables any third-party FastAPI application to mount the Voice AI Copilot with
custom MCP registries, backends, and endpoints.
"""

import json
import base64
import logging
from typing import Optional
from fastapi import APIRouter, WebSocket, WebSocketDisconnect
from pydantic import ValidationError

from .schemas.context import ViewContext
from .schemas.actions import AgentResponse
from .schemas.messages import (
    ClientMessageType,
    ServerMessageType,
    InboundWSMessage,
    OutboundWSMessage,
)
from .registry import MCPRegistry
from .engine import CopilotEngine
from .audio.stt import STTService
from .audio.tts import TTSService
from .llm.dynamic_reasoner import DynamicReasoner
from .llm.openai_guided import OpenAIGuidedProvider

logger = logging.getLogger(__name__)


def create_copilot_router(
    registry: Optional[MCPRegistry] = None,
    llm_backend: str = "mock",
    llm_api_base: Optional[str] = "http://localhost:8001/v1",
    llm_api_key: Optional[str] = None,
    model_name: str = "Qwen/Qwen2.5-7B-Instruct-AWQ",
    endpoint_path: str = "/ws/copilot",
    stt_model_size: str = "base",
    stt_device: str = "cpu",
    preload_stt: bool = True,
) -> APIRouter:
    """Factory creating a decoupled FastAPI APIRouter for the Voice AI Copilot.
    
    Args:
        registry: Injected MCPRegistry instance for Inversion of Control.
        llm_backend: "mock", "dynamic", "vllm", "llama_cpp", or "openai_compatible".
        llm_api_base: URL base for RunPod / local OpenAI-compatible endpoints.
        llm_api_key: Optional API key / Bearer token for RunPod or cloud proxies.
        model_name: Model identifier for guided LLM completion.
        endpoint_path: WebSocket route path (default: "/ws/copilot").
        stt_model_size: faster-whisper model size (default: "base").
        stt_device: Execution device for STT (default: "cpu" for 0 VRAM).
        preload_stt: Eagerly pre-warm and download faster-whisper model on CPU (default: True).
        
    Returns:
        FastAPI APIRouter ready to be mounted via app.include_router().
    """
    router = APIRouter()
    shared_registry = registry or MCPRegistry()

    # Configure LLM provider
    if llm_backend in ["vllm", "llama_cpp", "openai_compatible", "runpod", "ollama"]:
        llm_provider = OpenAIGuidedProvider(
            api_base=llm_api_base or "http://localhost:8001/v1",
            api_key=llm_api_key,
            model_name=model_name,
        )
    else:
        llm_provider = DynamicReasoner()

    stt_service = STTService(
        model_size=stt_model_size,
        device=stt_device,
        use_mock=(llm_backend == "mock"),
    )
    tts_service = TTSService(sample_rate=16000)

    # Pre-fetch faster-whisper model weights in background on startup (e.g. RunPod)
    if preload_stt and llm_backend != "mock":
        import threading
        threading.Thread(
            target=stt_service.initialize,
            daemon=True,
            name="copilot-stt-preloader"
        ).start()

    @router.websocket(endpoint_path)
    async def copilot_ws_endpoint(websocket: WebSocket):
        await websocket.accept()
        engine = CopilotEngine(
            registry=shared_registry,
            llm_provider=llm_provider,
            stt_service=stt_service,
            tts_service=tts_service,
        )
        logger.info(f"[CopilotRouter] Client connected to {endpoint_path}")

        # Send session initialization handshake
        init_msg = OutboundWSMessage(
            type=ServerMessageType.SESSION_INIT,
            data={
                "status": "ready",
                "sample_rate": 16000,
                "audio_format": "pcm_s16le",
                "backend": llm_backend,
            }
        )
        await websocket.send_text(init_msg.model_dump_json())

        try:
            while True:
                ws_msg = await websocket.receive()
                if ws_msg.get("type") == "websocket.disconnect":
                    break

                # Binary 16kHz PCM audio stream chunk
                if "bytes" in ws_msg and ws_msg["bytes"]:
                    engine.append_audio_chunk(ws_msg["bytes"])
                    continue

                # JSON text message envelope
                if "text" in ws_msg and ws_msg["text"]:
                    try:
                        raw = json.loads(ws_msg["text"])
                        inbound = InboundWSMessage.model_validate(raw)
                    except (json.JSONDecodeError, ValidationError) as err:
                        err_resp = OutboundWSMessage(
                            type=ServerMessageType.ERROR,
                            error=f"Invalid message format: {str(err)}"
                        )
                        await websocket.send_text(err_resp.model_dump_json())
                        continue

                    if inbound.type == ClientMessageType.PING:
                        await websocket.send_text(OutboundWSMessage(type=ServerMessageType.PONG).model_dump_json())

                    elif inbound.type == ClientMessageType.VIEW_CONTEXT_UPDATE:
                        if inbound.view_context:
                            engine.update_view_context(inbound.view_context)
                        elif inbound.data:
                            try:
                                ctx = ViewContext.model_validate(inbound.data)
                                engine.update_view_context(ctx)
                            except ValidationError as ve:
                                logger.error(f"[CopilotRouter] ViewContext error: {ve}")

                    elif inbound.type == ClientMessageType.AUDIO_CHUNK:
                        if inbound.audio_base64:
                            engine.append_audio_chunk(base64.b64decode(inbound.audio_base64))

                    elif inbound.type == ClientMessageType.AUDIO_END:
                        async for out_msg in engine.process_voice_stream():
                            await websocket.send_text(out_msg.model_dump_json())

                    elif inbound.type == ClientMessageType.TEXT_PROMPT:
                        if inbound.text:
                            async for out_msg in engine.process_text_prompt(inbound.text):
                                await websocket.send_text(out_msg.model_dump_json())

                    elif inbound.type == ClientMessageType.CLEAR_HISTORY if hasattr(ClientMessageType, 'CLEAR_HISTORY') else False:
                        engine.clear_history()
                    elif inbound.type == ClientMessageType.ACTION_ACK:
                        logger.debug(f"[CopilotRouter] Action ACK: {inbound.data}")

        except (WebSocketDisconnect, RuntimeError):
            logger.info(f"[CopilotRouter] Client disconnected from {endpoint_path}")
        except Exception as e:
            logger.exception(f"[CopilotRouter] WebSocket exception: {e}")

    @router.get("/copilot/health")
    async def health_endpoint():
        return {
            "status": "healthy",
            "backend": llm_backend,
            "sample_rate": 16000,
            "vram_safe_mode": True,
        }

    @router.get("/copilot/mcp/tools")
    async def mcp_tools_endpoint():
        return {"tools": shared_registry.get_tool_definitions()}

    @router.get("/copilot/schema/agent-response")
    async def schema_endpoint():
        return AgentResponse.model_json_schema()

    return router
