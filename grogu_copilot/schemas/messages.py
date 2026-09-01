"""WebSocket Envelope Schemas for Bi-directional Streaming."""

from enum import Enum
from typing import Any, Dict, List, Optional
from pydantic import BaseModel
from .context import ViewContext
from .actions import AgentResponse, UIAction


class ClientMessageType(str, Enum):
    VIEW_CONTEXT_UPDATE = "VIEW_CONTEXT_UPDATE"
    AUDIO_CHUNK = "AUDIO_CHUNK"
    AUDIO_END = "AUDIO_END"
    TEXT_PROMPT = "TEXT_PROMPT"
    ACTION_ACK = "ACTION_ACK"
    PING = "PING"


class ServerMessageType(str, Enum):
    SESSION_INIT = "SESSION_INIT"
    TRANSCRIPTION = "TRANSCRIPTION"
    AGENT_THINKING = "AGENT_THINKING"
    AGENT_RESPONSE = "AGENT_RESPONSE"
    AUDIO_RESPONSE = "AUDIO_RESPONSE"
    AUDIO_STREAM_END = "AUDIO_STREAM_END"
    ERROR = "ERROR"
    PONG = "PONG"


class InboundWSMessage(BaseModel):
    """Inbound message envelope sent from Frontend to Backend."""
    type: ClientMessageType
    data: Optional[Dict[str, Any]] = None
    audio_base64: Optional[str] = None
    text: Optional[str] = None
    view_context: Optional[ViewContext] = None
    client_timestamp: Optional[float] = None


class OutboundWSMessage(BaseModel):
    """Outbound message envelope sent from Backend to Frontend."""
    type: ServerMessageType
    text: Optional[str] = None
    is_final: Optional[bool] = None
    data: Optional[Dict[str, Any]] = None
    agent_response: Optional[AgentResponse] = None
    audio_base64: Optional[str] = None
    error: Optional[str] = None
