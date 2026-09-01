"""Grogu Voice AI Copilot - Decoupled Core Module.

Exports:
- create_copilot_router: FastAPI APIRouter factory
- MCPRegistry: Inversion of Control tool and action registry
- CopilotEngine: Session orchestrator
- ViewContext, UIComponent: Declarative semantic UI state models
- UIAction, ActionType, AgentResponse: Structured action models
"""

from .router import create_copilot_router
from .registry import MCPRegistry
from .engine import CopilotEngine
from .schemas import (
    ViewContext,
    UIComponent,
    UIAction,
    ActionType,
    AgentResponse,
    ClientMessageType,
    ServerMessageType,
    InboundWSMessage,
    OutboundWSMessage,
)

__version__ = "1.0.0"

__all__ = [
    "create_copilot_router",
    "MCPRegistry",
    "CopilotEngine",
    "ViewContext",
    "UIComponent",
    "UIAction",
    "ActionType",
    "AgentResponse",
    "ClientMessageType",
    "ServerMessageType",
    "InboundWSMessage",
    "OutboundWSMessage",
]
