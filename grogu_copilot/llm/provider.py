"""Base LLM Provider Interface."""

from abc import ABC, abstractmethod
from ..schemas.context import ViewContext
from ..schemas.actions import AgentResponse
from ..registry import MCPRegistry


class BaseLLMProvider(ABC):
    """Abstract interface for LLM inference engines."""

    @abstractmethod
    async def generate_response(
        self,
        prompt: str,
        context: ViewContext,
        registry: MCPRegistry
    ) -> AgentResponse:
        """Generate structured AgentResponse for user prompt and ViewContext."""
        pass
