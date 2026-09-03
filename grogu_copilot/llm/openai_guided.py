"""OpenAI-compatible Guided Decoding Provider for RunPod / vLLM / llama.cpp / Ollama."""

import json
import logging
import re
from typing import Optional, Dict, Any, List
import httpx
from ..schemas.context import ViewContext
from ..schemas.actions import AgentResponse
from ..registry import MCPRegistry
from .provider import BaseLLMProvider

logger = logging.getLogger(__name__)


class OpenAIGuidedProvider(BaseLLMProvider):
    """Guided JSON Decoding provider for vLLM / llama.cpp / Ollama / RunPod endpoints."""

    def __init__(
        self,
        api_base: str = "http://localhost:8001/v1",
        api_key: Optional[str] = None,
        model_name: str = "qwen2.5:14b",
        temperature: float = 0.1,
        max_tokens: int = 512,
    ):
        self.api_base = api_base.rstrip("/")
        self.api_key = api_key
        self.model_name = model_name
        self.temperature = temperature
        self.max_tokens = max_tokens

    async def generate_response(
        self,
        prompt: str,
        context: ViewContext,
        registry: MCPRegistry,
        history: Optional[List[Dict[str, str]]] = None,
    ) -> AgentResponse:
        """Call remote RunPod / local vLLM / Ollama endpoint with JSON schema guided decoding."""
        schema = AgentResponse.model_json_schema()
        components_list = [comp.model_dump() for comp in context.components]

        system_prompt = f"""You are Grogu Voice AI Copilot, a high-precision, UI-aware intelligent assistant.
You receive the user's spoken command and the declarative ViewContext (semantic UI state).
You maintain conversational awareness across dialogue turns.
You must output ONLY a valid JSON object strictly conforming to the JSON Schema. Do NOT wrap output in markdown fences.

JSON Schema:
{json.dumps(schema, indent=2)}

Current Screen: {context.screen_id} ({context.title})
Active Modal: {context.active_modal or 'None'}

Available Interactive UI Components:
{json.dumps(components_list, indent=2)}

High-level Domain State:
{json.dumps(context.state_summary, indent=2)}
"""

        # Build message history
        messages = [{"role": "system", "content": system_prompt}]
        if history:
            for turn in history[-10:]:
                messages.append(turn)
        messages.append({"role": "user", "content": prompt})

        payload: Dict[str, Any] = {
            "model": self.model_name,
            "messages": messages,
            "temperature": self.temperature,
            "max_tokens": self.max_tokens,
            "response_format": {"type": "json_object"},
        }

        headers = {
            "Content-Type": "application/json",
        }
        if self.api_key:
            headers["Authorization"] = f"Bearer {self.api_key}"

        logger.info(f"[OpenAIGuidedProvider] Sending prompt with history ({len(messages)} messages) to LLM endpoint: {self.api_base}/chat/completions")

        async with httpx.AsyncClient(timeout=60.0) as client:
            resp = await client.post(
                f"{self.api_base}/chat/completions",
                json=payload,
                headers=headers
            )
            resp.raise_for_status()
            res_data = resp.json()
            raw_content = res_data["choices"][0]["message"]["content"].strip()
            
            # Clean markdown JSON code fences if model returned them
            if raw_content.startswith("```"):
                raw_content = re.sub(r"^```(?:json)?\s*", "", raw_content)
                raw_content = re.sub(r"\s*```$", "", raw_content)

            parsed = json.loads(raw_content)
            agent_response = AgentResponse.model_validate(parsed)
            return agent_response
