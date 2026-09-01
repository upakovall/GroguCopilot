"""Decoupled Grogu Voice AI Copilot Session Engine."""

import base64
import logging
from typing import Optional, AsyncGenerator, List, Dict
from .schemas.context import ViewContext
from .schemas.actions import AgentResponse
from .schemas.messages import OutboundWSMessage, ServerMessageType
from .registry import MCPRegistry
from .llm.provider import BaseLLMProvider
from .llm.dynamic_reasoner import DynamicReasoner
from .audio.stt import STTService
from .audio.tts import TTSService

logger = logging.getLogger(__name__)


class CopilotEngine:
    """Decoupled Voice AI Copilot Session Engine with Multi-Turn Conversation Memory."""

    def __init__(
        self,
        registry: Optional[MCPRegistry] = None,
        llm_provider: Optional[BaseLLMProvider] = None,
        stt_service: Optional[STTService] = None,
        tts_service: Optional[TTSService] = None,
    ):
        self.registry = registry or MCPRegistry()
        self.llm = llm_provider or DynamicReasoner()
        self.stt = stt_service or STTService()
        self.tts = tts_service or TTSService()
        self._audio_buffer = bytearray()
        self.conversation_history: List[Dict[str, str]] = []

    def update_view_context(self, context: ViewContext) -> None:
        """Update active ViewContext in registry."""
        self.registry.update_context(context)

    def append_audio_chunk(self, chunk: bytes) -> None:
        """Accumulate incoming 16kHz PCM audio chunk."""
        self._audio_buffer.extend(chunk)

    def clear_audio_buffer(self) -> None:
        """Reset audio accumulation buffer."""
        self._audio_buffer.clear()

    def clear_history(self) -> None:
        """Reset multi-turn conversation memory."""
        self.conversation_history.clear()

    async def process_voice_stream(self) -> AsyncGenerator[OutboundWSMessage, None]:
        """Transcribe accumulated 16kHz PCM stream, execute reasoning, and yield responses."""
        audio_data = bytes(self._audio_buffer)
        self.clear_audio_buffer()

        if not audio_data:
            yield OutboundWSMessage(
                type=ServerMessageType.ERROR,
                error="No audio stream data received"
            )
            return

        # 1. Transcribe speech using faster-whisper on CPU
        yield OutboundWSMessage(
            type=ServerMessageType.AGENT_THINKING,
            text="Transcribing voice input..."
        )

        transcript = self.stt.transcribe(audio_data)
        if not transcript:
            transcript = "Status check"

        yield OutboundWSMessage(
            type=ServerMessageType.TRANSCRIPTION,
            text=transcript,
            is_final=True
        )

        # 2. Process text prompt against active ViewContext
        async for msg in self.process_text_prompt(transcript):
            yield msg

    async def process_text_prompt(self, prompt: str) -> AsyncGenerator[OutboundWSMessage, None]:
        """Process natural language command against ViewContext with multi-turn memory."""
        context = self.registry.get_context()
        if not context:
            yield OutboundWSMessage(
                type=ServerMessageType.ERROR,
                error="No active ViewContext set. Frontend must send ViewContext before dispatching commands."
            )
            return

        yield OutboundWSMessage(
            type=ServerMessageType.AGENT_THINKING,
            text=f"Reasoning over ViewContext '{context.screen_id}'..."
        )

        # 3. Dynamic LLM structured reasoning with dialogue history
        agent_response: AgentResponse = await self.llm.generate_response(
            prompt=prompt,
            context=context,
            registry=self.registry,
            history=self.conversation_history,
        )

        # 4. Save dialogue turn to history
        self.conversation_history.append({"role": "user", "content": prompt})
        self.conversation_history.append({
            "role": "assistant",
            "content": f"{agent_response.speech_output} (Thought: {agent_response.thought})"
        })

        # Emit structured response immediately
        yield OutboundWSMessage(
            type=ServerMessageType.AGENT_RESPONSE,
            agent_response=agent_response
        )

        # 5. Synthesize spoken response via TTS (emitting 44-byte RIFF/WAV header)
        if agent_response.speech_output:
            yield OutboundWSMessage(
                type=ServerMessageType.AGENT_THINKING,
                text="Synthesizing speech response..."
            )

            wav_bytes = await self.tts.synthesize(agent_response.speech_output)
            audio_base64 = base64.b64encode(wav_bytes).decode("utf-8")

            yield OutboundWSMessage(
                type=ServerMessageType.AUDIO_RESPONSE,
                audio_base64=audio_base64,
                text=agent_response.speech_output
            )

        yield OutboundWSMessage(type=ServerMessageType.AUDIO_STREAM_END)
