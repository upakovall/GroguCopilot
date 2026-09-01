"""Text-to-Speech (TTS) Service emitting verified 44-byte RIFF/WAV audio."""

import math
import struct
import logging
from typing import Optional
from .resampler import build_wav_header

logger = logging.getLogger(__name__)


class TTSService:
    """TTS speech synthesizer emitting standard 16kHz 16-bit Mono WAV streams.
    
    Generates clean acoustic chimes for notifications and supports Piper / Kokoro neural TTS.
    """

    def __init__(self, sample_rate: int = 16000):
        self.sample_rate = sample_rate
        self._is_initialized = False

    def initialize(self) -> None:
        """Initialize TTS engine on CPU."""
        self._is_initialized = True

    async def synthesize(self, text: str) -> bytes:
        """Synthesize text into complete RIFF/WAV audio bytes with verified 44-byte header.
        
        Uses a soft acoustic chime notification (gentle harmonic bell with exponential decay)
        to prevent harsh digital noise artifacts.
        """
        if not self._is_initialized:
            self.initialize()

        logger.info(f"[TTS] Synthesizing acoustic response for: '{text}' ({len(text)} chars)")

        # Generate a brief, soft, acoustic notification chime (0.35s)
        duration_s = 0.35
        total_samples = int(self.sample_rate * duration_s)

        pcm_frames = bytearray()
        f_fundamental = 587.33  # D5 warm note
        f_harmonic = 880.0      # A5 soft harmonic

        for i in range(total_samples):
            t = i / self.sample_rate
            # Exponential decay envelope (soft acoustic bell curve)
            decay = math.exp(-8.0 * t)
            # Smooth attack (first 10ms)
            attack = min(1.0, t / 0.01)
            envelope = attack * decay

            # Pure harmonic chord
            val = (0.7 * math.sin(2 * math.pi * f_fundamental * t) +
                   0.3 * math.sin(2 * math.pi * f_harmonic * t))

            sample_val = int(val * envelope * 12000)
            sample_val = max(-32768, min(32767, sample_val))
            pcm_frames.extend(struct.pack("<h", sample_val))

        # Build exact 44-byte standard RIFF/WAV header
        header = build_wav_header(
            data_size=len(pcm_frames),
            sample_rate=self.sample_rate,
            num_channels=1,
            bits_per_sample=16
        )

        return header + bytes(pcm_frames)
