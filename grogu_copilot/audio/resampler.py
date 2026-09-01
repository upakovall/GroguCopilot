"""Audio Resampling & Format Normalization Utilities.

Constructs strict 44-byte RIFF/WAV headers and verifies 16kHz 16-bit PCM binary payloads.
"""

import struct
from typing import Tuple


def build_wav_header(
    data_size: int,
    sample_rate: int = 16000,
    num_channels: int = 1,
    bits_per_sample: int = 16
) -> bytes:
    """Constructs a strict 44-byte standard RIFF/WAV header for PCM audio.
    
    Args:
        data_size: Size of raw PCM audio data in bytes.
        sample_rate: Audio sampling frequency in Hz (default: 16000).
        num_channels: 1 for Mono, 2 for Stereo (default: 1).
        bits_per_sample: Bit depth (default: 16).
        
    Returns:
        Exact 44-byte RIFF/WAV binary header.
    """
    byte_rate = sample_rate * num_channels * (bits_per_sample // 8)
    block_align = num_channels * (bits_per_sample // 8)
    total_file_size = 36 + data_size

    header = struct.pack(
        "<4sI4s4sIHHIIHH4sI",
        b"RIFF",
        total_file_size,
        b"WAVE",
        b"fmt ",
        16,              # Subchunk1Size for PCM
        1,               # AudioFormat: 1 = Linear PCM
        num_channels,
        sample_rate,
        byte_rate,
        block_align,
        bits_per_sample,
        b"data",
        data_size
    )
    return header


def wrap_pcm_to_wav(
    pcm_bytes: bytes,
    sample_rate: int = 16000,
    num_channels: int = 1,
    bits_per_sample: int = 16
) -> bytes:
    """Wraps raw 16-bit PCM bytes with a standard 44-byte WAV header."""
    header = build_wav_header(
        data_size=len(pcm_bytes),
        sample_rate=sample_rate,
        num_channels=num_channels,
        bits_per_sample=bits_per_sample
    )
    return header + pcm_bytes


def validate_pcm_16(pcm_bytes: bytes) -> Tuple[bool, str]:
    """Validates that bytes represent aligned 16-bit PCM (even byte length)."""
    if len(pcm_bytes) % 2 != 0:
        return False, "PCM data length is not aligned to 2 bytes (16-bit)"
    return True, "Valid 16-bit PCM"
