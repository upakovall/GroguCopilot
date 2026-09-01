"""Unit tests for audio pipeline and RIFF/WAV header formatting."""

import struct
from grogu_copilot.audio.resampler import (
    build_wav_header,
    wrap_pcm_to_wav,
    validate_pcm_16,
)


def test_build_wav_header_exact_length():
    """Verify that build_wav_header generates an exact 44-byte standard header."""
    header = build_wav_header(data_size=32000, sample_rate=16000, num_channels=1, bits_per_sample=16)
    assert len(header) == 44

    # Verify RIFF chunk descriptor
    assert header[0:4] == b"RIFF"
    total_size = struct.unpack("<I", header[4:8])[0]
    assert total_size == 36 + 32000
    assert header[8:12] == b"WAVE"

    # Verify 'fmt ' subchunk
    assert header[12:16] == b"fmt "
    fmt_size = struct.unpack("<I", header[16:20])[0]
    assert fmt_size == 16
    audio_fmt = struct.unpack("<H", header[20:22])[0]
    assert audio_fmt == 1  # Linear PCM

    channels = struct.unpack("<H", header[22:24])[0]
    assert channels == 1

    sample_rate = struct.unpack("<I", header[24:28])[0]
    assert sample_rate == 16000

    byte_rate = struct.unpack("<I", header[28:32])[0]
    assert byte_rate == 32000  # 16000 * 1 * 2

    block_align = struct.unpack("<H", header[32:34])[0]
    assert block_align == 2

    bits_per_sample = struct.unpack("<H", header[34:36])[0]
    assert bits_per_sample == 16

    # Verify 'data' subchunk
    assert header[36:40] == b"data"
    data_size = struct.unpack("<I", header[40:44])[0]
    assert data_size == 32000


def test_wrap_pcm_to_wav():
    fake_pcm = b"\x00\x00" * 8000  # 8000 samples = 16000 bytes
    wav_bytes = wrap_pcm_to_wav(fake_pcm, sample_rate=16000)
    assert len(wav_bytes) == 44 + 16000
    assert wav_bytes[:4] == b"RIFF"
    assert wav_bytes[44:] == fake_pcm


def test_validate_pcm_16():
    valid_pcm = b"\x01\x02\x03\x04"
    is_valid, _ = validate_pcm_16(valid_pcm)
    assert is_valid is True

    odd_pcm = b"\x01\x02\x03"
    is_invalid, err = validate_pcm_16(odd_pcm)
    assert is_invalid is False
    assert "not aligned" in err
