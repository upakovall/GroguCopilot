/**
 * Audio Resampler & 16-bit PCM Converter.
 * 
 * Performs high-precision linear interpolation downsampling from browser-native
 * sample rates (44.1kHz / 48kHz) to strict 16,000Hz, and converts Float32 to Int16Array.
 */

export class AudioResampler {
  /**
   * @param {number} inputSampleRate Source sample rate (e.g. 44100 or 48000)
   * @param {number} targetSampleRate Output sample rate (default: 16000)
   */
  constructor(inputSampleRate, targetSampleRate = 16000) {
    this.inputSampleRate = inputSampleRate;
    this.targetSampleRate = targetSampleRate;
    this.ratio = inputSampleRate / targetSampleRate;
  }

  /**
   * Downsamples a Float32Array to target sample rate using linear interpolation.
   * @param {Float32Array} inputBuffer Source audio buffer (Float32 in [-1.0, 1.0])
   * @returns {Float32Array} Downsampled audio buffer at 16,000Hz
   */
  downsample(inputBuffer) {
    if (this.inputSampleRate === this.targetSampleRate) {
      return inputBuffer;
    }

    const outputLength = Math.round(inputBuffer.length / this.ratio);
    const outputBuffer = new Float32Array(outputLength);

    for (let i = 0; i < outputLength; i++) {
      const position = i * this.ratio;
      const index = Math.floor(position);
      const weight = position - index;

      const current = inputBuffer[index] || 0;
      const next = inputBuffer[index + 1] !== undefined ? inputBuffer[index + 1] : current;

      // Linear interpolation
      outputBuffer[i] = current + weight * (next - current);
    }

    return outputBuffer;
  }

  /**
   * Converts Float32 audio samples [-1.0, 1.0] to signed 16-bit Little-Endian PCM (Int16Array).
   * @param {Float32Array} floatBuffer Downsampled audio buffer
   * @returns {Int16Array} 16-bit PCM samples in range [-32768, 32767]
   */
  floatTo16BitPCM(floatBuffer) {
    const pcm = new Int16Array(floatBuffer.length);
    for (let i = 0; i < floatBuffer.length; i++) {
      const s = Math.max(-1.0, Math.min(1.0, floatBuffer[i]));
      pcm[i] = s < 0 ? s * 0x8000 : s * 0x7FFF;
    }
    return pcm;
  }

  /**
   * Full pipeline: downsample Float32 and return 16-bit PCM Int16Array.
   * @param {Float32Array} inputBuffer Raw audio from Web Audio API
   * @returns {Int16Array} 16kHz 16-bit PCM buffer
   */
  process(inputBuffer) {
    const downsampled = this.downsample(inputBuffer);
    return this.floatTo16BitPCM(downsampled);
  }
}
