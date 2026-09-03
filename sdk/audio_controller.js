/**
 * Client-Side Audio Controller with Voice Activity Detection (VAD).
 *
 * Implements full Web Audio API management (AudioContext, AnalyserNode, ScriptProcessorNode),
 * real-time Root-Mean-Square (RMS) energy analysis, a circular pre-speech buffer to catch
 * the first syllable, and an event-driven VAD state machine for continuous voice conversation.
 */

import { AudioResampler } from './audio_resampler.js';

/**
 * State machine enum representing the current conversational voice state.
 * @readonly
 * @enum {string}
 */
export const CopilotVoiceState = Object.freeze({
  IDLE: 'IDLE',                         // Mic is inactive/stopped
  LISTENING_SILENT: 'LISTENING_SILENT', // Mic active, analyzing volume, populating pre-speech buffer (no WS stream)
  LISTENING_SPEAKING: 'LISTENING_SPEAKING', // Speech detected, streaming 16kHz PCM chunks over WebSocket
  PROCESSING: 'PROCESSING',             // Speech paused > silenceTimeout; waiting for STT/LLM response
  SPEAKING: 'SPEAKING'                  // AI is speaking response (TTS); mic input ignored to prevent echo
});

export class AudioController {
  /**
   * @param {Object} [options={}] Configuration options for Audio Controller
   * @param {number} [options.silenceThreshold=0.015] RMS volume threshold above which speech is detected
   * @param {number} [options.silenceTimeoutMs=1500] Milliseconds of silence to trigger end-of-speech
   * @param {number} [options.preSpeechBufferMs=400] Milliseconds of pre-speech rolling audio buffer
   * @param {number} [options.targetSampleRate=16000] Target downsampled PCM sample rate (Hz)
   * @param {boolean} [options.continuousMode=true] Whether to auto-resume listening after AI speaks
   * @param {boolean} [options.echoCancellation=true] Enable browser acoustic echo cancellation
   * @param {boolean} [options.noiseSuppression=true] Enable browser noise suppression
   * @param {boolean} [options.autoGainControl=true] Enable browser automatic gain control
   * @param {Function} [options.onVoiceStateChange] Callback: (newState, oldState) => {}
   * @param {Function} [options.onAudioChunk] Callback when speech PCM is ready: (arrayBuffer) => {}
   * @param {Function} [options.onSpeechStart] Callback when user starts speaking: (preSpeechChunks) => {}
   * @param {Function} [options.onSpeechEnd] Callback when silence timeout triggers: () => {}
   * @param {Function} [options.onVolumeLevel] Callback with live RMS volume level: (rms) => {}
   * @param {Function} [options.onError] Callback on audio errors: (err) => {}
   */
  constructor(options = {}) {
    this.silenceThreshold = options.silenceThreshold ?? 0.015;
    this.silenceTimeoutMs = options.silenceTimeoutMs ?? 1500;
    this.preSpeechBufferMs = options.preSpeechBufferMs ?? 400;
    this.targetSampleRate = options.targetSampleRate ?? 16000;
    this.continuousMode = options.continuousMode ?? true;

    this.echoCancellation = options.echoCancellation ?? true;
    this.noiseSuppression = options.noiseSuppression ?? true;
    this.autoGainControl = options.autoGainControl ?? true;

    // Callbacks
    this.onVoiceStateChange = options.onVoiceStateChange || (() => {});
    this.onAudioChunk = options.onAudioChunk || (() => {});
    this.onSpeechStart = options.onSpeechStart || (() => {});
    this.onSpeechEnd = options.onSpeechEnd || (() => {});
    this.onVolumeLevel = options.onVolumeLevel || (() => {});
    this.onError = options.onError || (() => {});

    // State machine
    this.voiceState = CopilotVoiceState.IDLE;

    // Web Audio API members
    this.audioContext = null;
    this.mediaStream = null;
    this.sourceNode = null;
    this.analyserNode = null;
    this.scriptProcessor = null;
    this.muteGainNode = null;
    this.resampler = null;

    // Pre-speech rolling buffer & silence timing
    this.preSpeechBuffer = [];
    this.maxPreSpeechChunks = 10;
    this.lastSpeechTimestamp = 0;
    this.silenceCheckInterval = null;

    // Audio element for TTS playback
    this.audioPlayer = new Audio();
  }

  /**
   * Get current voice state.
   * @returns {string}
   */
  get state() {
    return this.voiceState;
  }

  /**
   * Whether the microphone is actively capturing audio.
   * @returns {boolean}
   */
  get isCapturing() {
    return this.voiceState !== CopilotVoiceState.IDLE;
  }

  /**
   * Transition the state machine to a new state and notify listeners.
   * @param {string} newState One of CopilotVoiceState
   */
  setVoiceState(newState) {
    if (this.voiceState === newState) return;
    const oldState = this.voiceState;
    this.voiceState = newState;
    console.debug(`[AudioController] Voice state: ${oldState} -> ${newState}`);
    try {
      this.onVoiceStateChange(newState, oldState);
    } catch (e) {
      console.error('[AudioController] Error in onVoiceStateChange callback:', e);
    }
  }

  /**
   * Calculate Root-Mean-Square (RMS) energy of Float32 audio samples.
   * @param {Float32Array} samples 
   * @returns {number} RMS energy value in [0.0, 1.0]
   */
  calculateRMS(samples) {
    if (!samples || samples.length === 0) return 0;
    let sumSquares = 0;
    for (let i = 0; i < samples.length; i++) {
      sumSquares += samples[i] * samples[i];
    }
    return Math.sqrt(sumSquares / samples.length);
  }

  /**
   * Push a 16kHz PCM chunk into the circular pre-speech buffer.
   * @param {ArrayBuffer} buffer 
   */
  _pushPreSpeechBuffer(buffer) {
    this.preSpeechBuffer.push(buffer);
    if (this.preSpeechBuffer.length > this.maxPreSpeechChunks) {
      this.preSpeechBuffer.shift();
    }
  }

  /**
   * Flush and return all stored pre-speech chunks.
   * @returns {ArrayBuffer[]}
   */
  _flushPreSpeechBuffer() {
    const chunks = [...this.preSpeechBuffer];
    this.preSpeechBuffer = [];
    return chunks;
  }

  /**
   * Clear the pre-speech rolling buffer.
   */
  _clearPreSpeechBuffer() {
    this.preSpeechBuffer = [];
  }

  /**
   * Initialize microphone capture and start the VAD state machine in LISTENING_SILENT mode.
   * @returns {Promise<void>}
   */
  async start() {
    if (this.isCapturing) return;

    try {
      if (!navigator?.mediaDevices?.getUserMedia) {
        throw new Error('Web Audio getUserMedia is not supported in this browser environment.');
      }

      this.mediaStream = await navigator.mediaDevices.getUserMedia({
        audio: {
          channelCount: 1,
          echoCancellation: this.echoCancellation,
          noiseSuppression: this.noiseSuppression,
          autoGainControl: this.autoGainControl,
        }
      });

      const AudioCtxClass = window.AudioContext || window.webkitAudioContext;
      this.audioContext = new AudioCtxClass();
      const nativeSampleRate = this.audioContext.sampleRate;

      // Resampler from native rate (e.g. 44.1k/48k) to 16kHz
      this.resampler = new AudioResampler(nativeSampleRate, this.targetSampleRate);

      // Estimate max pre-speech chunks needed to hold preSpeechBufferMs of audio
      const bufferSize = 2048;
      const chunkDurationMs = (bufferSize / nativeSampleRate) * 1000;
      this.maxPreSpeechChunks = Math.max(3, Math.ceil(this.preSpeechBufferMs / chunkDurationMs));

      // Audio Graph Setup
      this.sourceNode = this.audioContext.createMediaStreamSource(this.mediaStream);
      this.analyserNode = this.audioContext.createAnalyser();
      this.analyserNode.fftSize = 512;
      this.analyserNode.smoothingTimeConstant = 0.2;

      this.scriptProcessor = this.audioContext.createScriptProcessor(bufferSize, 1, 1);

      // Create a zero-gain node so mic audio is processed without echoing to speakers
      this.muteGainNode = this.audioContext.createGain();
      this.muteGainNode.gain.value = 0.0;

      // Connect graph: Source -> Analyser -> ScriptProcessor -> MuteGain -> Destination
      this.sourceNode.connect(this.analyserNode);
      this.analyserNode.connect(this.scriptProcessor);
      this.scriptProcessor.connect(this.muteGainNode);
      this.muteGainNode.connect(this.audioContext.destination);

      // ScriptProcessor audio processing handler
      this.scriptProcessor.onaudioprocess = (event) => {
        this._handleAudioFrame(event);
      };

      // Periodic silence watcher (checks if speech paused even during quiet frames)
      this._startSilenceWatcher();

      // Transition to LISTENING_SILENT
      this.setVoiceState(CopilotVoiceState.LISTENING_SILENT);
      console.log(`[AudioController] Started continuous VAD listening (SampleRate: ${nativeSampleRate}Hz -> 16kHz).`);
    } catch (err) {
      console.error('[AudioController] Failed to initialize microphone capture:', err);
      this.stop();
      this.onError(err);
      throw err;
    }
  }

  /**
   * Process incoming Web Audio Float32 frame.
   * @param {AudioProcessingEvent} event 
   */
  _handleAudioFrame(event) {
    if (this.voiceState === CopilotVoiceState.IDLE) return;

    // Calculate RMS energy level of the incoming microphone frame
    const inputFloat32 = event.inputBuffer.getChannelData(0);
    const rms = this.calculateRMS(inputFloat32);

    // Notify volume visualizer callback
    try {
      this.onVolumeLevel(rms);
    } catch (e) {}

    // When AI is SPEAKING or backend is PROCESSING, discard mic input to prevent acoustic feedback
    if (this.voiceState === CopilotVoiceState.SPEAKING || this.voiceState === CopilotVoiceState.PROCESSING) {
      return;
    }

    // Downsample and convert Float32 to 16kHz PCM Int16Array
    const pcm16 = this.resampler.process(inputFloat32);
    const pcmBuffer = pcm16.buffer;

    // State 1: LISTENING_SILENT
    if (this.voiceState === CopilotVoiceState.LISTENING_SILENT) {
      this._pushPreSpeechBuffer(pcmBuffer);

      // Detect speech start when volume exceeds silence threshold
      if (rms >= this.silenceThreshold) {
        this.setVoiceState(CopilotVoiceState.LISTENING_SPEAKING);
        this.lastSpeechTimestamp = Date.now();

        // Flush all pre-speech buffer chunks to avoid cutting off the first syllable
        const preSpeechChunks = this._flushPreSpeechBuffer();
        try {
          this.onSpeechStart(preSpeechChunks);
        } catch (e) {
          console.error('[AudioController] Error in onSpeechStart:', e);
        }

        // Stream the current chunk as well
        try {
          this.onAudioChunk(pcmBuffer);
        } catch (e) {
          console.error('[AudioController] Error in onAudioChunk:', e);
        }
      }
      return;
    }

    // State 2: LISTENING_SPEAKING
    if (this.voiceState === CopilotVoiceState.LISTENING_SPEAKING) {
      // Stream audio chunk over WebSocket
      try {
        this.onAudioChunk(pcmBuffer);
      } catch (e) {
        console.error('[AudioController] Error in onAudioChunk:', e);
      }

      if (rms >= this.silenceThreshold) {
        // User continues speaking -> update speech timestamp
        this.lastSpeechTimestamp = Date.now();
      } else {
        // Volume below threshold -> check if silence duration exceeded
        const elapsedSilence = Date.now() - this.lastSpeechTimestamp;
        if (elapsedSilence >= this.silenceTimeoutMs) {
          this._triggerSpeechEnd();
        }
      }
    }
  }

  /**
   * Start interval timer to enforce silence timeout when no frames exceed threshold.
   */
  _startSilenceWatcher() {
    this._stopSilenceWatcher();
    this.silenceCheckInterval = setInterval(() => {
      if (this.voiceState === CopilotVoiceState.LISTENING_SPEAKING) {
        const elapsedSilence = Date.now() - this.lastSpeechTimestamp;
        if (elapsedSilence >= this.silenceTimeoutMs) {
          this._triggerSpeechEnd();
        }
      }
    }, 100);
  }

  /**
   * Stop interval timer.
   */
  _stopSilenceWatcher() {
    if (this.silenceCheckInterval) {
      clearInterval(this.silenceCheckInterval);
      this.silenceCheckInterval = null;
    }
  }

  /**
   * Trigger transition from LISTENING_SPEAKING to PROCESSING on silence timeout.
   */
  _triggerSpeechEnd() {
    if (this.voiceState !== CopilotVoiceState.LISTENING_SPEAKING) return;
    this.setVoiceState(CopilotVoiceState.PROCESSING);
    this._clearPreSpeechBuffer();

    try {
      this.onSpeechEnd();
    } catch (e) {
      console.error('[AudioController] Error in onSpeechEnd:', e);
    }
  }

  /**
   * Play synthesized WAV audio response (Base64 WAV) and handle transition to SPEAKING and back.
   * @param {string} base64Wav Base64-encoded WAV audio data
   * @returns {Promise<void>} Resolves when audio playback completes
   */
  playAudioWav(base64Wav) {
    return new Promise((resolve) => {
      try {
        const binary = atob(base64Wav);
        const bytes = new Uint8Array(binary.length);
        for (let i = 0; i < binary.length; i++) {
          bytes[i] = binary.charCodeAt(i);
        }
        const blob = new Blob([bytes.buffer], { type: 'audio/wav' });
        const url = URL.createObjectURL(blob);

        this.setVoiceState(CopilotVoiceState.SPEAKING);

        this.audioPlayer.src = url;
        this.audioPlayer.volume = 1.0;

        const onFinish = () => {
          this.audioPlayer.removeEventListener('ended', onFinish);
          this.audioPlayer.removeEventListener('error', onFinish);
          URL.revokeObjectURL(url);
          this._onPlaybackFinished();
          resolve();
        };

        this.audioPlayer.addEventListener('ended', onFinish);
        this.audioPlayer.addEventListener('error', onFinish);

        this.audioPlayer.play().catch((err) => {
          console.warn('[AudioController] Audio play warning:', err);
          onFinish();
        });
      } catch (err) {
        console.error('[AudioController] Failed to play audio WAV:', err);
        this._onPlaybackFinished();
        resolve();
      }
    });
  }

  /**
   * Synthesize speech using browser Web Speech API SpeechSynthesisUtterance.
   * @param {string} text 
   * @returns {Promise<void>}
   */
  speakUtterance(text) {
    return new Promise((resolve) => {
      if (typeof window === 'undefined' || !('speechSynthesis' in window)) {
        this._onPlaybackFinished();
        return resolve();
      }

      try {
        window.speechSynthesis.cancel();
        this.setVoiceState(CopilotVoiceState.SPEAKING);

        const utterance = new SpeechSynthesisUtterance(text);
        const isRussian = /[а-яёА-ЯЁ]/.test(text);
        utterance.lang = isRussian ? 'ru-RU' : 'en-US';
        utterance.rate = 1.05;
        utterance.pitch = 1.0;

        const voices = window.speechSynthesis.getVoices();
        const targetLang = isRussian ? 'ru' : 'en';
        const bestVoice = voices.find(v => v.lang.toLowerCase().startsWith(targetLang));
        if (bestVoice) {
          utterance.voice = bestVoice;
        }

        const onFinish = () => {
          utterance.onend = null;
          utterance.onerror = null;
          this._onPlaybackFinished();
          resolve();
        };

        utterance.onend = onFinish;
        utterance.onerror = onFinish;

        window.speechSynthesis.speak(utterance);
      } catch (err) {
        console.warn('[AudioController] SpeechSynthesis warning:', err);
        this._onPlaybackFinished();
        resolve();
      }
    });
  }

  /**
   * Called when AI finishes speaking its response.
   * Automatically transitions back to LISTENING_SILENT if continuous mode is enabled.
   */
  _onPlaybackFinished() {
    this._clearPreSpeechBuffer();
    this.lastSpeechTimestamp = 0;

    if (this.continuousMode && this.mediaStream) {
      // Auto-resume continuous listening for the next conversational turn
      this.setVoiceState(CopilotVoiceState.LISTENING_SILENT);
      console.log('[AudioController] Finished speaking; auto-resumed to LISTENING_SILENT.');
    } else {
      this.stop();
    }
  }

  /**
   * Handle non-audio response completion (e.g., text-only action or error).
   */
  handleResponseFinished() {
    if (this.voiceState === CopilotVoiceState.PROCESSING) {
      this._onPlaybackFinished();
    }
  }

  /**
   * Stop recording, close audio nodes and streams, and return to IDLE state.
   */
  stop() {
    this._stopSilenceWatcher();
    this._clearPreSpeechBuffer();

    if (this.audioPlayer) {
      try {
        this.audioPlayer.pause();
      } catch (e) {}
    }

    if (typeof window !== 'undefined' && window.speechSynthesis) {
      try {
        window.speechSynthesis.cancel();
      } catch (e) {}
    }

    if (this.scriptProcessor) {
      try {
        this.scriptProcessor.disconnect();
      } catch (e) {}
      this.scriptProcessor = null;
    }

    if (this.analyserNode) {
      try {
        this.analyserNode.disconnect();
      } catch (e) {}
      this.analyserNode = null;
    }

    if (this.sourceNode) {
      try {
        this.sourceNode.disconnect();
      } catch (e) {}
      this.sourceNode = null;
    }

    if (this.muteGainNode) {
      try {
        this.muteGainNode.disconnect();
      } catch (e) {}
      this.muteGainNode = null;
    }

    if (this.mediaStream) {
      try {
        this.mediaStream.getTracks().forEach((track) => track.stop());
      } catch (e) {}
      this.mediaStream = null;
    }

    if (this.audioContext && this.audioContext.state !== 'closed') {
      try {
        this.audioContext.close();
      } catch (e) {}
      this.audioContext = null;
    }

    this.setVoiceState(CopilotVoiceState.IDLE);
    console.log('[AudioController] Stopped microphone capture; voice state is IDLE.');
  }
}
