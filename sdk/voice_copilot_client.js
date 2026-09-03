/**
 * Standalone Voice Copilot Client SDK with Client-Side VAD.
 *
 * Universal ES6 client for integrating Voice AI Copilot into ANY web application.
 * Features Continuous Voice Conversation mode powered by Web Audio API AnalyserNode VAD,
 * rolling pre-speech buffer, dual-mode Speech Recognition, natural speech synthesis,
 * and zero-DOM-scraping declarative ViewContext synchronization.
 */

import { AudioController, CopilotVoiceState } from './audio_controller.js';
import { AudioResampler } from './audio_resampler.js';

export { CopilotVoiceState } from './audio_controller.js';

export class VoiceCopilotClient {
  /**
   * @param {Object} [config={}] Configuration options
   * @param {string} [config.wsUrl] WebSocket endpoint URL (default: auto-detected)
   * @param {boolean} [config.continuousMode=true] Enable continuous conversational voice mode
   * @param {number} [config.silenceThreshold=0.015] RMS energy threshold for speech detection
   * @param {number} [config.silenceTimeoutMs=1500] Milliseconds of silence to trigger end-of-speech
   * @param {number} [config.preSpeechBufferMs=400] Milliseconds of pre-speech rolling audio buffer
   * @param {Function} [config.onUIAction] Callback when a UIAction is commanded: (action) => {}
   * @param {Function} [config.onTranscription] Callback on STT transcript: (text, isFinal) => {}
   * @param {Function} [config.onAgentThinking] Callback on status update: (text) => {}
   * @param {Function} [config.onAgentResponse] Callback on full response: (response) => {}
   * @param {Function} [config.onAudioResponse] Callback on synthesized speech: (base64Wav) => {}
   * @param {Function} [config.onStateChange] Callback on connection state: (state) => {}
   * @param {Function} [config.onVoiceStateChange] Callback on voice VAD state: (newState, oldState) => {}
   * @param {Function} [config.onVolumeLevel] Callback on live mic volume level: (rms) => {}
   * @param {Function} [config.onError] Callback on error: (err) => {}
   */
  constructor(config = {}) {
    this.wsUrl = config.wsUrl || this._getDefaultWsUrl();
    this.continuousMode = config.continuousMode ?? true;

    // Callbacks
    this.onUIAction = config.onUIAction || (() => {});
    this.onTranscription = config.onTranscription || (() => {});
    this.onAgentThinking = config.onAgentThinking || (() => {});
    this.onAgentResponse = config.onAgentResponse || (() => {});
    this.onAudioResponse = config.onAudioResponse || (() => {});
    this.onStateChange = config.onStateChange || (() => {});
    this.onVoiceStateChange = config.onVoiceStateChange || (() => {});
    this.onVolumeLevel = config.onVolumeLevel || (() => {});
    this.onError = config.onError || (() => {});

    // WebSocket state
    this.ws = null;
    this.isConnected = false;

    // Active ViewContext cache for resynchronization on reconnect
    this.activeViewContext = null;

    // Initialize Client-Side AudioController with VAD State Machine
    this.audioController = new AudioController({
      silenceThreshold: config.silenceThreshold ?? 0.015,
      silenceTimeoutMs: config.silenceTimeoutMs ?? 1500,
      preSpeechBufferMs: config.preSpeechBufferMs ?? 400,
      targetSampleRate: 16000,
      continuousMode: this.continuousMode,
      onVoiceStateChange: (newState, oldState) => {
        this.onVoiceStateChange(newState, oldState);
      },
      onSpeechStart: (preSpeechChunks) => {
        this._handleSpeechStart(preSpeechChunks);
      },
      onAudioChunk: (pcmBuffer) => {
        this._handleAudioChunk(pcmBuffer);
      },
      onSpeechEnd: () => {
        this._handleSpeechEnd();
      },
      onVolumeLevel: (rms) => {
        this.onVolumeLevel(rms);
      },
      onError: (err) => {
        this.onError(err);
      }
    });

    // Browser native Speech Recognition engine for low-latency visual feedback
    this.recognition = null;
    this.recognizedText = '';
    this._initSpeechRecognition();
  }

  /**
   * Whether the microphone is currently active.
   * @returns {boolean}
   */
  get isRecording() {
    return this.audioController.isCapturing;
  }

  /**
   * Current voice activity state: 'IDLE' | 'LISTENING_SILENT' | 'LISTENING_SPEAKING' | 'PROCESSING' | 'SPEAKING'
   * @returns {string}
   */
  get voiceState() {
    return this.audioController.state;
  }

  /**
   * Initialize native browser SpeechRecognition for supplementary live transcription.
   * @private
   */
  _initSpeechRecognition() {
    if (typeof window !== 'undefined') {
      const SpeechRecognition = window.SpeechRecognition || window.webkitSpeechRecognition;
      if (SpeechRecognition) {
        this.recognition = new SpeechRecognition();
        this.recognition.continuous = true;
        this.recognition.interimResults = true;
        this.recognition.lang = navigator.language || 'ru-RU';

        this.recognition.onresult = (event) => {
          // Ignore recognition results while AI is speaking
          if (this.audioController.state === CopilotVoiceState.SPEAKING) {
            return;
          }

          let interim = '';
          for (let i = event.resultIndex; i < event.results.length; ++i) {
            const transcript = event.results[i][0].transcript;
            if (event.results[i].isFinal) {
              this.recognizedText += transcript + ' ';
            } else {
              interim += transcript;
            }
          }
          const full = (this.recognizedText + ' ' + interim).trim();
          if (full) {
            this.onTranscription(full, false);
          }
        };

        this.recognition.onerror = (e) => {
          console.debug('[VoiceCopilotClient] SpeechRecognition note:', e.error);
        };
      }
    }
  }

  /**
   * Auto-detect default WebSocket endpoint URL based on current window location.
   * @private
   * @returns {string}
   */
  _getDefaultWsUrl() {
    if (typeof window !== 'undefined') {
      const proto = window.location.protocol === 'https:' ? 'wss:' : 'ws:';
      return `${proto}//${window.location.host}/ws/copilot`;
    }
    return 'ws://localhost:8000/ws/copilot';
  }

  /**
   * Connect to the Grogu Voice Copilot WebSocket backend.
   */
  connect() {
    console.log(`[VoiceCopilotClient] Connecting to ${this.wsUrl}...`);
    this.onStateChange('connecting');

    try {
      this.ws = new WebSocket(this.wsUrl);
      this.ws.binaryType = 'arraybuffer';

      this.ws.onopen = () => {
        this.isConnected = true;
        console.log('[VoiceCopilotClient] Connected successfully.');
        this.onStateChange('connected');

        // Resync active ViewContext if already set
        if (this.activeViewContext) {
          this.syncViewContext(this.activeViewContext);
        }
      };

      this.ws.onmessage = (event) => {
        if (typeof event.data === 'string') {
          try {
            const msg = JSON.parse(event.data);
            this._handleMessage(msg);
          } catch (e) {
            console.error('[VoiceCopilotClient] JSON Parse error:', e);
          }
        }
      };

      this.ws.onclose = () => {
        this.isConnected = false;
        console.log('[VoiceCopilotClient] Connection closed.');
        this.onStateChange('disconnected');
      };

      this.ws.onerror = (err) => {
        console.error('[VoiceCopilotClient] WS error:', err);
        this.onError(err);
      };
    } catch (e) {
      console.error('[VoiceCopilotClient] Connection failed:', e);
      this.onError(e);
    }
  }

  /**
   * Disconnect from the WebSocket backend and release audio resources.
   */
  disconnect() {
    this.stopListening();
    if (this.ws) {
      try {
        this.ws.close();
      } catch (e) {}
      this.ws = null;
    }
    this.isConnected = false;
  }

  /**
   * Handler for speech start: flushes the rolling pre-speech buffer to ensure the first syllable is streamed.
   * @private
   * @param {ArrayBuffer[]} preSpeechChunks 
   */
  _handleSpeechStart(preSpeechChunks) {
    this.recognizedText = '';
    // Send all buffered pre-speech audio chunks over WebSocket
    if (preSpeechChunks && preSpeechChunks.length > 0) {
      for (const chunk of preSpeechChunks) {
        this.sendBinary(chunk);
      }
    }
  }

  /**
   * Handler for live audio chunks: streams 16kHz PCM data over the WebSocket binary channel.
   * @private
   * @param {ArrayBuffer} pcmBuffer 
   */
  _handleAudioChunk(pcmBuffer) {
    this.sendBinary(pcmBuffer);
  }

  /**
   * Handler for speech pause / silence timeout: sends AUDIO_END signal to initiate backend STT and reasoning.
   * @private
   */
  _handleSpeechEnd() {
    console.log('[VoiceCopilotClient] Silence timeout reached. Dispatched AUDIO_END to server.');
    this.send({ type: 'AUDIO_END' });
  }

  /**
   * Process inbound server WebSocket envelopes.
   * @private
   * @param {Object} msg 
   */
  _handleMessage(msg) {
    switch (msg.type) {
      case 'SESSION_INIT':
        console.log('[VoiceCopilotClient] Server session initialized:', msg.data);
        break;

      case 'TRANSCRIPTION':
        this.onTranscription(msg.text, msg.is_final);
        break;

      case 'AGENT_THINKING':
        this.onAgentThinking(msg.text);
        break;

      case 'AGENT_RESPONSE':
        if (msg.agent_response) {
          this.onAgentResponse(msg.agent_response);

          // Dispatch structured UI actions
          if (msg.agent_response.actions && msg.agent_response.actions.length > 0) {
            msg.agent_response.actions.forEach((action) => {
              try {
                this.onUIAction(action);
              } catch (err) {
                console.error('[VoiceCopilotClient] Error executing UIAction:', err);
              }
            });

            // Send action execution acknowledgment
            this.send({
              type: 'ACTION_ACK',
              data: { count: msg.agent_response.actions.length, status: 'success' }
            });
          }

          // Fallback: If backend does not send an AUDIO_RESPONSE WAV, speak via Web Speech API
          if (msg.agent_response.speech_output && !this._expectingAudioWav) {
            this.speakNaturalVoice(msg.agent_response.speech_output);
          }
        }
        break;

      case 'AUDIO_RESPONSE':
        if (msg.audio_base64) {
          this._expectingAudioWav = true;
          this.onAudioResponse(msg.audio_base64);
          // Play backend synthesized WAV response; auto-resumes to LISTENING_SILENT upon completion
          this.playAudioChime(msg.audio_base64);
        }
        break;

      case 'AUDIO_STREAM_END':
        this._expectingAudioWav = false;
        // If no audio was played and client is still in PROCESSING state, resume listening
        if (this.audioController.state === CopilotVoiceState.PROCESSING) {
          this.audioController.handleResponseFinished();
        }
        break;

      case 'ERROR':
        this.onError(new Error(msg.error));
        if (this.audioController.state === CopilotVoiceState.PROCESSING) {
          this.audioController.handleResponseFinished();
        }
        break;

      default:
        console.debug('[VoiceCopilotClient] Unhandled message type:', msg.type);
    }
  }

  /**
   * Speak speech text using client-side Web Speech API and handle state transition.
   * @param {string} text Spoken response text
   * @returns {Promise<void>}
   */
  async speakNaturalVoice(text) {
    return this.audioController.speakUtterance(text);
  }

  /**
   * Play base64-encoded WAV chime/voice response and auto-resume continuous listening.
   * @param {string} base64Wav 
   * @returns {Promise<void>}
   */
  async playAudioChime(base64Wav) {
    return this.audioController.playAudioWav(base64Wav);
  }

  /**
   * Synchronize the declarative ViewContext snapshot with the backend session.
   * @param {Object} viewContext Declarative ViewContext object containing screen_id, title, components, etc.
   */
  syncViewContext(viewContext) {
    this.activeViewContext = viewContext;
    if (this.isConnected && this.ws) {
      this.send({
        type: 'VIEW_CONTEXT_UPDATE',
        view_context: viewContext
      });
    }
  }

  /**
   * Send JSON message payload over WebSocket.
   * @param {Object} payload 
   */
  send(payload) {
    if (this.isConnected && this.ws && this.ws.readyState === WebSocket.OPEN) {
      this.ws.send(JSON.stringify(payload));
    }
  }

  /**
   * Send binary ArrayBuffer (16kHz PCM audio chunk) over WebSocket.
   * @param {ArrayBuffer} arrayBuffer 
   */
  sendBinary(arrayBuffer) {
    if (this.isConnected && this.ws && this.ws.readyState === WebSocket.OPEN) {
      this.ws.send(arrayBuffer);
    }
  }

  /**
   * Send a direct text command to the Copilot.
   * @param {string} text Natural language text prompt
   */
  sendTextPrompt(text) {
    this.send({
      type: 'TEXT_PROMPT',
      text: text
    });
  }

  /**
   * Start microphone capture with client-side VAD in continuous conversation mode.
   * @returns {Promise<void>}
   */
  async startListening() {
    this.recognizedText = '';

    // Start native browser SpeechRecognition for interim live preview
    if (this.recognition) {
      try {
        this.recognition.start();
      } catch (e) {
        console.debug('[VoiceCopilotClient] Recognition start note:', e);
      }
    }

    await this.audioController.start();
  }

  /**
   * Stop microphone capture and transition to IDLE state.
   * @param {string} [simulatedVoiceText=null] Optional text override for instant testing
   */
  stopListening(simulatedVoiceText = null) {
    if (this.recognition) {
      try {
        this.recognition.stop();
      } catch (e) {}
    }

    const wasSpeaking = (this.audioController.state === CopilotVoiceState.LISTENING_SPEAKING);
    this.audioController.stop();

    const finalText = (simulatedVoiceText || this.recognizedText).trim();
    if (finalText) {
      this.onTranscription(finalText, true);
      this.sendTextPrompt(finalText);
      return;
    }

    if (wasSpeaking) {
      this.send({ type: 'AUDIO_END' });
    }
  }

  /**
   * Configure whether continuous conversational mode is enabled.
   * @param {boolean} enabled 
   */
  setContinuousMode(enabled) {
    this.continuousMode = !!enabled;
    this.audioController.continuousMode = this.continuousMode;
  }

  /**
   * Configure the VAD RMS silence detection threshold.
   * @param {number} threshold E.g., 0.015
   */
  setSilenceThreshold(threshold) {
    this.audioController.silenceThreshold = Number(threshold);
  }

  /**
   * Configure the silence timeout duration in milliseconds.
   * @param {number} timeoutMs E.g., 1500 (1.5 seconds)
   */
  setSilenceTimeout(timeoutMs) {
    this.audioController.silenceTimeoutMs = Number(timeoutMs);
  }
}
