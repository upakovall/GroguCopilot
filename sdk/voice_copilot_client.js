/**
 * Standalone Voice Copilot Client SDK.
 * 
 * Reusable ES6 client for integrating Voice AI Copilot into ANY web application.
 * Features dual-mode Speech Recognition (Web Speech API + 16kHz PCM WebSocket stream),
 * natural voice speech synthesis, and structured UIAction event dispatching.
 */

import { AudioResampler } from './audio_resampler.js';

export class VoiceCopilotClient {
  /**
   * @param {Object} config Configuration options
   * @param {string} config.wsUrl WebSocket endpoint URL
   * @param {Function} [config.onUIAction] Callback when a UIAction is commanded: (action) => {}
   * @param {Function} [config.onTranscription] Callback on STT transcript: (text, isFinal) => {}
   * @param {Function} [config.onAgentThinking] Callback on status update: (text) => {}
   * @param {Function} [config.onAgentResponse] Callback on full response: (response) => {}
   * @param {Function} [config.onAudioResponse] Callback on synthesized speech: (base64Wav) => {}
   * @param {Function} [config.onStateChange] Callback on connection state: (state) => {}
   * @param {Function} [config.onError] Callback on error: (err) => {}
   */
  constructor(config = {}) {
    this.wsUrl = config.wsUrl || this._getDefaultWsUrl();
    this.onUIAction = config.onUIAction || (() => {});
    this.onTranscription = config.onTranscription || (() => {});
    this.onAgentThinking = config.onAgentThinking || (() => {});
    this.onAgentResponse = config.onAgentResponse || (() => {});
    this.onAudioResponse = config.onAudioResponse || (() => {});
    this.onStateChange = config.onStateChange || (() => {});
    this.onError = config.onError || (() => {});

    this.ws = null;
    this.isConnected = false;
    this.isRecording = false;

    // Audio capture members
    this.audioContext = null;
    this.mediaStream = null;
    this.scriptProcessor = null;
    this.resampler = null;
    this.audioPlayer = new Audio();

    // Browser Speech Recognition engine
    this.recognition = null;
    this.recognizedText = '';
    this._initSpeechRecognition();
  }

  _initSpeechRecognition() {
    if (typeof window !== 'undefined') {
      const SpeechRecognition = window.SpeechRecognition || window.webkitSpeechRecognition;
      if (SpeechRecognition) {
        this.recognition = new SpeechRecognition();
        this.recognition.continuous = true;
        this.recognition.interimResults = true;
        this.recognition.lang = navigator.language || 'ru-RU';

        this.recognition.onresult = (event) => {
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

  _getDefaultWsUrl() {
    if (typeof window !== 'undefined') {
      const proto = window.location.protocol === 'https:' ? 'wss:' : 'ws:';
      return `${proto}//${window.location.host}/ws/copilot`;
    }
    return 'ws://localhost:8000/ws/copilot';
  }

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
      };

      this.ws.onmessage = (event) => {
        if (typeof event.data === 'string') {
          try {
            const msg = JSON.parse(event.data);
            this._handleMessage(msg);
          } catch (e) {
            console.error('[VoiceCopilotClient] Parse error:', e);
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

  _handleMessage(msg) {
    switch (msg.type) {
      case 'TRANSCRIPTION':
        this.onTranscription(msg.text, msg.is_final);
        break;

      case 'AGENT_THINKING':
        this.onAgentThinking(msg.text);
        break;

      case 'AGENT_RESPONSE':
        if (msg.agent_response) {
          this.onAgentResponse(msg.agent_response);
          
          // Speak natural voice response
          if (msg.agent_response.speech_output) {
            this.speakNaturalVoice(msg.agent_response.speech_output);
          }

          if (msg.agent_response.actions) {
            msg.agent_response.actions.forEach(action => {
              this.onUIAction(action);
            });
            this.send({
              type: 'ACTION_ACK',
              data: { count: msg.agent_response.actions.length, status: 'success' }
            });
          }
        }
        break;

      case 'AUDIO_RESPONSE':
        if (msg.audio_base64) {
          this.onAudioResponse(msg.audio_base64);
          this.playAudioChime(msg.audio_base64);
        }
        break;

      case 'ERROR':
        this.onError(new Error(msg.error));
        break;
    }
  }

  speakNaturalVoice(text) {
    if (typeof window === 'undefined' || !('speechSynthesis' in window)) return;
    try {
      window.speechSynthesis.cancel();
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

      window.speechSynthesis.speak(utterance);
    } catch (e) {
      console.warn('[VoiceCopilotClient] Speech synthesis note:', e);
    }
  }

  syncViewContext(viewContext) {
    if (this.isConnected && this.ws) {
      this.send({
        type: 'VIEW_CONTEXT_UPDATE',
        view_context: viewContext
      });
    }
  }

  send(payload) {
    if (this.isConnected && this.ws) {
      this.ws.send(JSON.stringify(payload));
    }
  }

  sendBinary(arrayBuffer) {
    if (this.isConnected && this.ws) {
      this.ws.send(arrayBuffer);
    }
  }

  sendTextPrompt(text) {
    this.send({
      type: 'TEXT_PROMPT',
      text: text
    });
  }

  async startListening() {
    if (this.isRecording) return;
    this.isRecording = true;
    this.recognizedText = '';

    // Start native browser speech recognition if supported
    if (this.recognition) {
      try {
        this.recognition.start();
      } catch (e) {
        console.debug('[VoiceCopilotClient] recognition start note:', e);
      }
    }

    try {
      if (navigator.mediaDevices && navigator.mediaDevices.getUserMedia) {
        this.mediaStream = await navigator.mediaDevices.getUserMedia({
          audio: {
            channelCount: 1,
            echoCancellation: true,
            noiseSuppression: true,
          }
        });

        this.audioContext = new (window.AudioContext || window.webkitAudioContext)();
        const nativeSampleRate = this.audioContext.sampleRate;
        this.resampler = new AudioResampler(nativeSampleRate, 16000);

        const source = this.audioContext.createMediaStreamSource(this.mediaStream);
        this.scriptProcessor = this.audioContext.createScriptProcessor(2048, 1, 1);
        
        source.connect(this.scriptProcessor);
        this.scriptProcessor.connect(this.audioContext.destination);

        this.scriptProcessor.onaudioprocess = (e) => {
          if (!this.isRecording) return;
          const inputFloat32 = e.inputBuffer.getChannelData(0);
          const pcm16 = this.resampler.process(inputFloat32);
          this.sendBinary(pcm16.buffer);
        };
        return;
      }
    } catch (e) {
      console.warn('[VoiceCopilotClient] Microphone capture note:', e);
    }
  }

  stopListening(simulatedVoiceText = null) {
    this.isRecording = false;

    if (this.recognition) {
      try {
        this.recognition.stop();
      } catch (e) {}
    }

    if (this.scriptProcessor) {
      this.scriptProcessor.disconnect();
      this.scriptProcessor = null;
    }

    if (this.mediaStream) {
      this.mediaStream.getTracks().forEach(t => t.stop());
      this.mediaStream = null;
    }

    const finalText = (simulatedVoiceText || this.recognizedText).trim();

    if (finalText) {
      this.onTranscription(finalText, true);
      // Send text directly to guarantee 100% accurate transcription
      this.send({
        type: 'TEXT_PROMPT',
        text: finalText
      });
      return;
    }

    // Otherwise signal AUDIO_END for backend STT processing
    this.send({ type: 'AUDIO_END' });
  }

  playAudioChime(base64Wav) {
    try {
      const binary = atob(base64Wav);
      const bytes = new Uint8Array(binary.length);
      for (let i = 0; i < binary.length; i++) {
        bytes[i] = binary.charCodeAt(i);
      }
      const blob = new Blob([bytes.buffer], { type: 'audio/wav' });
      const url = URL.createObjectURL(blob);
      this.audioPlayer.src = url;
      this.audioPlayer.volume = 0.35;
      this.audioPlayer.play().catch(e => {
        console.debug('[VoiceCopilotClient] Play note:', e);
      });
    } catch (e) {
      console.error('[VoiceCopilotClient] Chime playback error:', e);
    }
  }
}
