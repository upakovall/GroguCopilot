# 🎙️ Grogu Voice AI Copilot (`grogu-copilot`)

> **Universal, UI-Aware Voice AI Copilot for Web Platforms (16GB VRAM Limit)**

`grogu-copilot` is a standalone, decoupled Python & JavaScript library that equips any web application with bidirectional voice AI control, semantic UI awareness, and structured action dispatching without DOM scraping.

---

## 🌟 Key Features

- **🚫 Zero DOM Scraping**: Operates exclusively on declarative `ViewContext` (Semantic UI State).
- **🗣️ Continuous Voice Conversation (ChatGPT Voice Mode)**:
  - Client-Side Voice Activity Detection (VAD) via Web Audio API `AnalyserNode`.
  - Rolling pre-speech buffer (~400ms) prevents clipping initial syllables.
  - Zero backend CPU saturation (silence is never streamed over WebSockets).
  - Automatic conversation resumption after TTS playback completes.
- **⚡ Strict 16GB VRAM Budget**:
  - **STT** (`faster-whisper` CPU `int8`): Multi-threaded CPU execution (**0.0 GB VRAM**).
  - **TTS** (Acoustic Chime / Web Speech API): Lightweight CPU synthesis (**0.0 GB VRAM**).
  - **LLM** (`vLLM` / `llama.cpp` Qwen2.5-7B AWQ / Llama-3.1-8B): Dedicated **~11.0 GB VRAM** allocation for weights + 8K KV-Cache.
- **🛡️ Inversion of Control (`MCPRegistry`)**: Host applications inject custom safety validators and execution hooks.
- **🎯 Guided JSON Decoding**: 100% grammar-constrained JSON output conforming to strict Pydantic schemas.
- **🌐 Dual-Mode Audio Pipeline**: 16kHz PCM downsampler + native browser live SpeechRecognition.

---

## 📦 Installation

```bash
# Install directly from Git:
pip install git+https://github.com/your-username/grogu-copilot.git

# Or install with local neural STT (faster-whisper):
pip install "grogu-copilot[local-ai] @ git+https://github.com/your-username/grogu-copilot.git"
```

---

## 🚀 Quickstart

### 1. Backend Integration (FastAPI)

```python
from fastapi import FastAPI
from grogu_copilot import create_copilot_router, MCPRegistry, UIAction, ViewContext, ActionType

app = FastAPI(title="My Application")
registry = MCPRegistry()

# Optional: Register custom safety validator
def validate_amount(action: UIAction, ctx: ViewContext):
    if action.target_id == "quantity_input" and float(action.payload.get("value", 0)) > 100:
        return False, "Policy error: Quantity exceeds limit (100)."
    return True, None

registry.register_action_validator(ActionType.SET_INPUT_VALUE.value, validate_amount)

# Mount the decoupled Voice AI Copilot Router
app.include_router(create_copilot_router(
    registry=registry,
    llm_backend="dynamic",  # "dynamic", "vllm", "llama_cpp", "runpod"
    endpoint_path="/ws/copilot"
))
```

### 2. Frontend Integration (Browser ES6)

```html
<script type="module">
  import { VoiceCopilotClient } from './sdk/voice_copilot_client.js';

  const copilot = new VoiceCopilotClient({
    wsUrl: `ws://${window.location.host}/ws/copilot`,
    onUIAction: (action) => {
      console.log('Received UIAction command:', action);
      // Mutate application state:
      if (action.target_id === 'quantity_input') {
        document.getElementById('quantity_input').value = action.payload.value;
      }
    },
    onTranscription: (text, isFinal) => {
      console.log('STT Live Transcript:', text);
    },
    onAgentResponse: (response) => {
      console.log('Thought:', response.thought);
      console.log('Speech:', response.speech_output);
    }
  });

  copilot.connect();

  // Send declarative snapshot of actionable elements
  copilot.syncViewContext({
    screen_id: 'checkout',
    title: 'Checkout Page',
    components: [
      { id: 'quantity_input', type: 'input', label: 'Item Quantity', value: 1, allowed_actions: ['set_value'] },
      { id: 'submit_btn', type: 'button', label: 'Place Order', allowed_actions: ['click'] }
    ]
  });

  // Start/Stop microphone recording
  document.getElementById('mic_btn').onclick = () => {
    copilot.isRecording ? copilot.stopListening() : copilot.startListening();
  };
</script>
```

---

## 🧪 Running Tests

```bash
pip install -e ".[dev]"
pytest -v
```

---

## 📄 License

MIT License. Copyright (c) 2026 Grogu AI Team.
