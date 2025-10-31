# Zentrax — Voice & Gesture Desktop Controller

Lightweight voice and gesture control for desktop interactions. This repository provides a hybrid recognizer (Whisper + Google fallback), gesture detection (MediaPipe), and optional integrations (WhatsApp, a small Hill Climb demo). The code is designed to run on machines without all optional dependencies by providing safe fallbacks.

## Quick start (Windows PowerShell)

1. Create and activate a virtual environment (recommended):

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
```

2. Install required packages:

```powershell
# install base requirements
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

3. (Optional but recommended on Windows) Install PyAudio using pipwin to get reliable microphone support:

```powershell
python -m pip install pipwin
python -m pipwin install pyaudio
```

If pipwin cannot be used, download a prebuilt PyAudio wheel from Christoph Gohlke's site and install it with `pip install <wheel-file>`: https://www.lfd.uci.edu/~gohlke/pythonlibs/#pyaudio

4. Run the main program:

```powershell
python main.py
```

## Optional: Whisper & Torch (for offline, higher-quality transcription)

Whisper is optional. If installed, the app will lazily load the Whisper model and use it as the primary recognizer with Google as the fallback. Installing Torch can be system-specific — use the official instructions at https://pytorch.org/ for the correct command for your CUDA / CPU setup.

Basic install example (CPU):

```powershell
# Install Whisper (module name: whisper) and a CPU PyTorch wheel
python -m pip install --upgrade openai-whisper
# Follow https://pytorch.org/ for the correct torch command for your system
python -m pip install torch --index-url https://download.pytorch.org/whl/cpu
```

Note: model download happens when the Whisper model is first used. Expect additional download time and disk usage.

## Behavior & features

- Voice recognition: uses a HybridRecognizer (Whisper if available, otherwise Google Web Speech API via speech_recognition).
- Gesture recognition: uses MediaPipe to detect common hand gestures (open palm, closed fist, thumbs up/down).
- Non-blocking transcription: audio is enqueued and transcribed in a background worker so gesture processing and UI remain responsive.
- Safe fallbacks: when optional dependencies (PyAudio or Whisper) are missing the app tries alternative paths (system default mic, sounddevice fallback if installed, Google API fallback for speech). Minimal stubs are included for optional modules so the app doesn't crash on import.

## Usage notes

- Wake phrase: default is `hello`. Change `self.wake_phrase` in `main.py` if you want a different trigger.
- Voice commands include (examples): `open browser`, `play music <song>`, `send whatsapp message to <contact> saying <message>`, `exit program`, `play hill climb`.
- Mode switching: say `switch to gesture mode` or `switch to voice mode` to change active input mode.

## Troubleshooting

- No audio devices found / PyAudio errors:
  - Install PyAudio with pipwin (Windows) or system package manager.
  - As a fallback, install `sounddevice` to enable the alternative recorder used when PyAudio is unavailable.

- Whisper model fails to load or out of memory:
  - Ensure correct `torch` variant is installed for your platform (CUDA vs CPU).
  - If Whisper can't be loaded the app will fall back to Google speech recognition.

## Testing checklist

1. With PyAudio installed: confirm microphone lists and recognition works via `sr.Microphone`.
2. Without PyAudio but with `sounddevice` installed: confirm the fallback recorder can capture audio and trigger recognition.
3. With Whisper installed: confirm the model downloads and transcribes audio (lazy-load occurs when first used).
4. Test gestures: run the program and press `q` in the hand-tracking window to quit; ensure frames are captured and gestures map to expected actions.

## Development notes

- The main entrypoint is `main.py` and the Whisper wrapper is in `whisper_handler.py`.
- To tweak behavior (wake phrase, microphone selection, gesture sensitivity), edit `main.py`.
- Consider adding unit tests for `WhisperHandler.transcribe_audio` and `VoiceGestureControl._handle_recognized_text` as next steps.

## Contributing

Contributions welcome — open an issue or PR. If you add features that change the public behavior, update this README with usage and tests.

## License

See repository-level license (if present). If none, clarify with the project owner before publishing.
