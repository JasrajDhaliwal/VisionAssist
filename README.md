# VisionAssist

A desktop accessibility tool that describes a live camera feed out loud for people who are blind or visually impaired. Point a webcam (or a phone camera stream) at the world and VisionAssist tells you what's there, warns you about hazards, reads text, finds objects, and answers follow-up questions, all hands-free.

Built in Python with OpenCV, Tkinter and Claude vision models via the Anthropic API.

<!-- Add a screenshot or GIF of the app here, e.g. ![VisionAssist demo](docs/demo.gif) -->

## Features

| Mode | Shortcut | What it does |
|---|---|---|
| **Describe** | `Space` | 1–3 sentence description of the scene: people, objects, position (left / right / ahead) and rough distance |
| **Navigation** | `N` | Walking mode: reports only obstacles, steps, doors and people in your path, or "Path is clear." |
| **What Changed** | `W` | Compares the current frame with the last description and reports only what's new |
| **Auto-describe** | `A` | Describes the scene on a loop, with an adjustable minimum gap between updates |
| **Read Text** | `R` | Reads all visible text: signs, labels, menus, screens |
| **Find** | (text box) | Locates a named object ("my keys") and gives directions to reach it |
| **Ask** | `Q` or text box | Ask a question about the scene by voice or typing; follow-ups use recent descriptions and Q&A history as context |
| **Wake word** | (toggle) | Listens continuously for a keyword (default "help") and starts listening for a question when it hears it |

Hazards are prioritized in speech: **`ALERT:`** for immediate dangers (a step right ahead, something blocking your path) and **`HEADS UP:`** for things worth knowing (someone approaching, a low obstacle).

## How it works

- **Low-latency speech.** Responses are streamed from the API and split into sentences as they arrive, so the first sentence is spoken before the full answer finishes. Speech runs in a background queue and new output interrupts old output, so you always hear the latest information.
- **Prompt caching.** Each mode's system prompt is sent with `cache_control`, so repeated calls reuse the cached prompt and respond faster.
- **Model per task.** Fast scene, navigation, change and text-reading calls use Claude Haiku; questions and object finding use Claude Sonnet for better reasoning.
- **Reliable wake word.** The listener records in overlapping 2-second windows that step every 0.75 s, so the keyword isn't missed when it falls on a chunk boundary.
- **Flexible input.** Any camera index (`0`, `1`, …) or an `rtsp://` / `http://` stream URL; microphone input can be the system default or an iPhone mic via Continuity.
- **Export.** The output log can be exported from the app.

## Project structure

```
main.py          Tkinter UI, camera loop, keyboard shortcuts, speech queue
assistant.py     VisionAssistant: prompts and streaming calls to the Anthropic API
voice.py         Push-to-talk recording, wake-word listener, mic discovery
requirements.txt
```

## Getting started

**Requirements:** Python 3.10+, macOS (speech uses the built-in `say` and `afplay` commands), a webcam, and an [Anthropic API key](https://console.anthropic.com/).

```bash
git clone https://github.com/JasrajDhaliwal/VisionAssist.git
cd VisionAssist
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
echo "ANTHROPIC_API_KEY=your-key-here" > .env
python main.py
```

Press `Space` to get your first description.

## Notes

- Built as a solo project for COMP 380 at the University of the Fraser Valley.
- Voice questions and wake-word detection use Google's speech recognition through the `SpeechRecognition` library, so they need an internet connection.
