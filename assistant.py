import anthropic
import base64
import cv2
import os
from dotenv import load_dotenv

load_dotenv()

# ── System prompts (cached via cache_control to reduce latency) ───────────────

_SCENE_SYSTEM = (
    "You are a real-time vision assistant for a visually impaired person. "
    "Describe what you see in 1-3 concise sentences. "
    "For people: count them, note if approaching or stationary, rough distance. "
    "For objects: note position (left, right, center, ahead) and distance "
    "('right in front' <1 m, 'nearby' 1-3 m, 'in the distance' >3 m). "
    "Use these prefixes when relevant:\n"
    "  'ALERT:' — immediate hazard (moving vehicle, step/drop right ahead, person blocking path)\n"
    "  'HEADS UP:' — worth knowing (approaching person, low obstacle, cluttered floor)\n"
    "No prefix for safe, neutral scenes. Never describe what is not there."
)

_NAV_SYSTEM = (
    "You are a navigation guide for a visually impaired person who is walking. "
    "Report ONLY what affects safe movement in 1-2 sentences. "
    "Focus on: obstacles, steps or curbs, doors, narrow passages, people or objects in the path. "
    "Use directions: 'ahead', 'left', 'right', 'above'. "
    "If clear, say 'Path is clear.' "
    "Prefix 'ALERT:' for immediate hazards (step right ahead, obstacle blocking path). "
    "Prefix 'HEADS UP:' for cautions (person approaching, slight incline, door ahead). "
    "Never describe flooring material, decor, or irrelevant background."
)

_DELTA_SYSTEM = (
    "You are a vision assistant for a visually impaired person. "
    "You will receive the previous scene description and the current image. "
    "Describe any noticeable differences — new or removed people or objects, "
    "movement, lighting changes, or anything the person should know about. "
    "Use 'ALERT:' for new hazards, 'HEADS UP:' for notable changes. "
    "Only respond with exactly 'No change.' if the scene looks completely identical to the description. "
    "Keep to 1-2 sentences."
)

_QA_SYSTEM = (
    "You are a vision assistant helping a visually impaired person understand their environment. "
    "Answer questions in 1-3 sentences based on the camera image provided. "
    "Be specific about positions (left, right, center, near, far). "
    "If you cannot determine something from the image, say so clearly."
)

_READ_SYSTEM = (
    "Read ALL visible text in this image exactly as it appears — "
    "signs, labels, menus, documents, screens, handwriting. "
    "Group by source (e.g. 'Sign reads: Exit', 'Screen shows: 3:42 PM'). "
    "If no readable text is visible, say 'No text found.'"
)

_FIND_SYSTEM = (
    "You are helping a visually impaired person locate a specific object. "
    "If the object is visible: describe exactly where it is using directions "
    "(left / right / center / ahead) and distance (right in front / nearby / far away), "
    "then give brief guidance to reach it. "
    "If not visible: say 'Not visible.' and optionally suggest where it might be."
)


class VisionAssistant:
    def __init__(self):
        self.client = anthropic.Anthropic(api_key=os.getenv("ANTHROPIC_API_KEY"))
        self.last_description = ""

    # ── Image helpers ─────────────────────────────────────────────────────────

    def _frame_to_base64(self, frame, quality=82):
        _, buffer = cv2.imencode('.jpg', frame, [cv2.IMWRITE_JPEG_QUALITY, quality])
        return base64.standard_b64encode(buffer).decode('utf-8')

    def _image_content(self, frame, quality=82):
        return {
            "type": "image",
            "source": {
                "type": "base64",
                "media_type": "image/jpeg",
                "data": self._frame_to_base64(frame, quality)
            }
        }

    # ── Streaming core ────────────────────────────────────────────────────────

    def _stream_sentences(self, frame, user_text, system, model="claude-haiku-4-5-20251001",
                          max_tokens=200, quality=82):
        """Stream the model response, yielding one sentence at a time.

        The system prompt is sent with cache_control so repeated calls with the
        same prompt are served from Anthropic's cache — dramatically lower latency.
        """
        buffer = ""
        with self.client.messages.stream(
            model=model,
            max_tokens=max_tokens,
            system=[{"type": "text", "text": system, "cache_control": {"type": "ephemeral"}}],
            messages=[{
                "role": "user",
                "content": [
                    self._image_content(frame, quality),
                    {"type": "text", "text": user_text}
                ]
            }]
        ) as stream:
            for chunk in stream.text_stream:
                buffer += chunk
                while True:
                    end = -1
                    for sep in ['. ', '! ', '? ', '.\n', '!\n', '?\n']:
                        idx = buffer.find(sep)
                        if idx != -1 and (end == -1 or idx < end):
                            end = idx + len(sep)
                    if end == -1:
                        break
                    sentence = buffer[:end].strip()
                    buffer = buffer[end:]
                    if sentence:
                        yield sentence
            if buffer.strip():
                yield buffer.strip()

    # ── Scene description ─────────────────────────────────────────────────────

    def describe_scene_stream(self, frame, nav_mode=False):
        """Stream a full scene or navigation description, sentence by sentence."""
        system = _NAV_SYSTEM if nav_mode else _SCENE_SYSTEM
        yield from self._stream_sentences(frame, "Describe this scene.", system)

    def describe_changes_stream(self, frame, prev_description):
        """Stream only what changed since the last description.
        Yields 'No change.' if the scene is essentially the same."""
        if not prev_description:
            yield from self.describe_scene_stream(frame)
            return
        user_text = f"Previous description: {prev_description}\n\nWhat has changed?"
        yield from self._stream_sentences(frame, user_text, _DELTA_SYSTEM)

    # ── Q&A (streaming) ───────────────────────────────────────────────────────

    def answer_question_stream(self, frame, question, memory=None, qa_history=None):
        """Stream an answer to a question about the current scene.

        Passes recent scene descriptions and Q&A history as context so the
        assistant can handle follow-up questions naturally.
        """
        parts = []
        if memory:
            parts.append("Recent scene context:\n" + "\n".join(f"- {d}" for d in memory[-3:]))
        if qa_history:
            qa_text = "\n".join(
                f"Q: {p['q']}\nA: {p['a']}" for p in qa_history[-3:]
            )
            parts.append(f"Previous Q&A:\n{qa_text}")

        context = "\n\n".join(parts)
        user_text = f"{context}\n\nQuestion: {question}" if context else f"Question: {question}"

        yield from self._stream_sentences(
            frame, user_text, _QA_SYSTEM,
            model="claude-sonnet-4-6", max_tokens=200
        )

    # ── OCR ───────────────────────────────────────────────────────────────────

    def read_text(self, frame):
        """Extract and return all visible text in the frame."""
        response = self.client.messages.create(
            model="claude-haiku-4-5-20251001",
            max_tokens=400,
            system=[{"type": "text", "text": _READ_SYSTEM, "cache_control": {"type": "ephemeral"}}],
            messages=[{
                "role": "user",
                "content": [
                    self._image_content(frame, quality=90),   # higher quality for OCR
                    {"type": "text", "text": "Read all visible text."}
                ]
            }]
        )
        return response.content[0].text

    # ── Object finding ────────────────────────────────────────────────────────

    def find_object(self, frame, object_name):
        """Locate a named object in the frame and describe how to reach it."""
        response = self.client.messages.create(
            model="claude-sonnet-4-6",
            max_tokens=150,
            system=[{"type": "text", "text": _FIND_SYSTEM, "cache_control": {"type": "ephemeral"}}],
            messages=[{
                "role": "user",
                "content": [
                    self._image_content(frame),
                    {"type": "text", "text": f"Find: {object_name}"}
                ]
            }]
        )
        return response.content[0].text

    # ── Urgency ───────────────────────────────────────────────────────────────

    def urgency(self, description):
        if description.startswith("ALERT:"):
            return "alert"
        if description.startswith("HEADS UP:"):
            return "headsup"
        return ""
