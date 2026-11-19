import sounddevice as sd
import numpy as np
import tempfile
import os
import time
import threading
import speech_recognition as sr
import scipy.io.wavfile as wavfile

SAMPLE_RATE = 16000


def list_input_devices():
    """Returns list of (index, name) for all input devices."""
    devices = sd.query_devices()
    return [
        (i, d['name'])
        for i, d in enumerate(devices)
        if d['max_input_channels'] > 0
    ]


def find_phone_mic():
    """Try to find an iPhone/phone microphone device. Returns index or None."""
    keywords = ['iphone', 'continuity', 'phone', 'facetime']
    for idx, name in list_input_devices():
        if any(kw in name.lower() for kw in keywords):
            return idx, name
    return None, None


class VoiceRecorder:
    def __init__(self):
        self.recording = False
        self.frames = []
        self.stream = None
        self.device = None  # None = system default

    def start(self):
        self.frames = []
        self.recording = True
        self.stream = sd.InputStream(
            samplerate=SAMPLE_RATE, channels=1,
            dtype='int16', callback=self._callback,
            device=self.device
        )
        self.stream.start()

    def _callback(self, indata, frames, time, status):
        if self.recording:
            self.frames.append(indata.copy())

    def stop_and_transcribe(self):
        self.recording = False
        if self.stream:
            self.stream.stop()
            self.stream.close()
            self.stream = None
        if not self.frames:
            return ""
        audio_data = np.concatenate(self.frames, axis=0)
        return _transcribe(audio_data)


class WakeWordListener:
    """Continuously listens in overlapping windows for a trigger phrase."""

    def __init__(self, callback, phrase="vision"):
        self.callback = callback
        self.phrase = phrase.lower()
        self.running = False
        self.device = None  # None = system default

    def start(self):
        if self.running:
            return
        self.running = True
        threading.Thread(target=self._loop, daemon=True).start()

    def stop(self):
        self.running = False

    def _loop(self):
        """Records in overlapping 2s windows stepping every 0.75s so
        the wake word is very unlikely to be missed at a chunk boundary."""
        buffer = np.zeros((0, 1), dtype='int16')
        window = int(2.0 * SAMPLE_RATE)
        step   = int(0.75 * SAMPLE_RATE)

        frames = []

        def cb(indata, f, t, s):
            frames.append(indata.copy())

        try:
            stream = sd.InputStream(samplerate=SAMPLE_RATE, channels=1,
                                    dtype='int16', callback=cb,
                                    device=self.device)
            stream.start()
        except Exception:
            return

        while self.running:
            time.sleep(0.75)
            if not frames:
                continue

            chunk = np.concatenate(frames, axis=0)
            frames.clear()
            buffer = np.concatenate([buffer, chunk], axis=0)

            if len(buffer) >= window:
                segment = buffer[-window:]
                buffer  = buffer[-step:]   # keep overlap
                if _contains_phrase(segment, self.phrase):
                    self.callback()
                    buffer = np.zeros((0, 1), dtype='int16')  # reset after trigger
                    time.sleep(2.0)  # debounce

        stream.stop()
        stream.close()


def _contains_phrase(audio_data, phrase):
    """Return True if phrase appears in any Google Speech alternative."""
    recognizer = sr.Recognizer()
    recognizer.energy_threshold = 200
    recognizer.dynamic_energy_threshold = True
    fname = None
    try:
        with tempfile.NamedTemporaryFile(suffix='.wav', delete=False) as f:
            wavfile.write(f.name, SAMPLE_RATE, audio_data)
            fname = f.name
        with sr.AudioFile(fname) as source:
            audio = recognizer.record(source)
        result = recognizer.recognize_google(audio, show_all=True)
        if not result:
            return False
        for alt in result.get('alternative', []):
            if phrase in alt.get('transcript', '').lower():
                return True
        return False
    except Exception:
        return False
    finally:
        if fname:
            try:
                os.unlink(fname)
            except Exception:
                pass


def _transcribe(audio_data):
    recognizer = sr.Recognizer()
    fname = None
    try:
        with tempfile.NamedTemporaryFile(suffix='.wav', delete=False) as f:
            wavfile.write(f.name, SAMPLE_RATE, audio_data)
            fname = f.name
        with sr.AudioFile(fname) as source:
            audio = recognizer.record(source)
        return recognizer.recognize_google(audio)
    except sr.UnknownValueError:
        return ""
    except Exception:
        return ""
    finally:
        if fname:
            try:
                os.unlink(fname)
            except Exception:
                pass
