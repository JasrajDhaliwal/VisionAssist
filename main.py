import tkinter as tk
from tkinter import scrolledtext, filedialog
import threading
import queue as _q
import cv2
import numpy as np
import subprocess
import os
import time
from PIL import Image, ImageTk
from dotenv import load_dotenv
from assistant import VisionAssistant
from voice import VoiceRecorder, WakeWordListener, list_input_devices, find_phone_mic

load_dotenv()

# ── Speech queue ─────────────────────────────────────────────────────────────
_speech_queue = _q.Queue()
_say_proc = None


def _speech_worker():
    global _say_proc
    while True:
        text = _speech_queue.get()
        if text is None:
            break
        _say_proc = subprocess.Popen(['say', '-r', '200', text])
        _say_proc.wait()
        _speech_queue.task_done()


threading.Thread(target=_speech_worker, daemon=True).start()


def speak(text, interrupt=True):
    global _say_proc
    if interrupt:
        while not _speech_queue.empty():
            try:
                _speech_queue.get_nowait()
                _speech_queue.task_done()
            except Exception:
                pass
        if _say_proc and _say_proc.poll() is None:
            _say_proc.terminate()
    if text:
        _speech_queue.put(text)


def is_speaking():
    return ((_say_proc is not None and _say_proc.poll() is None)
            or not _speech_queue.empty())


def audio_cue():
    subprocess.Popen(['afplay', '/System/Library/Sounds/Tink.aiff'])


class _MacButton(tk.Label):
    def __init__(self, master, command=None, activebackground=None,
                 activeforeground=None, **kw):
        self._command = command
        self._bg = kw.get('bg', '#333')
        self._fg = kw.get('fg', '#ffffff')
        self._disabled = False
        kw.pop('relief', None)
        super().__init__(master, **kw)
        self.bind('<Button-1>', self._on_click)
        self.bind('<Enter>',    self._on_enter)
        self.bind('<Leave>',    self._on_leave)

    def _on_click(self, _=None):
        if not self._disabled and self._command:
            self._command()

    def _on_enter(self, _=None):
        if not self._disabled:
            try:
                r, g, b = (int(self._bg[i:i+2], 16) for i in (1, 3, 5))
                darker = '#{:02x}{:02x}{:02x}'.format(
                    max(0, r - 25), max(0, g - 25), max(0, b - 25))
                super().configure(bg=darker)
            except Exception:
                pass

    def _on_leave(self, _=None):
        super().configure(bg=self._bg)

    def configure(self, cnf=None, **kw):
        if 'bg' in kw:
            self._bg = kw['bg']
        if 'fg' in kw:
            self._fg = kw['fg']
        if 'state' in kw:
            self._disabled = (kw.pop('state') == 'disabled')
            kw['fg'] = '#555' if self._disabled else self._fg
        super().configure(**(cnf or {}), **kw)

    config = configure


# ── App ───────────────────────────────────────────────────────────────────────

class VisionAssistApp:
    def __init__(self, root):
        self.root = root
        self.root.title("VisionAssist")
        self.root.configure(bg='#1a1a1a')
        self.root.geometry('1040x700')
        self.root.resizable(True, True)

        self.assistant = VisionAssistant()
        self.recorder = VoiceRecorder()
        self.wake_listener = WakeWordListener(callback=self._wake_triggered, phrase='help')

        self.running = True
        self.current_frame = None
        self.cap = None
        self.cam_source = tk.StringVar(value="0")

        self.describing = False
        self.auto_describe = False
        self.nav_mode = False
        self.delta_mode = False
        self.voice_active = False
        self.wake_enabled = False

        self.interval_var = tk.IntVar(value=3)
        self.history = []
        self.description_memory = []
        self.qa_history = []
        self._last_described_frame = None
        self._resize_job = None

        self._init_fonts()
        self._build_ui()
        self._bind_shortcuts()
        self._connect_camera("0")
        self._check_env()
        self.root.bind('<Configure>', self._on_resize)
        self.root.bind('<Map>', lambda e: self.root.after(500, self._apply_scale_deferred))

    # ── Fonts ─────────────────────────────────────────────────────────────────

    def _init_fonts(self):
        from tkinter import font as tkfont
        self._fonts = {
            'title':  tkfont.Font(family='Helvetica', size=18, weight='bold'),
            'btn_lg': tkfont.Font(family='Helvetica', size=11, weight='bold'),
            'btn':    tkfont.Font(family='Helvetica', size=10, weight='bold'),
            'normal': tkfont.Font(family='Helvetica', size=10),
            'small':  tkfont.Font(family='Helvetica', size=9),
            'tiny':   tkfont.Font(family='Helvetica', size=8),
            'hint':   tkfont.Font(family='Helvetica', size=7),
        }
        self._base_font_sizes = {
            'title': 18, 'btn_lg': 11, 'btn': 10,
            'normal': 10, 'small': 9, 'tiny': 8, 'hint': 7,
        }

    # ── UI ────────────────────────────────────────────────────────────────────

    def _build_ui(self):
        f = self._fonts

        # Header
        header = tk.Frame(self.root, bg='#111', pady=8)
        header.pack(fill='x')
        tk.Label(header, text="VisionAssist", font=f['title'],
                 bg='#111', fg='#4CAF50').pack(side='left', padx=15)
        tk.Label(header, text="Accessibility tool for the visually impaired",
                 font=f['normal'], bg='#111', fg='#666').pack(side='left')

        # Camera bar — two pack sub-rows so entry expands and nothing overflows
        cam_bar = tk.Frame(self.root, bg='#0d0d0d', pady=5)
        cam_bar.pack(fill='x')

        # Row 1: [Camera:] [entry, expands] [Connect] [Scan]
        cam_row1 = tk.Frame(cam_bar, bg='#0d0d0d')
        cam_row1.pack(fill='x', padx=12, pady=(0, 2))
        tk.Label(cam_row1, text="Camera:", bg='#0d0d0d', fg='#777',
                 font=f['small']).pack(side='left', padx=(0, 4))
        _MacButton(cam_row1, text="Scan", command=self._scan_cameras,
                   bg='#3a3a3a', fg='#cccccc', font=f['small'],
                   padx=6, cursor='hand2').pack(side='right')
        _MacButton(cam_row1, text="Connect", command=self._reconnect_camera,
                   bg='#4a4a4a', fg='#ffffff', font=f['small'],
                   padx=8, cursor='hand2').pack(side='right', padx=(0, 6))
        cam_entry = tk.Entry(cam_row1, textvariable=self.cam_source,
                             bg='#2a2a2a', fg='white', insertbackground='white',
                             font=f['small'], relief='flat')
        cam_entry.pack(side='left', fill='x', expand=True, ipady=3)
        cam_entry.bind('<Return>', lambda _: self._reconnect_camera())

        # Row 2: [URL hint] [Mic: device]
        cam_row2 = tk.Frame(cam_bar, bg='#0d0d0d')
        cam_row2.pack(fill='x', padx=12)
        tk.Label(cam_row2, text="0, 1, 2 ... or rtsp:// / http:// URL",
                 bg='#0d0d0d', fg='#3a3a3a', font=f['tiny']).pack(side='left')
        self.mic_label = tk.Label(cam_row2, text="System default",
                                  bg='#0d0d0d', fg='#555', font=f['tiny'])
        self.mic_label.pack(side='right')
        tk.Label(cam_row2, text="Mic:", bg='#0d0d0d', fg='#777',
                 font=f['tiny']).pack(side='right', padx=(0, 4))

        # Body
        body = tk.Frame(self.root, bg='#1a1a1a')
        body.pack(fill='both', expand=True, padx=12, pady=8)

        # Left — camera feed
        left = tk.Frame(body, bg='#000', bd=1, relief='solid')
        left.pack(side='left', fill='both', expand=True)
        self.video_label = tk.Label(left, bg='#000', text="Initializing camera...",
                                    fg='#444', font=f['normal'])
        self.video_label.pack(fill='both', expand=True)

        # Right — controls (natural width so text is never clipped)
        self.right_frame = tk.Frame(body, bg='#1a1a1a')
        self.right_frame.pack(side='right', fill='y', padx=(10, 0))

        # Describe + Nav
        btn_row = tk.Frame(self.right_frame, bg='#1a1a1a')
        btn_row.pack(fill='x', pady=(0, 4))
        self.describe_btn = _MacButton(
            btn_row, text="Describe  [Space]",
            command=self._describe_now,
            bg='#4CAF50', fg='white',
            font=f['btn_lg'], pady=9, cursor='hand2'
        )
        self.describe_btn.pack(side='left', fill='x', expand=True, padx=(0, 4))
        self.nav_btn = _MacButton(
            btn_row, text="Nav  [N]",
            command=self._toggle_nav,
            bg='#4a4a4a', fg='#ffffff',
            font=f['btn'], pady=9, cursor='hand2'
        )
        self.nav_btn.pack(side='right')

        # What Changed (delta mode)
        self.delta_btn = _MacButton(
            self.right_frame, text="What Changed  [W]",
            command=self._toggle_delta,
            bg='#4a4a4a', fg='#ffffff',
            font=f['btn'], pady=7, cursor='hand2'
        )
        self.delta_btn.pack(fill='x', pady=(0, 6))

        # Separator
        tk.Frame(self.right_frame, bg='#2a2a2a', height=1).pack(fill='x', pady=(0, 6))

        # Auto-describe + interval
        auto_row = tk.Frame(self.right_frame, bg='#1a1a1a')
        auto_row.pack(fill='x', pady=(0, 2))
        self.auto_var = tk.BooleanVar()
        tk.Checkbutton(auto_row, text="Auto-describe  [A]",
                       variable=self.auto_var, command=self._toggle_auto,
                       bg='#1a1a1a', fg='#ccc', selectcolor='#333',
                       activebackground='#1a1a1a', font=f['normal']
                       ).pack(side='left')

        slider_row = tk.Frame(self.right_frame, bg='#1a1a1a')
        slider_row.pack(fill='x', pady=(0, 8))
        tk.Label(slider_row, text="Min gap:", bg='#1a1a1a', fg='#555',
                 font=f['tiny']).pack(side='left')
        self.interval_scale = tk.Scale(slider_row, from_=2, to=15, orient='horizontal',
                 variable=self.interval_var, bg='#1a1a1a', fg='#ccc',
                 highlightthickness=0, troughcolor='#333', length=160,
                 font=f['tiny'])
        self.interval_scale.pack(side='left')
        tk.Label(slider_row, text="sec", bg='#1a1a1a', fg='#555',
                 font=f['tiny']).pack(side='left')

        # Read Text
        _MacButton(self.right_frame, text="Read Text  [R]",
                   command=self._read_text,
                   bg='#7B1FA2', fg='white',
                   font=f['btn'], pady=6, cursor='hand2'
                   ).pack(fill='x', pady=(0, 4))

        # Find object
        find_row = tk.Frame(self.right_frame, bg='#1a1a1a')
        find_row.pack(fill='x', pady=(0, 8))
        self.find_entry = tk.Entry(find_row, bg='#2a2a2a', fg='#888',
                                   insertbackground='white',
                                   font=f['normal'], relief='flat')
        self.find_entry.insert(0, 'Find object...')
        self.find_entry.bind('<FocusIn>',  lambda _: self._clear_find())
        self.find_entry.bind('<FocusOut>', lambda _: self._restore_find())
        self.find_entry.bind('<Return>', lambda _: self._find_object())
        self.find_entry.pack(side='left', fill='x', expand=True, ipady=5, padx=(0, 4))
        _MacButton(find_row, text="Find", command=self._find_object,
                   bg='#FF6F00', fg='white',
                   font=f['btn'], padx=8, cursor='hand2'
                   ).pack(side='right')

        # Q&A
        tk.Label(self.right_frame, text="Ask about the scene:", bg='#1a1a1a',
                 fg='#bbb', font=f['btn']).pack(anchor='w')
        qa_row = tk.Frame(self.right_frame, bg='#1a1a1a')
        qa_row.pack(fill='x', pady=(3, 4))
        self.question_entry = tk.Entry(qa_row, bg='#2a2a2a', fg='#888',
                                       insertbackground='white',
                                       font=f['normal'], relief='flat')
        self.question_entry.insert(0, 'e.g. Is there anyone nearby?')
        self.question_entry.bind('<FocusIn>',  self._clear_placeholder)
        self.question_entry.bind('<FocusOut>', self._restore_placeholder)
        self.question_entry.bind('<Return>', self._ask_question)
        self.question_entry.pack(side='left', fill='x', expand=True, ipady=6, padx=(0, 4))
        _MacButton(qa_row, text="Ask", command=self._ask_question,
                   bg='#2196F3', fg='white',
                   font=f['btn'], padx=8, cursor='hand2'
                   ).pack(side='right')

        # Voice + Wake word
        voice_row = tk.Frame(self.right_frame, bg='#1a1a1a')
        voice_row.pack(fill='x', pady=(0, 4))
        self.mic_btn = _MacButton(
            voice_row, text="Speak Question  [Q]",
            command=self._toggle_voice,
            bg='#4a4a4a', fg='#ffffff',
            font=f['normal'], pady=6, cursor='hand2'
        )
        self.mic_btn.pack(side='left', fill='x', expand=True, padx=(0, 4))
        self.wake_btn = _MacButton(
            voice_row, text='Wake Word',
            command=self._toggle_wake,
            bg='#4a4a4a', fg='#ffffff',
            font=f['small'], pady=6, cursor='hand2'
        )
        self.wake_btn.pack(side='right')

        wake_row = tk.Frame(self.right_frame, bg='#1a1a1a')
        wake_row.pack(fill='x', pady=(0, 6))
        tk.Label(wake_row, text='Keyword:', bg='#1a1a1a', fg='#555',
                 font=f['tiny']).pack(side='left')
        self.wake_word_var = tk.StringVar(value='help')
        wake_entry = tk.Entry(wake_row, textvariable=self.wake_word_var,
                              bg='#2a2a2a', fg='white', insertbackground='white',
                              font=f['small'], relief='flat', width=10)
        wake_entry.pack(side='left', padx=(4, 6), ipady=3)
        _MacButton(wake_row, text='Set', command=self._set_wake_word,
                   bg='#4a4a4a', fg='#ffffff',
                   font=f['tiny'], padx=6, cursor='hand2').pack(side='left')

        tk.Label(self.right_frame,
                 text='Space=describe  W=changes  Q=voice  A=auto  R=read  N=nav',
                 bg='#1a1a1a', fg='#333', font=f['hint']
                 ).pack(anchor='w', pady=(0, 4))

        # Output log
        log_header = tk.Frame(self.right_frame, bg='#1a1a1a')
        log_header.pack(fill='x')
        tk.Label(log_header, text="Output:", bg='#1a1a1a',
                 fg='#bbb', font=f['btn']).pack(side='left')
        _MacButton(log_header, text="Export", command=self._export_log,
                   bg='#3a3a3a', fg='#cccccc',
                   font=f['tiny'], cursor='hand2').pack(side='right')

        self.output = scrolledtext.ScrolledText(
            self.right_frame, bg='#111', fg='#ddd', font=f['normal'],
            relief='flat', wrap='word', state='disabled', pady=6, padx=6
        )
        self.output.pack(fill='both', expand=True, pady=(3, 0))
        self.output.tag_config('alert',    foreground='#FF5722', font=('Helvetica', 10, 'bold'))
        self.output.tag_config('headsup',  foreground='#FFC107', font=('Helvetica', 10, 'bold'))
        self.output.tag_config('info',     foreground='#4CAF50')
        self.output.tag_config('question', foreground='#64B5F6', font=('Helvetica', 10, 'italic'))
        self.output.tag_config('ocr',      foreground='#CE93D8')
        self.output.tag_config('find',     foreground='#FFB74D')
        self.output.tag_config('nav',      foreground='#80DEEA')
        self.output.tag_config('delta',    foreground='#A5D6A7', font=('Helvetica', 10, 'italic'))
        self.output.tag_config('warn',     foreground='#FF9800')

        # Status bar
        self.status_var = tk.StringVar(value="Ready")
        tk.Label(self.root, textvariable=self.status_var, anchor='w',
                 bg='#0d0d0d', fg='#555', font=f['small'],
                 padx=12).pack(fill='x', side='bottom')

    def _on_resize(self, event):
        if event.widget is not self.root:
            return
        if self._resize_job:
            self.root.after_cancel(self._resize_job)
        # 400ms delay lets macOS finish the fullscreen animation before we measure
        self._resize_job = self.root.after(400, self._apply_scale_deferred)

    def _apply_scale_deferred(self):
        self._resize_job = None
        # Force tkinter to finish pending geometry work, then read the real size
        self.root.update_idletasks()
        width = self.root.winfo_width()
        if width < 100:
            return
        self._apply_scale(width)

    def _apply_scale(self, width):
        scale = max(0.75, min(1.9, width / 1040))
        for name, base in self._base_font_sizes.items():
            self._fonts[name].configure(size=max(7, int(base * scale)))
        self.interval_scale.configure(length=int(160 * scale))
        # Force layout recalculation after font changes
        self.root.update_idletasks()
        s = max(7, int(10 * scale))
        self.output.tag_config('alert',    font=('Helvetica', s, 'bold'))
        self.output.tag_config('headsup',  font=('Helvetica', s, 'bold'))
        self.output.tag_config('question', font=('Helvetica', s, 'italic'))
        self.output.tag_config('delta',    font=('Helvetica', s, 'italic'))

    # ── Placeholders ──────────────────────────────────────────────────────────

    def _clear_placeholder(self, _=None):
        if self.question_entry.get() == 'e.g. Is there anyone nearby?':
            self.question_entry.delete(0, 'end')
            self.question_entry.config(fg='white')

    def _restore_placeholder(self, _=None):
        if not self.question_entry.get():
            self.question_entry.insert(0, 'e.g. Is there anyone nearby?')
            self.question_entry.config(fg='#888')

    def _clear_find(self):
        if self.find_entry.get() == 'Find object...':
            self.find_entry.delete(0, 'end')
            self.find_entry.config(fg='white')

    def _restore_find(self):
        if not self.find_entry.get():
            self.find_entry.insert(0, 'Find object...')
            self.find_entry.config(fg='#888')

    # ── Thread-safe UI ────────────────────────────────────────────────────────

    def _log(self, text, tag=''):
        ts = time.strftime('%H:%M:%S')
        self.history.append((ts, text))
        def _do():
            self.output.configure(state='normal')
            self.output.insert('end', f"[{ts}] {text}\n\n", tag)
            self.output.see('end')
            self.output.configure(state='disabled')
        self.root.after(0, _do)

    def _set_status(self, text):
        self.root.after(0, lambda: self.status_var.set(text))

    def _set_describe_btn(self, enabled):
        self.root.after(0, lambda: self.describe_btn.configure(
            state='normal' if enabled else 'disabled'))

    # ── Camera ────────────────────────────────────────────────────────────────

    def _connect_camera(self, source):
        self.assistant.last_description = ""
        self._last_described_frame = None
        if self.cap:
            self.cap.release()
        try:
            source = int(source)
        except ValueError:
            pass
        self.cap = cv2.VideoCapture(source)
        if self.cap.isOpened():
            self._set_status(f"Camera connected: {source}")
            self._start_video_feed()
            self._auto_select_mic(source)
        else:
            self._log(f"Could not open camera source: {source}", 'warn')
            self._set_status("Camera error — check source")

    def _auto_select_mic(self, cam_source):
        if cam_source == 0:
            self._set_mic_device(None, "System default")
        else:
            idx, name = find_phone_mic()
            if idx is not None:
                self._set_mic_device(idx, name)
                self._log(f"Mic switched to: {name}", 'info')
            else:
                devices = list_input_devices()
                device_list = ", ".join(f"{i}={n}" for i, n in devices)
                self._log(f"Phone mic not found. Available inputs: {device_list}", 'warn')

    def _set_mic_device(self, device_index, name):
        self.recorder.device = device_index
        self.wake_listener.device = device_index
        self.root.after(0, lambda: self.mic_label.configure(
            text=name[:30] + ('...' if len(name) > 30 else ''),
            fg='#4CAF50' if device_index is not None else '#555'
        ))

    def _reconnect_camera(self):
        source = self.cam_source.get().strip()
        if source:
            self._connect_camera(source)

    def _scan_cameras(self):
        threading.Thread(target=self._do_scan_cameras, daemon=True).start()

    def _do_scan_cameras(self):
        self._set_status("Scanning for cameras...")
        found = []

        # Probe indices silently (suppress OpenCV's stderr spam for missing devices)
        devnull = open(os.devnull, 'w')
        old_stderr = os.dup(2)
        os.dup2(devnull.fileno(), 2)
        try:
            for i in range(10):
                cap = cv2.VideoCapture(i, cv2.CAP_AVFOUNDATION)
                if cap.isOpened():
                    ret, _ = cap.read()
                    if ret:
                        found.append(str(i))
                cap.release()
        finally:
            os.dup2(old_stderr, 2)
            os.close(old_stderr)
            devnull.close()

        # Also ask macOS what cameras it sees
        os_cameras = self._list_os_cameras()

        if found:
            msg = f"Cameras found at index: {', '.join(found)}"
            self._log(msg, 'info')
            speak(msg, interrupt=True)
        else:
            self._log("No usable camera found via index scan.", 'warn')

        if os_cameras:
            self._log(f"macOS sees: {', '.join(os_cameras)}", 'info')
            if not found:
                self._log(
                    "Your phone camera may appear at a higher index or need a URL. "
                    "iPhone: enable Continuity Camera (Settings → General → AirPlay & Handoff) and connect USB or same Wi-Fi + Bluetooth. "
                    "Android: use DroidCam and enter its URL (e.g. http://192.168.x.x:4747/video).",
                    'warn'
                )

        self._set_status("Ready")

    def _list_os_cameras(self):
        """Use macOS system_profiler to list camera device names."""
        try:
            result = subprocess.run(
                ['system_profiler', 'SPCameraDataType'],
                capture_output=True, text=True, timeout=5
            )
            names = []
            for line in result.stdout.splitlines():
                line = line.strip()
                if line and not line.startswith(('Camera', 'Unique', 'Model',
                                                  'Manufacturer', 'Suspend')):
                    if ':' not in line:
                        names.append(line)
            return names
        except Exception:
            return []

    def _start_video_feed(self):
        def update():
            if not self.running or not self.cap:
                return
            ret, frame = self.cap.read()
            if ret:
                self.current_frame = frame
                rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                img = Image.fromarray(rgb)
                img.thumbnail((680, 540), Image.LANCZOS)
                photo = ImageTk.PhotoImage(img)
                self.video_label.configure(image=photo, text='')
                self.video_label.image = photo
            else:
                self.video_label.configure(image='', text='No signal', fg='#444')
            self.root.after(33, update)
        update()

    # ── Scene change detection ────────────────────────────────────────────────

    def _scene_changed(self, frame, threshold=0.14):
        small = cv2.resize(frame, (32, 32))
        gray  = cv2.cvtColor(small, cv2.COLOR_BGR2GRAY).astype(float)
        if self._last_described_frame is None:
            return True
        prev = cv2.resize(self._last_described_frame, (32, 32))
        prev = cv2.cvtColor(prev, cv2.COLOR_BGR2GRAY).astype(float)
        diff = np.mean(np.abs(gray - prev)) / 255.0
        return diff > threshold

    # ── Actions ───────────────────────────────────────────────────────────────

    def _check_env(self):
        if not os.getenv("ANTHROPIC_API_KEY"):
            self._log("No API key — add ANTHROPIC_API_KEY to .env", 'warn')

    def _describe_now(self):
        threading.Thread(target=self._do_describe, args=(True,), daemon=True).start()

    def _do_describe(self, manual=False):
        if self.describing or self.current_frame is None:
            return
        self.describing = True
        self._set_describe_btn(False)
        self._set_status("Analyzing...")
        audio_cue()

        try:
            frame = self.current_frame.copy()
            full_text = ""
            first_sentence = True

            # Use delta only for small changes — big scene shifts (panning to a new
            # area) get a fresh full description so stale context doesn't confuse the AI
            big_change = self._scene_changed(frame, threshold=0.30)
            if self.delta_mode and self.assistant.last_description and not big_change:
                stream = self.assistant.describe_changes_stream(
                    frame, self.assistant.last_description)
            else:
                stream = self.assistant.describe_scene_stream(frame, nav_mode=self.nav_mode)

            for sentence in stream:
                full_text += sentence + " "
                if sentence.strip().lower() not in ('no change.', 'no change'):
                    speak(sentence, interrupt=first_sentence)
                    first_sentence = False

            full_text = full_text.strip()
            if not full_text:
                return

            if full_text.lower() in ('no change.', 'no change'):
                self._set_status("No change detected")
                return

            level = self.assistant.urgency(full_text)
            if level:
                tag = level
            elif self.delta_mode:
                tag = 'delta'
            elif self.nav_mode:
                tag = 'nav'
            else:
                tag = ''

            self._log(full_text, tag)

            self.description_memory.append(full_text)
            if len(self.description_memory) > 4:
                self.description_memory.pop(0)
            self.assistant.last_description = full_text
            self._last_described_frame = frame

        except Exception as e:
            self._log(f"Error: {e}", 'warn')
        finally:
            self.describing = False
            self._set_describe_btn(True)
            self._set_status("Ready")

    def _toggle_auto(self):
        self.auto_describe = self.auto_var.get()
        if self.auto_describe:
            self._log("Auto-describe on.", 'info')
            threading.Thread(target=self._scene_monitor, daemon=True).start()
        else:
            self._log("Auto-describe off.", 'info')

    def _scene_monitor(self):
        last_described = 0.0
        while self.running and self.auto_describe:
            time.sleep(1.0)
            if self.current_frame is None or self.describing:
                continue
            now = time.time()
            if now - last_described < self.interval_var.get():
                continue
            if self._scene_changed(self.current_frame):
                last_described = now
                self._do_describe(manual=False)

    def _toggle_nav(self):
        self.nav_mode = not self.nav_mode
        if self.nav_mode:
            self.nav_btn.configure(bg='#00838F', fg='white')
            self._log("Navigation mode on.", 'nav')
            speak("Navigation mode on.", interrupt=True)
        else:
            self.nav_btn.configure(bg='#4a4a4a', fg='#ffffff')
            self._log("Navigation mode off.", 'info')

    def _toggle_delta(self):
        self.delta_mode = not self.delta_mode
        if self.delta_mode:
            self.delta_btn.configure(bg='#2E7D32', fg='white')
            self._log("Scene change mode on — will report only what changes.", 'info')
            speak("Scene change mode on.", interrupt=True)
        else:
            self.delta_btn.configure(bg='#4a4a4a', fg='#ffffff')
            self._log("Scene change mode off.", 'info')

    def _read_text(self):
        threading.Thread(target=self._do_read_text, daemon=True).start()

    def _do_read_text(self):
        if self.current_frame is None:
            return
        self._set_status("Reading text...")
        audio_cue()
        try:
            result = self.assistant.read_text(self.current_frame.copy())
            self._log(f"Text: {result}", 'ocr')
            speak(result, interrupt=True)
        except Exception as e:
            self._log(f"Error: {e}", 'warn')
        finally:
            self._set_status("Ready")

    def _find_object(self):
        obj = self.find_entry.get().strip()
        if not obj or obj == 'Find object...' or self.current_frame is None:
            return
        threading.Thread(target=self._do_find_object, args=(obj,), daemon=True).start()

    def _do_find_object(self, obj):
        self._set_status(f"Looking for '{obj}'...")
        audio_cue()
        try:
            result = self.assistant.find_object(self.current_frame.copy(), obj)
            self._log(f"Find '{obj}': {result}", 'find')
            speak(result, interrupt=True)
        except Exception as e:
            self._log(f"Error: {e}", 'warn')
        finally:
            self._set_status("Ready")

    def _ask_question(self, _=None):
        question = self.question_entry.get().strip()
        if not question or question == 'e.g. Is there anyone nearby?':
            return
        if self.current_frame is None:
            return
        self.question_entry.delete(0, 'end')
        self._log(f"Q: {question}", 'question')
        threading.Thread(target=self._do_ask, args=(question,), daemon=True).start()

    def _do_ask(self, question):
        self._set_status("Thinking...")
        audio_cue()
        try:
            frame = self.current_frame.copy()
            full_answer = ""
            first_sentence = True

            for sentence in self.assistant.answer_question_stream(
                frame, question,
                memory=self.description_memory,
                qa_history=self.qa_history
            ):
                full_answer += sentence + " "
                speak(sentence, interrupt=first_sentence)
                first_sentence = False

            full_answer = full_answer.strip()
            if full_answer:
                self._log(f"A: {full_answer}")
                self.qa_history.append({"q": question, "a": full_answer})
                if len(self.qa_history) > 6:
                    self.qa_history.pop(0)
        except Exception as e:
            self._log(f"Error: {e}", 'warn')
        finally:
            self._set_status("Ready")

    def _export_log(self):
        if not self.history:
            return
        path = filedialog.asksaveasfilename(
            defaultextension='.txt',
            filetypes=[('Text files', '*.txt')],
            initialfile=f"visionassist_{time.strftime('%Y%m%d_%H%M%S')}.txt"
        )
        if path:
            with open(path, 'w') as f:
                f.write("VisionAssist Session Log\n")
                f.write(f"Date: {time.strftime('%Y-%m-%d')}\n\n")
                for ts, text in self.history:
                    f.write(f"[{ts}] {text}\n\n")
            self._set_status(f"Saved: {os.path.basename(path)}")

    # ── Keyboard shortcuts ────────────────────────────────────────────────────

    def _bind_shortcuts(self):
        self.root.bind('<space>', lambda e: self._describe_now()
                       if not isinstance(e.widget, tk.Entry) else None)
        self.root.bind('q', lambda e: self._shortcut_voice(e))
        self.root.bind('a', lambda e: self._shortcut_auto(e))
        self.root.bind('r', lambda e: self._read_text()
                       if not isinstance(e.widget, tk.Entry) else None)
        self.root.bind('n', lambda e: self._toggle_nav()
                       if not isinstance(e.widget, tk.Entry) else None)
        self.root.bind('w', lambda e: self._toggle_delta()
                       if not isinstance(e.widget, tk.Entry) else None)

    def _shortcut_auto(self, event):
        if not isinstance(event.widget, tk.Entry):
            self.auto_var.set(not self.auto_var.get())
            self._toggle_auto()

    def _shortcut_voice(self, event):
        if not isinstance(event.widget, tk.Entry):
            self._toggle_voice()

    # ── Voice Q&A ─────────────────────────────────────────────────────────────

    def _toggle_voice(self):
        if not self.voice_active:
            self.voice_active = True
            self.mic_btn.configure(text="Recording...  [Q] to Stop",
                                   bg='#b71c1c', fg='white')
            self._set_status("Listening...")
            speak("Listening", interrupt=True)
            self.recorder.start()
        else:
            self.voice_active = False
            self.mic_btn.configure(text="Speak Question  [Q]",
                                   bg='#4a4a4a', fg='#ffffff')
            threading.Thread(target=self._do_voice_question, daemon=True).start()

    def _do_voice_question(self):
        self._set_status("Transcribing...")
        question = self.recorder.stop_and_transcribe()
        if not question:
            self._set_status("Didn't catch that — try again")
            speak("Sorry, I didn't catch that.", interrupt=True)
            return
        self.root.after(0, lambda: [
            self.question_entry.delete(0, 'end'),
            self.question_entry.insert(0, question),
            self.question_entry.config(fg='white')
        ])
        self._log(f"Q: {question}", 'question')
        self._do_ask(question)

    # ── Wake word ─────────────────────────────────────────────────────────────

    def _set_wake_word(self):
        word = self.wake_word_var.get().strip().lower()
        if not word:
            return
        was_running = self.wake_enabled
        if was_running:
            self.wake_listener.stop()
        self.wake_listener.phrase = word
        self._log(f'Keyword set to "{word}".', 'info')
        if was_running:
            self.wake_listener.running = False
            self.wake_listener.start()

    def _toggle_wake(self):
        self.wake_enabled = not self.wake_enabled
        word = self.wake_word_var.get().strip().lower() or 'help'
        self.wake_listener.phrase = word
        if self.wake_enabled:
            self.wake_btn.configure(bg='#1565C0', fg='white')
            self._log(f'Keyword enabled — say "{word}" to ask a question.', 'info')
            self.wake_listener.start()
        else:
            self.wake_btn.configure(bg='#4a4a4a', fg='#ffffff')
            self._log("Keyword disabled.", 'info')
            self.wake_listener.stop()

    def _wake_triggered(self):
        if self.voice_active:
            return
        self.voice_active = True
        self.root.after(0, lambda: self.mic_btn.configure(
            text="Listening...  (5s)", bg='#b71c1c', fg='white'))
        self._set_status("Keyword detected — speak your question now")
        speak("Yes?", interrupt=True)
        self.recorder.start()
        threading.Timer(5.0, self._wake_auto_stop).start()

    def _wake_auto_stop(self):
        if not self.voice_active:
            return
        self.voice_active = False
        self.root.after(0, lambda: self.mic_btn.configure(
            text="Speak Question  [Q]", bg='#4a4a4a', fg='#ffffff'))
        threading.Thread(target=self._do_voice_question, daemon=True).start()

    # ── Cleanup ───────────────────────────────────────────────────────────────

    def on_close(self):
        self.running = False
        self.auto_describe = False
        self.wake_listener.stop()
        if self.cap:
            self.cap.release()
        _speech_queue.put(None)
        self.root.destroy()


if __name__ == '__main__':
    root = tk.Tk()
    app = VisionAssistApp(root)
    root.protocol("WM_DELETE_WINDOW", app.on_close)
    root.mainloop()
