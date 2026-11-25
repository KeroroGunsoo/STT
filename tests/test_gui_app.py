import importlib
import sys
import types

import pytest

from speaker_transcriber.pipeline import Transcript, TranscriptSegment


class _Var:
    default = ""

    def __init__(self, value=None):
        self._v = self.default if value is None else value

    def get(self):
        return self._v

    def set(self, v):
        self._v = v


class _IntVar(_Var):
    default = 0


class _DoubleVar(_Var):
    default = 0.0


class _BoolVar(_Var):
    default = False


class _Widget:
    def __init__(self, *a, **k):
        self.kwargs = dict(k)

    def pack(self, *a, **k):
        pass

    def grid(self, *a, **k):
        pass

    def place(self, *a, **k):
        pass

    def config(self, *a, **k):
        self.kwargs.update(k)

    configure = config

    def columnconfigure(self, *a, **k):
        pass

    def start(self, *a, **k):
        pass

    def stop(self, *a, **k):
        pass

    def bind(self, *a, **k):
        pass


class _Text(_Widget):
    def __init__(self, *a, **k):
        super().__init__(*a, **k)
        self.content = []
        self.tags = {}

    def insert(self, where, text, tag=None):
        self.content.append((text, tag))

    def delete(self, *a):
        self.content.clear()

    def see(self, *a):
        pass

    def tag_configure(self, tag, **k):
        self.tags[tag] = k


class _Tk(_Widget):
    def title(self, t):
        self._title = t

    def minsize(self, *a):
        pass

    def after(self, ms, fn):
        pass

    def mainloop(self):
        pass


@pytest.fixture
def fake_tk(monkeypatch):
    tk = types.ModuleType("tkinter")
    ttk = types.ModuleType("tkinter.ttk")
    filedialog = types.ModuleType("tkinter.filedialog")
    messagebox = types.ModuleType("tkinter.messagebox")
    tk.Tk = _Tk
    tk.Text = _Text
    tk.StringVar = _Var
    tk.IntVar = _IntVar
    tk.DoubleVar = _DoubleVar
    tk.BooleanVar = _BoolVar
    tk.END = "end"
    for name in ["Frame", "Label", "LabelFrame", "Entry", "Button", "Combobox", "Spinbox", "Checkbutton", "Progressbar", "Scrollbar"]:
        setattr(ttk, name, type(name, (_Widget,), {}))
    filedialog.askopenfilename = lambda **k: ""
    filedialog.askdirectory = lambda **k: ""
    errors = []
    messagebox.showerror = lambda *a, **k: errors.append(a)
    tk.ttk = ttk
    tk.filedialog = filedialog
    tk.messagebox = messagebox
    monkeypatch.setitem(sys.modules, "tkinter", tk)
    monkeypatch.setitem(sys.modules, "tkinter.ttk", ttk)
    monkeypatch.setitem(sys.modules, "tkinter.filedialog", filedialog)
    monkeypatch.setitem(sys.modules, "tkinter.messagebox", messagebox)
    sys.modules.pop("speaker_transcriber.gui", None)
    gui = importlib.import_module("speaker_transcriber.gui")
    return gui, errors


def _transcript():
    return Transcript(
        source="a.wav",
        language="en",
        duration=4.0,
        speakers=["SPEAKER_00", "SPEAKER_01"],
        segments=[
            TranscriptSegment(0.0, 2.0, "SPEAKER_00", "Hello Bob.", []),
            TranscriptSegment(2.0, 4.0, "SPEAKER_01", "Hi Alice.", []),
        ],
    )


def test_app_runs_pipeline_and_renders(fake_tk, monkeypatch, tmp_path):
    gui, errors = fake_tk
    transcript = _transcript()
    seen_logs = []
    monkeypatch.setattr(
        gui, "transcribe_file",
        lambda path, log=None, **kw: (seen_logs.append("model loaded"), log("transcribing..."), transcript)[2],
    )

    app = gui.TranscriberApp(gui.tk.Tk())
    wav = tmp_path / "a.wav"
    wav.write_bytes(b"RIFFfake")
    app.vars["input"].set(str(wav))
    app.vars["outdir"].set(str(tmp_path / "out"))
    app.vars["model"].set("tiny")
    app.vars["num_speakers"].set(2)

    app._on_run()
    assert app.worker is not None
    app.worker.join(timeout=30)
    app._poll()

    assert not errors
    assert seen_logs == ["model loaded"]
    assert any("transcribing..." in line for line, _ in app.log_text.content)
    rendered = app.result_text.content
    assert ("[00:00:00.000] SPEAKER_00: Hello Bob.\n", "SPEAKER_00") in rendered
    assert ("[00:00:02.000] SPEAKER_01: Hi Alice.\n", "SPEAKER_01") in rendered
    assert app.result_text.tags["SPEAKER_00"]["foreground"] != app.result_text.tags["SPEAKER_01"]["foreground"]
    assert app.run_btn.kwargs["state"] == "normal"
    assert app.open_btn.kwargs["state"] == "normal"
    assert (tmp_path / "out" / "a.txt").exists()
    assert (tmp_path / "out" / "a.json").exists()
    assert (tmp_path / "out" / "a.srt").exists()


def test_app_rejects_missing_input(fake_tk):
    gui, errors = fake_tk
    app = gui.TranscriberApp(gui.tk.Tk())
    app.vars["input"].set("/nonexistent/file.mp3")
    app._on_run()
    assert app.worker is None
    assert errors and "audio file" in errors[0][1]


def test_app_reports_worker_error(fake_tk, monkeypatch, tmp_path):
    gui, errors = fake_tk

    def boom(path, log=None, **kw):
        raise RuntimeError("model exploded")

    monkeypatch.setattr(gui, "transcribe_file", boom)
    app = gui.TranscriberApp(gui.tk.Tk())
    wav = tmp_path / "a.wav"
    wav.write_bytes(b"RIFFfake")
    app.vars["input"].set(str(wav))
    app._on_run()
    app.worker.join(timeout=30)
    app._poll()
    assert any("model exploded" in str(a) for a in errors)
    assert app.run_btn.kwargs["state"] == "normal"
