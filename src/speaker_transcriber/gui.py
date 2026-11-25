from __future__ import annotations

import os
import queue
import subprocess
import sys
import threading
import tkinter as tk
from pathlib import Path
from tkinter import filedialog, messagebox, ttk

from .formats import write_outputs
from .gui_support import (
    GuiOptions,
    options_to_kwargs,
    resolve_output_dir,
    speaker_color,
    speaker_order,
    transcript_lines,
)
from .pipeline import transcribe_file

MODEL_CHOICES = ["tiny", "base", "small", "medium", "large-v3"]
DEVICE_CHOICES = ["cpu", "cuda"]
FORMAT_CHOICES = ["txt", "json", "srt"]


class TranscriberApp:
    def __init__(self, root: tk.Tk):
        self.root = root
        self.root.title("STT — speaker-aware transcription")
        self.root.minsize(780, 600)
        self.msg_q: queue.Queue = queue.Queue()
        self.worker: threading.Thread | None = None
        self.vars: dict = {}
        self._build()

    def _build(self) -> None:
        pad = {"padx": 6, "pady": 4}

        top = ttk.Frame(self.root)
        top.pack(fill="x", **pad)
        self.vars["input"] = tk.StringVar()
        self.vars["outdir"] = tk.StringVar()
        ttk.Label(top, text="Audio file:").grid(row=0, column=0, sticky="w")
        ttk.Entry(top, textvariable=self.vars["input"], width=62).grid(row=0, column=1, sticky="we")
        ttk.Button(top, text="Browse…", command=self._browse_input).grid(row=0, column=2)
        ttk.Label(top, text="Output dir:").grid(row=1, column=0, sticky="w")
        ttk.Entry(top, textvariable=self.vars["outdir"], width=62).grid(row=1, column=1, sticky="we")
        ttk.Button(top, text="Browse…", command=self._browse_outdir).grid(row=1, column=2)
        top.columnconfigure(1, weight=1)

        opts = ttk.LabelFrame(self.root, text="Options")
        opts.pack(fill="x", **pad)
        self.vars["model"] = tk.StringVar(value="base")
        self.vars["language"] = tk.StringVar(value="")
        self.vars["device"] = tk.StringVar(value="cpu")
        self.vars["num_speakers"] = tk.IntVar(value=0)
        self.vars["cluster_threshold"] = tk.DoubleVar(value=0.30)
        self.vars["formats"] = {f: tk.BooleanVar(value=True) for f in FORMAT_CHOICES}
        ttk.Label(opts, text="Model:").grid(row=0, column=0, sticky="w")
        ttk.Combobox(opts, textvariable=self.vars["model"], values=MODEL_CHOICES, state="readonly", width=12).grid(row=0, column=1, sticky="w")
        ttk.Label(opts, text="Language (blank=auto):").grid(row=0, column=2, sticky="w")
        ttk.Entry(opts, textvariable=self.vars["language"], width=8).grid(row=0, column=3, sticky="w")
        ttk.Label(opts, text="Device:").grid(row=0, column=4, sticky="w")
        ttk.Combobox(opts, textvariable=self.vars["device"], values=DEVICE_CHOICES, state="readonly", width=8).grid(row=0, column=5, sticky="w")
        ttk.Label(opts, text="Speakers (0=auto):").grid(row=1, column=0, sticky="w")
        ttk.Spinbox(opts, from_=0, to=12, textvariable=self.vars["num_speakers"], width=6).grid(row=1, column=1, sticky="w")
        ttk.Label(opts, text="Cluster threshold:").grid(row=1, column=2, sticky="w")
        ttk.Spinbox(opts, from_=0.05, to=0.9, increment=0.05, textvariable=self.vars["cluster_threshold"], width=6).grid(row=1, column=3, sticky="w")
        ttk.Label(opts, text="Formats:").grid(row=1, column=4, sticky="w")
        for i, fmt in enumerate(FORMAT_CHOICES):
            ttk.Checkbutton(opts, text=fmt, variable=self.vars["formats"][fmt]).grid(row=1, column=5 + i, sticky="w")
        ttk.Label(opts, text="First run downloads the models (~150 MB for base).").grid(row=2, column=0, columnspan=6, sticky="w")

        run = ttk.Frame(self.root)
        run.pack(fill="x", **pad)
        self.run_btn = ttk.Button(run, text="Transcribe", command=self._on_run)
        self.run_btn.pack(side="left")
        self.open_btn = ttk.Button(run, text="Open output folder", command=self._open_outdir, state="disabled")
        self.open_btn.pack(side="left", padx=6)
        self.status = ttk.Label(run, text="Idle")
        self.status.pack(side="left", padx=12)
        self.progress = ttk.Progressbar(run, mode="indeterminate", length=160)
        self.progress.pack(side="right")

        log_frame = ttk.LabelFrame(self.root, text="Log")
        log_frame.pack(fill="x", **pad)
        self.log_text = tk.Text(log_frame, height=5, state="disabled")
        self.log_text.pack(fill="x")

        res_frame = ttk.LabelFrame(self.root, text="Transcript")
        res_frame.pack(fill="both", expand=True, **pad)
        self.result_text = tk.Text(res_frame, wrap="word")
        self.result_text.pack(fill="both", expand=True)

        self.root.after(100, self._poll)

    def _browse_input(self) -> None:
        path = filedialog.askopenfilename(
            title="Choose audio file",
            filetypes=[("Audio", "*.mp3 *.wav *.m4a *.mp4 *.flac *.ogg *.webm *.mkv"), ("All files", "*.*")],
        )
        if path:
            self.vars["input"].set(path)
            if not self.vars["outdir"].get():
                self.vars["outdir"].set(str(Path(path).resolve().parent))

    def _browse_outdir(self) -> None:
        chosen = filedialog.askdirectory(title="Choose output folder")
        if chosen:
            self.vars["outdir"].set(chosen)

    def _collect(self) -> GuiOptions:
        return GuiOptions(
            input_path=self.vars["input"].get().strip(),
            output_dir=self.vars["outdir"].get().strip(),
            model=self.vars["model"].get(),
            language=self.vars["language"].get(),
            device=self.vars["device"].get(),
            num_speakers=int(self.vars["num_speakers"].get() or 0),
            cluster_threshold=float(self.vars["cluster_threshold"].get()),
            formats=tuple(f for f, v in self.vars["formats"].items() if v.get()),
        )

    def _on_run(self) -> None:
        opts = self._collect()
        if not opts.input_path or not Path(opts.input_path).is_file():
            messagebox.showerror("Missing input", "Please choose an existing audio file.")
            return
        if not opts.formats:
            messagebox.showerror("No formats", "Select at least one output format.")
            return
        self.run_btn.config(state="disabled")
        self.open_btn.config(state="disabled")
        self._set_status("Running…")
        self.progress.start(80)
        self.worker = threading.Thread(target=self._worker, args=(opts,), daemon=True)
        self.worker.start()

    def _worker(self, opts: GuiOptions) -> None:
        try:
            transcript = transcribe_file(opts.input_path, log=self._log, **options_to_kwargs(opts))
            outdir = resolve_output_dir(opts)
            written = write_outputs(transcript, outdir, formats=opts.formats)
            self.msg_q.put(("done", transcript, written))
        except Exception as exc:
            self.msg_q.put(("error", exc))

    def _log(self, msg) -> None:
        self.msg_q.put(("log", str(msg)))

    def _poll(self) -> None:
        try:
            while True:
                kind, *payload = self.msg_q.get_nowait()
                if kind == "log":
                    self._append_log(payload[0])
                elif kind == "done":
                    transcript, written = payload
                    self._render(transcript)
                    self._set_status(f"Done — {len(transcript.segments)} segment(s), {len(transcript.speakers)} speaker(s)")
                    self._append_log("wrote: " + ", ".join(str(p) for p in written.values()))
                    self.open_btn.config(state="normal")
                elif kind == "error":
                    self._set_status("Error")
                    self._append_log(f"ERROR: {payload[0]}")
                    messagebox.showerror("Transcription failed", str(payload[0]))
        except queue.Empty:
            pass
        if self.worker is not None and not self.worker.is_alive():
            self.progress.stop()
            self.run_btn.config(state="normal")
        self.root.after(100, self._poll)

    def _append_log(self, line: str) -> None:
        self.log_text.config(state="normal")
        self.log_text.insert("end", line + "\n")
        self.log_text.see("end")
        self.log_text.config(state="disabled")

    def _render(self, transcript) -> None:
        self.result_text.delete("1.0", "end")
        order = speaker_order(transcript)
        for speaker in order:
            self.result_text.tag_configure(speaker, foreground=speaker_color(speaker, order))
        for ts, speaker, text in transcript_lines(transcript):
            self.result_text.insert("end", f"[{ts}] {speaker}: {text}\n", speaker)
        self.result_text.see("1.0")

    def _set_status(self, text: str) -> None:
        self.status.config(text=text)

    def _open_outdir(self) -> None:
        outdir = resolve_output_dir(self._collect()) or "."
        try:
            if sys.platform.startswith("win"):
                os.startfile(outdir)
            elif sys.platform == "darwin":
                subprocess.Popen(["open", outdir])
            else:
                subprocess.Popen(["xdg-open", outdir])
        except Exception as exc:
            self._append_log(f"could not open folder: {exc}")


def main(argv: list[str] | None = None) -> int:
    root = tk.Tk()
    TranscriberApp(root)
    root.mainloop()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
