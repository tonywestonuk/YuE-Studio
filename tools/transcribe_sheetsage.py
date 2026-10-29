#!/usr/bin/env python3
"""Transcribe a recording to melody ABC with SheetSage2, reporting JSON progress lines.

Runs inside the dedicated SheetSage2 environment (torch 2.8, transformers 4.45 —
incompatible with the YuE2 environment, hence a separate interpreter).  Prints one
JSON object per line on stdout so the app worker can forward progress verbatim:

  {"stage": "starting", "detail": ...}
  {"stage": "progress", "fraction": 0..1 | absent, "detail": ...}
  {"stage": "done", "abc": <score text>, "warnings": [...], "output": <dir>}
  {"stage": "failed", "code": "afconvert|abc_error|crash", "message": ...}

Audio is decoded with macOS's own afconvert and handed to the model as a raw
waveform, so no FFmpeg install is needed.  Standalone: no imports outside the
standard library until the model's cache is checked (huggingface_hub) and it
actually loads (torch, transformers).
"""

import argparse
import hashlib
import importlib.metadata
import inspect
import json
import re
import subprocess
import sys
import threading
import time
from pathlib import Path

AFCONVERT = "/usr/bin/afconvert"


def say(**obj):
    print(json.dumps(obj), flush=True)


def fail(code, message, output=None):
    if output is not None:
        write_json(output / "failure.json", {"status": "failed", "code": code, "error": message})
    say(stage="failed", code=code, message=message)


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def write_json(path, value):
    Path(path).write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def check_melody_abc(text):
    """A melody score must have a voice and no chord symbols; returns a problem or None."""
    body = [l for l in text.splitlines() if l.strip() and not re.match(r"^[A-Za-z]:", l.strip())]
    if not body:
        return "score has no music lines"
    if not any(l.strip().startswith("V:") for l in text.splitlines()):
        return "score has no voice header"
    if any(re.search(r'"[^"]+"', l) for l in body):
        return "melody transcription contains unexpected chord symbols"
    return None


def decode(audio, output):
    """Decode any macOS-readable format to 16-bit WAV; returns the WAV path."""
    wav = output / "input.wav"
    proc = subprocess.run([AFCONVERT, str(audio), str(wav), "-f", "WAVE", "-d", "LEI16"],
                          capture_output=True, text=True)
    if proc.returncode != 0 or not wav.is_file():
        tail = (proc.stderr or proc.stdout or "").strip().splitlines()
        raise DecodeError(tail[-1] if tail else f"afconvert exited {proc.returncode}")
    return wav


class DecodeError(Exception):
    pass


class BackboneError(Exception):
    pass


def pinned_backbone(model):
    """(parent, revision) SheetSage2's cached config.json pins, or None when not cached/pinned."""
    from huggingface_hub import hf_hub_download
    try:
        config = json.loads(Path(hf_hub_download(model, "config.json", local_files_only=True)).read_text(encoding="utf-8"))
    except Exception:
        return None        # nothing cached yet: an online load fetches everything itself
    parent, revision = config.get("base_model_name_or_path"), config.get("base_model_revision")
    if isinstance(parent, str) and isinstance(revision, str) and parent and revision and not Path(parent).is_dir():
        return parent, revision
    return None


def ensure_backbone(model):
    """Make sure the MERT-v2 backbone is cached at the revision SheetSage2 pins.

    SheetSage2's remote code resolves its parent at ``base_model_revision`` only.  An
    install that fetched the parent's ``main`` after that branch moved on has a newer
    snapshot the pinned lookup cannot use, and offline loads then fail with
    "We couldn't connect to huggingface.co ... configuration_mert2.py" (issues #8, #11).
    Fetches the pinned revision even under --offline; raises BackboneError if it cannot.
    """
    from huggingface_hub import snapshot_download
    from huggingface_hub.constants import HF_HUB_CACHE
    pinned = pinned_backbone(model)
    if pinned is None:
        return
    parent, revision = pinned
    try:
        snapshot_download(parent, revision=revision, local_files_only=True)
        return
    except Exception:
        pass
    detail = "downloading the MERT-v2 backbone revision SheetSage2 pins (about 1.3 GB)"
    say(stage="progress", fraction=0.03, detail=detail)
    started = time.time()
    finished = threading.Event()

    def heartbeat():
        while not finished.wait(5):
            elapsed = int(time.time() - started)
            say(stage="progress", detail=f"{detail} · {elapsed // 60}:{elapsed % 60:02d} elapsed")

    threading.Thread(target=heartbeat, daemon=True).start()
    try:
        snapshot_download(parent, revision=revision)
    except Exception as exc:
        raise BackboneError(
            f"SheetSage2 needs {parent} at revision {revision}, which is missing from {HF_HUB_CACHE} "
            f"and could not be downloaded ({type(exc).__name__}: {exc}). "
            "Reinstall SheetSage2 from the app (Transcribe), or run "
            f"tools/download_models.py {model} with HF_HOME pointed at the models directory.") from exc
    finally:
        finished.set()


def run(args):
    if not args.audio.is_file():
        fail("crash", f"no such file: {args.audio}")
        return 2
    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=True)

    say(stage="starting", detail="decoding audio")
    try:
        wav = decode(args.audio, output)
    except DecodeError as exc:
        fail("afconvert", str(exc), output)
        return 3

    try:
        ensure_backbone(args.model)
    except BackboneError as exc:
        fail("crash", str(exc), output)
        return 2

    say(stage="progress", fraction=0.04, detail="loading SheetSage2")
    import numpy as np
    import soundfile
    import torch
    from transformers import AutoModel

    data, sample_rate = soundfile.read(wav, dtype="float32", always_2d=True)
    waveform = np.ascontiguousarray(data.T)          # channels-first, as transcribe() expects
    wav.unlink()                                      # decoded copy no longer needed

    torch.set_num_threads(args.threads)
    model = AutoModel.from_pretrained(args.model, trust_remote_code=True,
                                      local_files_only=args.offline).eval().to(args.device)
    parameter = inspect.signature(model.transcribe).parameters.get("melody_only")
    if parameter is None or parameter.kind == inspect.Parameter.POSITIONAL_ONLY:
        fail("crash", "this SheetSage2 revision does not expose melody_only", output)
        return 2

    prompts = ["timestamp", "downbeat_meter", "structure", "key",
               "melody_vocal" if args.task == "melody-vocal" else "melody_full"]
    write_json(output / "input.json", {
        "source_name": args.audio.name, "source_audio_sha256": sha256(args.audio),
        "model": args.model, "offline": args.offline, "prompts": prompts,
        "task": args.task, "max_seconds": args.max_seconds,
        "device": args.device, "dtype": args.dtype,
        "packages": {name: importlib.metadata.version(name)
                     for name in ("torch", "transformers", "huggingface-hub")},
    })

    started = time.time()
    seen = [0.0]

    def on_progress(*progress, **_):
        # The remote code's callback contract isn't pinned; accept a leading
        # 0..1 float when one appears and never let a surprise shape crash us.
        try:
            fraction = float(progress[0])
            if 0.0 <= fraction <= 1.0 and fraction > seen[0]:
                seen[0] = fraction
                say(stage="progress", fraction=round(0.05 + 0.95 * fraction, 3),
                    detail=f"transcribing · {int(time.time() - started) // 60}:{int(time.time() - started) % 60:02d}")
        except (TypeError, ValueError, IndexError):
            pass

    def heartbeat():
        while not finished.is_set():
            finished.wait(5)
            if not finished.is_set() and seen[0] == 0.0:
                elapsed = int(time.time() - started)
                say(stage="progress", detail=f"transcribing · {elapsed // 60}:{elapsed % 60:02d} elapsed")

    finished = threading.Event()
    threading.Thread(target=heartbeat, daemon=True).start()
    say(stage="progress", fraction=0.05, detail="transcribing")
    options = dict(sampling_rate=sample_rate, output_dir=str(output), prompts=prompts,
                   dtype=args.dtype, max_seconds=args.max_seconds, melody_only=True)
    try:
        try:
            result = model.transcribe(waveform, progress=on_progress, **options)
        except TypeError as exc:
            if "progress" not in str(exc):
                raise
            result = model.transcribe(waveform, **options)   # revision without a callback
    except Exception as exc:
        finished.set()
        fail("abc_error", f"{type(exc).__name__}: {exc}", output)
        return 4
    finished.set()

    abc = result.get("abc")
    if result.get("abc_error") or not abc:
        fail("abc_error", str(result.get("abc_error") or "transcription produced no ABC"), output)
        return 4
    problem = check_melody_abc(abc)
    if problem:
        fail("abc_error", problem, output)
        return 4
    if not (output / "score.abc").is_file():
        (output / "score.abc").write_text(abc, encoding="utf-8")

    write_json(output / "transcription_manifest.json", {
        "status": "complete", "warnings": result.get("warnings", []),
        "seconds": round(time.time() - started, 1),
        "artifacts": {str(p.relative_to(output)): sha256(p)
                      for p in sorted(output.rglob("*")) if p.is_file()},
    })
    say(stage="done", abc=abc, warnings=result.get("warnings", []), output=str(output))
    return 0


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("audio", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--task", choices=("melody-full", "melody-vocal"), default="melody-full")
    parser.add_argument("--model", default="m-a-p/SheetSage2")
    parser.add_argument("--offline", action="store_true")
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--dtype", choices=("bf16", "fp32"), default="fp32")
    parser.add_argument("--max-seconds", type=float)
    parser.add_argument("--threads", type=int, default=4)
    args = parser.parse_args()
    try:
        return run(args)
    except Exception as exc:
        fail("crash", f"{type(exc).__name__}: {exc}")
        return 2


if __name__ == "__main__":
    sys.exit(main())
