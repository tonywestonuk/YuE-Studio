#!/usr/bin/env python3
"""Long-lived YuE2 worker for native front ends: JSON lines in on stdin, JSON events out on stdout.

Songs are processes, the GPU and the Neural Engine are resources, and one scheduler assigns
resources to songs by priority (the order the songs were added). See ``Scheduler.schedule`` for
the whole policy; in short:

  Neural Engine   the highest-priority song that wants it, including a song already solving on
                  the GPU, which moves across mid-solve if it outranks everything waiting.
  GPU, tokenizing the highest-priority queued song, taking every other queued song of the same
                  kind along in one batch (one row or four cost the same per step).
  GPU, synthesis  drafts, and full-quality songs while the Neural Engine is busy. A queued song
                  that outranks the song synthesizing on the GPU starts its batch and the
                  synthesis pauses between steps until the GPU is free again.
  Rendering       always admitted: its tiles interleave with token steps.

Requests: {"cmd": "generate", "style", "lyrics", "cot": "full|melody|off", "seed", "random_seed", "batch",
           "max_tokens", "abc": str|null, "abc_open": bool (the abc is a hummed opening the planner continues),
           "quality": "draft|full", "engines": "gpu|gpu+ane", "draft_steps",
           "instrumental": bool, "title": str}     (engines: whether the Neural Engine may synthesize; "gpu" keeps
                                                     its 2.8 GB unmapped. title: names the run folder and is stored with each song)
          {"cmd": "render", "path": song directory or its audio.flac, "quality": "full|draft", "engines": "gpu|gpu+ane"}
          {"cmd": "cancel", "path"}   {"cmd": "stop"}   {"cmd": "ping"}   {"cmd": "quit"}
          {"cmd": "transcribe", "id", "audio", "task": "melody-full|melody-vocal", "offline": bool}
          {"cmd": "transcribe_cancel", "id"}
Events:   {"event": "ready"}   {"event": "log", "message"}   {"event": "pong"}   {"event": "error", "message"}
          {"event": "started", "job", "output", "songs": [{"index", "seed", "path", "priority"}]}
          {"event": "stage", "path", "priority", "stage": "queued|planning|tokens|synth|decode|ready|failed|cancelled",
           "detail", "engine"}
          {"event": "progress", "path", "fraction": 0-1, "detail", "gflops": rate}
          {"event": "song", "index", "path", "score", "seconds", "seed", "truncated", "quality", "steps", "engine"}
          {"event": "failed", "path", "message"}   {"event": "idle"}   (every song finished or was cancelled)
          {"event": "transcribe", "id", "stage": "starting|progress|done|failed|cancelled",
           "fraction", "detail", "abc", "warnings", "output", "message", "code"}   (keyed by the
           request's client id, never by "path", so song-keyed handlers can't misapply it)
"""
import datetime as dt, itertools, json, os, subprocess, sys, threading, time, traceback
from pathlib import Path
os.environ.setdefault("TQDM_DISABLE", "1")          # coremltools progress bars would otherwise flood the app log
import warnings
warnings.filterwarnings("ignore")

ROOT = Path(__file__).resolve().parents[1]
OUTPUT_DIR = Path(os.environ.get("YUE2_OUTPUT_DIR", ROOT / "outputs" / "app"))
PIPE = None
LOCK = threading.Lock()                              # stdout
MODEL_LOCK = threading.Lock()                        # model load/unload and memory release
IDLE_UNLOAD_S = float(os.environ.get("YUE2_IDLE_UNLOAD_S", 600))   # drop the model after this long idle (reloads in ~1 s)
LAST_ACTIVE = [time.time()]
PHYSICAL_GIB = float(os.environ.get("YUE2_PHYSICAL_GIB") or os.sysconf("SC_PAGE_SIZE") * os.sysconf("SC_PHYS_PAGES") / 2**30)
CONCURRENT = os.environ.get("YUE2_PIPELINE", "1" if PHYSICAL_GIB >= 24 else "0") != "0"   # overlap the resources
MAX_BATCH = int(os.environ.get("YUE2_MAX_BATCH", 4 if PHYSICAL_GIB >= 24 else 2))          # songs per token batch
# Token engine: "mlx" streams 8-bit weights (2.3 GB, ~1.8x faster steps), quantized in about a
# second before each batch and dropped after it, since the PyTorch copy the synthesis prefill
# needs stays resident anyway; "torch" is the original path.
AR_ENGINE = os.environ.get("YUE2_AR_ENGINE", "mlx" if PHYSICAL_GIB >= 24 else "torch")
ANE_MAX_FRAMES = 12288    # MIL v7 compiles up to here (9216 x 14336 verified); 12288 x 14336 is refused, and a
                          # refused compile fails within seconds and falls back to the GPU
# PyTorch's Metal backend encodes every thread into one command buffer, so all PyTorch GPU work
# (token steps, the synthesis prefill, decode tiles) takes turns on this lock; the Neural Engine
# and MLX solvers run outside it, which is what lets synthesis overlap token generation.
from yue2.locks import FairLock
TORCH_LOCK = FairLock()
# SheetSage2 transcription runs as a subprocess in its own environment (its pins conflict with
# ours) with its own Metal context, so TORCH_LOCK does not apply — only one at a time, though.
TRANSCRIBE_LOCK = threading.Lock()
TRANSCRIBE_PROC = [None]


def emit(**event):
    event.setdefault("t", round(time.time(), 2))
    with LOCK:
        sys.stdout.write(json.dumps(event) + "\n"); sys.stdout.flush()


def log(message):
    emit(event="log", message=message, time=dt.datetime.now().strftime("%H:%M:%S"))


def brief(exc, limit=100):
    """The first line of an error, cut short, for a status line or a one-line log message."""
    lines = str(exc).strip().splitlines()
    return lines[0][:limit] if lines else type(exc).__name__


def log_full_error(what, exc):
    """The complete text of an error for the log pane, when the one-line summary left part of it
    out. The Neural Engine compiler's real reason (for example its
    'Error Domain=com.apple.appleneuralengine.compiler Code=22 "Compiler internal error: Couldn't
    do topological sort"') is in the underlying error on a later line; the pane keeps newlines,
    so continuation lines are indented under the heading."""
    text = str(exc).strip()
    lines = text.splitlines()
    if len(lines) > 1 or len(text) > 100:
        log(f"{what}:\n" + "\n".join("    " + line for line in lines))


def pipeline():
    """The loaded pipeline (loading or reloading the model as needed). Call under MODEL_LOCK."""
    global PIPE
    if PIPE is None:
        import torch
        from yue2 import YuE2Pipeline
        device = "cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu"
        log(f"Loading YuE2 model on {device} (first run only)")
        t0 = time.perf_counter()
        # Apple Silicon: synthesis runs on the Neural Engine (or MLX), so only the AR path is
        # kept in PyTorch; the NAR weights are read from the checkpoint by those engines.
        lean = device == "mps" and os.environ.get("YUE2_LEAN", "1") != "0"
        # The VAE decodes in tiles; smaller tiles halve its activation peak (about 3.6 GB at 1024
        # frames) on machines where the whole memory is shared with the model.
        PIPE = YuE2Pipeline.from_pretrained("m-a-p/YuE2-3B", device=device, progress=False, lean=lean,
                                            vae_core_frames=1024 if PHYSICAL_GIB >= 24 else 512)
        log(f"Physical memory {PHYSICAL_GIB:.0f} GB: {'lean' if lean else 'full'} model, VAE tile {PIPE.vae_core_frames} frames, "
            f"token batches of up to {MAX_BATCH} on {'MLX with 8-bit weights' if token_engine() == 'mlx' else 'PyTorch'}, "
            f"resources {'overlap' if CONCURRENT else 'take turns'}")
        PIPE._load_model()
        log(f"Model ready in {time.perf_counter() - t0:.0f} s")
    elif PIPE._model is None:                        # dropped while idle
        t0 = time.perf_counter()
        PIPE._load_model()
        log(f"Model reloaded in {time.perf_counter() - t0:.0f} s")
    return PIPE


def token_engine():
    if AR_ENGINE == "mlx":
        try:
            import mlx.core  # noqa: F401
            return "mlx"
        except ImportError:
            return "torch"
    return "torch"


def generate_tokens(model, *args, **kwargs):
    """The batched token decoder on the configured engine (same interface either way)."""
    if token_engine() == "mlx":
        from yue2 import ar_mlx
        if getattr(model, "_yue2_ar_mlx", None) is None:
            t0 = time.perf_counter()
            ar_mlx.weights_for(model)
            log(f"Token weights quantized to 8 bits for MLX in {time.perf_counter() - t0:.0f} s ({model._yue2_ar_mlx.bytes() / 1e9:.1f} GB)")
        return ar_mlx.generate_tokens_batched(model, *args, **kwargs)
    from yue2.batched import generate_tokens_batched
    return generate_tokens_batched(model, *args, **kwargs)


def trim_gpu_memory(model, why):
    """Drop GPU-side copies nothing is using right now (they come back in seconds when needed):
    the 8-bit token weights between batches, the MLX synthesis weights when no song is on the
    GPU lane, and MLX's buffer cache. Keeps the worker's footprint from squeezing the Neural
    Engine's mapped memory, whose evaluate fails under memory pressure."""
    import gc
    with SCHED.cv:
        gpu_synth_needed = SCHED.gpu_synth is not None or any(
            s.state == SYNTH_WAIT and (not s.wants_ane or SCHED.ane_synth is not None) for s in SCHED.songs)
        batch_running = SCHED.token_batch is not None
    freed = []
    if not batch_running and getattr(model, "_yue2_ar_mlx", None) is not None:
        model._yue2_ar_mlx = None; freed.append("token weights")
    if not gpu_synth_needed and getattr(model, "_yue2_mlx_weights", None) is not None:
        model._yue2_mlx_weights = None; freed.append("synthesis weights")
    gc.collect()
    try:
        import mlx.core as mx
        (getattr(mx, "clear_cache", None) or mx.metal.clear_cache)()
    except Exception:
        pass
    if freed:
        log(f"Freed MLX {' and '.join(freed)} ({why})")


def acquire_model():
    with MODEL_LOCK:
        LAST_ACTIVE[0] = time.time()
        pipe = pipeline()
        return pipe, pipe._load_model()


def ane_available():
    from yue2.ane import runtime as ane_runtime
    return ane_runtime.available()


def ane_can_take(n_frames):
    from yue2.ane import runtime as ane_runtime
    return ane_available() and ane_runtime.bucket(n_frames + 2, ane_runtime.S_STEP) <= ANE_MAX_FRAMES


# ── The iPhone companion (app/YuERemote): a second Neural Engine on the local network ─────────
REMOTE = None                                     # yue2.remote.client.RemoteClient while a phone is connected
REMOTE_LOCK = threading.Lock()
REMOTE_MAX_ROWS = int(os.environ.get("YUE2_REMOTE_MAX_ROWS", 4096))   # largest bucket its compiler has accepted


def remote_can_take(song):
    from yue2.remote.client import S_STEP, bucket
    return REMOTE is not None and song.allow_ane and not song.remote_failed and bucket(song.frames + 2, S_STEP) <= REMOTE_MAX_ROWS


def set_remote(req):
    """The app found (or lost) the phone: {"cmd": "remote", "host", "port", "name"} / host null."""
    global REMOTE
    from yue2.remote.client import RemoteClient
    host, port, name = req.get("host"), req.get("port"), req.get("name", "")
    with REMOTE_LOCK:
        old = REMOTE
        if not host:
            if old is not None:
                REMOTE = None; old.close()
                log(f"{old.label()} disconnected")
                emit(event="remote", state="gone", name=old.label())
            SCHED.tick(); return
        if old is not None and (old.host, old.port) == (host, int(port)):
            return
        try:
            client = RemoteClient(host, port, name)
        except Exception as exc:
            log(f"Cannot use the iPhone at {host}:{port} ({str(exc)[:120]})")
            emit(event="remote", state="error", name=name, detail=str(exc)[:200]); return
        if old is not None:
            old.close()
        REMOTE = client
        weights = "weights cached" if client.info.get("weights") else "weights are sent on first use (2.8 GB)"
        log(f"{client.label()} ready as a synthesis engine ({client.info.get('model', '')}, "
            f"{client.info.get('memory_available_mb', 0)} MB free, {weights})")
        emit(event="remote", state="connected", name=client.label(), detail=weights)
    SCHED.tick()


def clear_remote(reason):
    global REMOTE
    with REMOTE_LOCK:
        old, REMOTE = REMOTE, None
    if old is not None:
        old.close()
        log(f"{old.label()} dropped: {reason}")
        emit(event="remote", state="gone", name=old.label(), detail=reason)
    SCHED.tick()


# ── Throughput estimates (rough, for the status line) ────────────────────────
# Matmul work only, 2 FLOP per multiply-add, derived from the model shapes; attention counted as
# QK^T and PV over the attended length. The VAE figure was measured with torch's FLOP counter.
VAE_GFLOP_PER_FRAME = 2.31


def _shapes(cfg):
    return cfg.hidden_size, cfg.intermediate_size, cfg.num_attention_heads, cfg.num_key_value_heads, cfg.head_dim, cfg.num_hidden_layers


def flops_per_token(cfg, context):
    """One autoregressive step of one sequence attending ``context`` tokens."""
    D, F, H, KV, HD, L = _shapes(cfg)
    linear = 2 * (D * H * HD + 2 * D * KV * HD + H * HD * D + 3 * D * F)
    attention = 4 * context * H * HD
    return L * (linear + attention) + 2 * D * 32770             # sliced output head (song phase)


def flops_per_pass(cfg, S, P):
    """One velocity evaluation of the synthesis network over S frames attending S + P keys."""
    D, F, H, KV, HD, L = _shapes(cfg)
    linear = 2 * S * (D * H * HD + 2 * D * KV * HD + H * HD * D + 3 * D * F)
    attention = 4 * S * (S + P) * H * HD
    return L * (linear + attention)


class Rate:
    """Smoothed throughput in GFLOP/s from (work done, time) samples."""

    def __init__(self, alpha=0.3):
        self.alpha, self.value, self.last = alpha, None, None

    def add(self, flops):
        now = time.perf_counter()
        if self.last is not None and now > self.last:
            sample = flops / (now - self.last) / 1e9
            self.value = sample if self.value is None else self.alpha * sample + (1 - self.alpha) * self.value
        self.last = now
        return self.value


# ── Processes ────────────────────────────────────────────────────────────────

QUEUED, PLANNING, TOKENIZING, SYNTH_WAIT, SYNTHING, RENDER_WAIT, RENDERING, DONE, FAILED, CANCELLED = (
    "queued", "planning", "tokenizing", "synth_wait", "synthing", "render_wait", "rendering", "done", "failed", "cancelled")
UI_STAGE = {QUEUED: "queued", PLANNING: "planning", TOKENIZING: "tokens", SYNTH_WAIT: "synth", SYNTHING: "synth",
            RENDER_WAIT: "decode", RENDERING: "decode", DONE: "ready", FAILED: "failed", CANCELLED: "cancelled"}


class Song:
    """One song on its way through the stages: a process with a priority (its arrival number)."""
    _sequence = itertools.count(1)

    def __init__(self, run, index, seed, request, directory, quality, steps, limit, instrumental=False, allow_ane=True, title=""):
        self.priority = next(Song._sequence)
        self.title = title
        self.run, self.index, self.seed, self.request = run, index, seed, request
        self.directory = Path(directory)
        self.path = str(self.directory / "audio.flac")
        self.quality, self.steps, self.limit, self.instrumental = quality, steps, limit, instrumental
        self.allow_ane = allow_ane                # the user's engine choice: the Neural Engine may take this song
        self.mode = request.cot
        self.needs_plan = request.cot != "off" and (request.abc is None or request.abc_open)
        self.cancel = threading.Event()
        self.state = QUEUED
        self.plan = None; self.codec = None; self.timing = {}; self.truncated = False
        self.latents = None; self.used_engine = None; self.nar_seconds = 0.0
        self.wants_ane = False                    # decided once the song's length is known
        self.program_ready = threading.Event()    # its Neural Engine program is compiled
        self.program_failed = False
        self.migrate = None                       # "ane" / "remote": granted that engine mid-solve
        self.migrated = False
        self.remote_failed = False                # the iPhone could not run this song's length

    @property
    def label(self):
        return f"{self.directory.parent.name}/{self.directory.name}"

    @property
    def frames(self):
        return len(self.codec) if self.codec is not None else 0

    def set_state(self, state, detail="", **extra):
        self.state = state
        emit(event="stage", path=self.path, priority=self.priority, stage=UI_STAGE[state], detail=detail, **extra)

    def progress(self, fraction, detail="", gflops=None):
        extra = {} if gflops is None else {"gflops": round(gflops, 1)}
        emit(event="progress", path=self.path, fraction=max(0.0, min(1.0, fraction)), detail=detail, **extra)

    def decide_engine(self):
        """A song goes to the Neural Engine when the user allowed it and its compiler accepts the length."""
        self.wants_ane = self.allow_ane and ane_can_take(self.frames)


# ── Scheduler ────────────────────────────────────────────────────────────────

class Scheduler:
    """Assigns the GPU and the Neural Engine to songs by priority. Every event (a song arrives, a
    stage ends, a resource frees) calls ``tick``; ``schedule`` holds the entire policy."""

    def __init__(self):
        self.cv = threading.Condition()
        self.songs = []                           # live processes, any state before done/failed/cancelled
        self.token_batch = None                   # songs tokenizing together on the GPU
        self.gpu_synth = None                     # song synthesizing on the GPU
        self.ane_synth = None                     # song on the Neural Engine (or granted it)
        self.remote_synth = None                  # song on the iPhone (or granted it)
        self.render = None                        # song rendering on the GPU
        self.ane_prefill = False                  # the Neural Engine song is briefly on the GPU (prefill)

    # -- bookkeeping ---------------------------------------------------------
    def submit(self, songs):
        with self.cv:
            self.songs.extend(songs)
        for s in songs:
            s.set_state(s.state, "waiting for the GPU" if s.state == QUEUED else "waiting")
        self.tick()

    def finish(self, song, state, detail="", **extra):
        with self.cv:
            if song not in self.songs:
                return
            self.songs.remove(song)
            for slot in ("gpu_synth", "ane_synth", "remote_synth", "render"):
                if getattr(self, slot) is song:
                    setattr(self, slot, None)
            empty = not self.songs
        song.set_state(state, detail, **extra)
        if empty:
            self.drained()
        else:
            self.tick()

    def drained(self):
        LAST_ACTIVE[0] = time.time()
        with MODEL_LOCK:
            with self.cv:
                if self.songs:
                    return
            try:
                with TORCH_LOCK:
                    release()
            except Exception as exc:
                log(f"Memory release failed: {exc}")
        emit(event="idle")

    def cancel(self, song, reason="cancelled"):
        song.cancel.set()
        if song.state in (QUEUED, SYNTH_WAIT, RENDER_WAIT):
            self.finish(song, CANCELLED, reason)         # waiting: gone at once
        else:
            log(f"Cancelling {song.label} ({song.state})")   # running: its thread drops it at the next check

    def stop(self):
        with self.cv:
            songs = sorted(self.songs, key=lambda s: s.priority)
        log(f"Stop requested: cancelling {len(songs)} song(s)")
        for s in songs:
            self.cancel(s, "stopped")

    def find(self, path):
        with self.cv:
            for s in self.songs:
                if s.path == path or str(s.directory) == path:
                    return s
        return None

    def tick(self):
        with self.cv:
            starts = self.schedule()
        for target, arg in starts:
            threading.Thread(target=target, args=(arg,), daemon=True).start()

    # -- the policy ------------------------------------------------------------
    def schedule(self):
        """Decide what starts now. Returns the worker functions to run in threads."""
        live = sorted(self.songs, key=lambda s: s.priority)
        def first(state, ok=lambda s: True):
            return next((s for s in live if s.state == state and not s.cancel.is_set() and ok(s)), None)
        running = any(x is not None for x in (self.token_batch, self.gpu_synth, self.ane_synth, self.remote_synth, self.render))
        starts = []

        # Neural Engine: the highest-priority song that wants it, whether waiting or already on the GPU.
        if self.ane_synth is None and (CONCURRENT or not running):
            waiting = first(SYNTH_WAIT, lambda s: s.wants_ane)
            g = self.gpu_synth
            movable = g if (g is not None and g.wants_ane and g.program_ready.is_set() and not g.migrate and not g.cancel.is_set()) else None
            best = min((s for s in (waiting, movable) if s is not None), key=lambda s: s.priority, default=None)
            if best is not None and best is movable:
                best.migrate = "ane"; self.ane_synth = best         # picked up at its next step boundary
            elif best is not None:
                self.ane_synth = best; best.state = SYNTHING
                starts.append((run_ane, best))
                running = True

        # iPhone: the next song allowed a Neural Engine that the phone can hold, waiting or on the GPU.
        if REMOTE is not None and self.remote_synth is None and (CONCURRENT or not running):
            waiting = first(SYNTH_WAIT, lambda s: remote_can_take(s) and s is not self.ane_synth)
            g = self.gpu_synth
            movable = g if (g is not None and remote_can_take(g) and not g.migrate and not g.cancel.is_set() and g is not self.ane_synth) else None
            best = min((s for s in (waiting, movable) if s is not None), key=lambda s: s.priority, default=None)
            if best is not None and best is movable:
                best.migrate = "remote"; self.remote_synth = best
            elif best is not None:
                self.remote_synth = best; best.state = SYNTHING
                starts.append((run_remote, best))
                running = True

        # Rendering: always admitted (tiles interleave with token steps; a GPU synthesis pauses for it).
        if self.render is None and (CONCURRENT or not running):
            s = first(RENDER_WAIT)
            if s is not None:
                self.render = s; s.state = RENDERING
                starts.append((run_render, s))
                running = True

        # GPU: the highest-priority song that needs it decides between tokenizing and synthesis.
        q = first(QUEUED)
        cand = first(SYNTH_WAIT, lambda s: not s.wants_ane or self.ane_synth is not None) if self.gpu_synth is None else None
        if q is not None and cand is not None and cand.priority < q.priority:
            q = None                                            # the synthesis candidate outranks the queue
        # Tokenizing: the top queued song and every queued song of the same kind with it. It also
        # starts over a lower-priority GPU synthesis, which then pauses (preemption).
        if q is not None and self.token_batch is None and (CONCURRENT or not running) \
                and (self.gpu_synth is None or q.priority < self.gpu_synth.priority):
            batch = [s for s in live if s.state == QUEUED and not s.cancel.is_set()
                     and s.mode == q.mode and s.needs_plan == q.needs_plan][:MAX_BATCH]
            self.token_batch = batch
            for s in batch:
                s.state = PLANNING if s.needs_plan else TOKENIZING
            starts.append((run_batch, batch))
            running = True
        # Synthesis on the GPU: drafts, and full-quality songs while the Neural Engine is busy (they
        # move across when granted). Only when no batch runs and no queued song outranks it.
        elif cand is not None and q is None and self.token_batch is None and (CONCURRENT or not running):
            self.gpu_synth = cand; cand.state = SYNTHING
            starts.append((run_gpu, cand))
        return starts

    def gpu_wanted_elsewhere(self):
        """A GPU synthesis pauses while this is true (tokenizing, rendering, or the Neural Engine
        song's prefill are on the GPU: MLX's long kernels would starve them)."""
        return self.token_batch is not None or self.render is not None or self.ane_prefill

    def summary(self):
        with self.cv:
            return {"queued": sum(s.state == QUEUED for s in self.songs), "live": len(self.songs)}


SCHED = Scheduler()


# ── Workers (one thread each, started by the scheduler) ───────────────────────

def fail(song, exc):
    traceback.print_exc(file=sys.stderr)
    log(f"{song.label} failed: {type(exc).__name__}: {exc}")
    emit(event="failed", path=song.path, message=f"{type(exc).__name__}: {exc}")
    SCHED.finish(song, FAILED, str(exc)[:200])


def save_tokens(song):
    """Persist the plan and song tokens as soon as they exist, so a song can be synthesized later
    (render command) even if the worker stops before its audio is written."""
    import numpy as np
    from yue2.storage import write_json
    song.directory.mkdir(parents=True, exist_ok=True)
    song.plan.save(song.directory)
    np.save(song.directory / "semantic.npy", np.asarray(song.codec, dtype=np.int32))
    write_json(song.directory / "request.json", song.request.to_dict())
    write_json(song.directory / "tokens.json", {"seed": song.seed, "frames": len(song.codec), "quality": song.quality,
                                               "steps": song.steps, "priority": song.priority, "truncated": bool(song.truncated),
                                               "title": song.title, "timing": song.timing})


def run_batch(batch):
    """Plan (if needed) and tokenize a batch of songs together on the GPU."""
    import dataclasses
    from yue2.pipeline import SymbolicPlan
    from yue2.protocol import CODEC_OFFSET, token_prefixes
    songs = list(batch)
    try:
        pipe, model = acquire_model()
        # The GPU is ours now: take along any songs of the same kind that queued meanwhile (a job
        # submitted during the model load, or while the GPU was busy) up to the batch limit.
        with SCHED.cv:
            if SCHED.token_batch is batch:
                for s in sorted(SCHED.songs, key=lambda s: s.priority):
                    if len(songs) >= MAX_BATCH:
                        break
                    if s.state == QUEUED and not s.cancel.is_set() and s.mode == songs[0].mode and s.needs_plan == songs[0].needs_plan and s not in songs:
                        s.state = PLANNING if s.needs_plan else TOKENIZING
                        songs.append(s)
                SCHED.token_batch = songs
        n = len(songs)
        tokenizer = pipe.tokenizer
        cancelled = lambda: all(s.cancel.is_set() for s in songs)
        log(f"Tokenizing batch: {[s.label for s in songs]} (priorities {[s.priority for s in songs]})")
        counts = [0] * n; last = [0.0]

        def reporter(phase, expected, prefix_len):
            """Per-song progress and throughput: a batch step costs the same however many rows still
            produce, so each row's figure is its own tokens per second."""
            from yue2.protocol import ABC_END, MUSIC_END
            end = ABC_END if phase == "abc" else MUSIC_END
            rates, reported, finished = [Rate() for _ in songs], [0] * n, [False] * n
            for r in rates:
                r.add(0)
            def on_token(row, _phase, token):
                counts[row] += 1
                if token == end or (phase == "semantic" and counts[row] >= songs[row].limit):
                    finished[row] = True
                if time.perf_counter() - last[0] > 0.5 or finished[row]:
                    last[0] = time.perf_counter()
                    for i, s in enumerate(songs):
                        if finished[i]:
                            if reported[i] >= 0:
                                s.progress(1.0, "tokens finished"); reported[i] = -1
                            continue
                        gflops = rates[i].add((counts[i] - reported[i]) * flops_per_token(model.config, prefix_len + counts[i]))
                        reported[i] = counts[i]
                        detail = f"{counts[i]} score tokens" if phase == "abc" else f"{counts[i]} tokens (about {counts[i] / 25:.0f} s of audio)"
                        s.progress(min(1.0, counts[i] / expected), detail, gflops)
            return on_token

        requests = [s.request for s in songs]
        if songs[0].needs_plan:
            for s in songs:
                s.set_state(PLANNING, "planning the score")
            prefixes = [token_prefixes(r, tokenizer) for r in requests]
            rows, timing = generate_tokens(model, prefixes, pipe.generation_config.abc, [s.seed for s in songs], "abc",
                                                   cancelled=cancelled, on_token=reporter("abc", 900, max(len(p) for p in prefixes)),
                                                   lock=TORCH_LOCK)
            from yue2.pipeline import complete_score_ids
            rows = [(complete_score_ids(r, tokenizer, ids), t, trunc) for r, (ids, t, trunc) in zip(requests, rows)]
            plans = [SymbolicPlan(r, tokenizer.decode(ids), ids, token_prefixes(r, tokenizer, ids), t, trunc)
                     for r, (ids, t, trunc) in zip(requests, rows)]
            log(f"Scores planned: {[len(p.abc_ids) for p in plans]} tokens in {timing['seconds']:.0f} s")
            if any(s.instrumental for s in songs):
                # Re-plan from each score with its vocal voice silenced: the tokens then carry no sung melody.
                from yue2.instrumental import silence_vocals
                plans = [pipe.plan(request=dataclasses.replace(p.request, abc=silence_vocals(p.abc), abc_open=False)) if (s.instrumental and p.abc) else p
                         for s, p in zip(songs, plans)]
                log("Instrumental: vocal voice silenced in the planned score(s)")
        else:
            plans = [pipe.plan(request=r) for r in requests]
        counts[:] = [0] * n
        for s in songs:
            s.set_state(TOKENIZING, "generating song tokens")
        limits = [s.limit for s in songs]
        sampling = dataclasses.replace(pipe.generation_config.semantic, max_tokens=max(limits))

        released = set()
        def release_song(i, tokens, t, truncated):
            s, plan = songs[i], plans[i]
            released.add(i)
            s.plan, s.codec, s.timing, s.truncated = plan, [int(x) - CODEC_OFFSET for x in tokens], t, truncated
            s.decide_engine()
            save_tokens(s)
            if s.cancel.is_set():
                SCHED.finish(s, CANCELLED, "stopped"); return
            log(f"Song tokens for {s.label}: {len(tokens)} ({t['seconds']:.0f} s)" + (", truncated" if truncated else "")
                + ("; Neural Engine" if s.wants_ane else "; GPU"))
            s.set_state(SYNTH_WAIT, "waiting", engine="ane" if s.wants_ane else "mlx")
            SCHED.tick()                                   # synthesis can start while the batch continues
        def on_row_done(i, tokens, t):
            if i not in released:
                release_song(i, tokens, t, bool(t.get("truncated", False)))

        rows, timing = generate_tokens(model, [p.prefix for p in plans], sampling, [s.seed for s in songs], "semantic",
                                               legacy_off=(songs[0].mode == "off"), cancelled=cancelled,
                                               on_token=reporter("semantic", max(limits), max(len(p.prefix) for p in plans)),
                                               lock=TORCH_LOCK, on_row_done=on_row_done, limits=limits)
        log(f"Batch done: {[len(r[0]) for r in rows]} tokens in {timing['seconds']:.0f} s ({1000 * (timing['mean_step_seconds'] or 0):.0f} ms per step)")
        for i, (tokens, t, truncated) in enumerate(rows):
            if i in released:
                songs[i].timing = t
            else:
                release_song(i, tokens, t, truncated)
    except InterruptedError:
        for s in songs:
            if s.state in (PLANNING, TOKENIZING):
                SCHED.finish(s, CANCELLED, "stopped")
    except Exception as exc:
        for s in songs:
            if s.state in (PLANNING, TOKENIZING):
                fail(s, exc)
    finally:
        with SCHED.cv:
            SCHED.token_batch = None
        try:
            trim_gpu_memory(model, "batch finished")
        except Exception:
            pass
        SCHED.tick()


def synth_common(song, model, pipe):
    """Progress, throughput and phase callbacks shared by both synthesis resources."""
    audio_s = song.frames / 25
    S, P = song.frames + 2, len(song.plan.prefix) + song.frames + 1
    per_step = 2 * flops_per_pass(model.config, S, P)               # midpoint solver: two passes per step
    rate, seen = Rate(), [0]
    def on_nar(done, total):
        if seen[0] == 0:
            rate.last = time.perf_counter(); gflops = None            # the first step also carried the prefill/compile: skip it
        else:
            gflops = rate.add((done - seen[0]) * per_step)
        seen[0] = done
        song.progress(done / max(total, 1), f"solver step {done}/{total}", gflops)
    def on_prepare(done, total):
        song.progress(0.0, f"compiling Neural Engine program {done}/{total}")
        if done in (1, total):
            log(f"Neural Engine program {done}/{total} ready for {song.label}")
    return dict(steps=song.steps, context=pipe.generation_config.context, cancelled=song.cancel.is_set,
                on_progress=on_nar, on_prepare=on_prepare, lock=TORCH_LOCK), audio_s, (S, P)


WARMING = threading.Lock()


def precompile(song, model, why):
    """Compile a song's Neural Engine program in the background; sets song.program_ready."""
    from yue2.ane import runtime as ane_runtime
    S, P = song.frames + 2, len(song.plan.prefix) + song.frames + 1
    bucket = ane_runtime.buckets_for((S, P))
    programs = ane_runtime.programs_for(model)
    if bucket in programs.loaded:
        song.program_ready.set(); return
    def work():
        with WARMING:                                 # the compiler service is single-threaded anyway
            try:
                if bucket not in programs.loaded:
                    log(f"Background: compiling Neural Engine programs for {song.label} (bucket {bucket[0]} x {bucket[1]}) {why}")
                    t0 = time.perf_counter()
                    programs.precompile(*bucket)
                    log(f"Background: Neural Engine programs for {song.label} ready in {time.perf_counter() - t0:.0f} s")
            except Exception as exc:
                song.program_failed = True
                log(f"{song.label}: the Neural Engine cannot compile its program ({brief(exc)})")
                log_full_error("Neural Engine compiler said", exc)
            finally:
                song.program_ready.set()
                SCHED.tick()
    threading.Thread(target=work, daemon=True).start()


def run_ane(song):
    """Synthesize on the Neural Engine."""
    from yue2.nar import synthesize
    try:
        pipe, model = acquire_model()
        kwargs, audio_s, _ = synth_common(song, model, pipe)
        log(f"Synthesizing {song.label} on the Neural Engine: about {audio_s:.0f} s of audio, {song.quality} quality ({song.steps} steps)")
        song.set_state(SYNTHING, "preparing", engine="ane")
        warmed = [False]
        on_nar = kwargs["on_progress"]
        def on_nar_warm(done, total):
            if not warmed[0]:                          # solving: compile the next Neural Engine song's program meanwhile
                warmed[0] = True
                with SCHED.cv:
                    nxt = next((s for s in sorted(SCHED.songs, key=lambda s: s.priority)
                                if s.state == SYNTH_WAIT and s.wants_ane and not s.program_ready.is_set()), None)
                if nxt is not None:
                    precompile(nxt, model, "while the current song solves")
            on_nar(done, total)
        def on_phase(text):
            song.progress(0.0, text)
            with SCHED.cv:
                SCHED.ane_prefill = text.startswith("prefilling")
            SCHED.tick()
        kwargs.update(on_progress=on_nar_warm, on_phase=on_phase)
        song.used_engine = "ane"
        t0 = time.perf_counter()
        try:
            latents = synthesize(model, song.plan.prefix, song.codec, song.seed, engine="ane", offload_ar=pipe.offload_ar, **kwargs)
        except RuntimeError as exc:
            if not str(exc).startswith("compile"):
                raise
            # The compiler rejects some very large shapes; the song goes back to wait for the GPU.
            log(f"The Neural Engine cannot compile programs for {song.label} ({brief(exc)}); it will use the GPU")
            log_full_error("Neural Engine compiler said", exc)
            song.wants_ane = False; song.program_failed = True
            with SCHED.cv:
                SCHED.ane_synth = None; SCHED.ane_prefill = False
            song.set_state(SYNTH_WAIT, "waiting for the GPU", engine="mlx")
            SCHED.tick(); return
        song.latents = latents.detach().float().cpu().numpy()
        song.nar_seconds = time.perf_counter() - t0
        log(f"Synthesis of {song.label} done in {song.nar_seconds:.0f} s ({song.nar_seconds / audio_s:.1f} s per second of audio, "
            f"{song.steps} steps, ane)")
        with SCHED.cv:
            SCHED.ane_synth = None; SCHED.ane_prefill = False
        if song.cancel.is_set():
            SCHED.finish(song, CANCELLED, "stopped"); return
        song.set_state(RENDER_WAIT, "waiting", engine="ane")
        SCHED.tick()
    except InterruptedError:
        with SCHED.cv:
            SCHED.ane_synth = None; SCHED.ane_prefill = False
        SCHED.finish(song, CANCELLED, "stopped")
    except Exception as exc:
        with SCHED.cv:
            SCHED.ane_synth = None; SCHED.ane_prefill = False
        fail(song, exc)


def run_remote(song):
    """Synthesize on the iPhone's Neural Engine (the prefix prefill still runs on the GPU)."""
    from yue2.nar import synthesize
    from yue2.remote.protocol import RemoteError
    client = REMOTE
    def release():
        with SCHED.cv:
            if SCHED.remote_synth is song:
                SCHED.remote_synth = None
            SCHED.ane_prefill = False
    try:
        pipe, model = acquire_model()
        kwargs, audio_s, _ = synth_common(song, model, pipe)
        if client is None:
            raise RemoteError("the iPhone is no longer connected")
        log(f"Synthesizing {song.label} on {client.label()}: about {audio_s:.0f} s of audio, {song.quality} quality ({song.steps} steps)")
        song.set_state(SYNTHING, "preparing", engine="remote")
        def on_phase(text):
            song.progress(0.0, text)
            with SCHED.cv:
                SCHED.ane_prefill = text.startswith("prefilling")
            SCHED.tick()
        kwargs["on_phase"] = on_phase
        song.used_engine = "remote"
        t0 = time.perf_counter()
        try:
            latents = synthesize(model, song.plan.prefix, song.codec, song.seed, engine="remote", remote=client,
                                 offload_ar=pipe.offload_ar, **kwargs)
        except (RemoteError, OSError) as exc:
            reason = brief(exc, 120)                  # the song row and the remote status show this line
            log_full_error(f"{client.label()} said", exc)
            if isinstance(exc, OSError) or "closed the connection" in reason:
                clear_remote(reason)
            else:
                song.remote_failed = True
                log(f"{client.label()} cannot run {song.label} ({reason}); it will use the Mac")
            release()
            song.set_state(SYNTH_WAIT, "waiting", engine="ane" if song.wants_ane else "mlx")
            SCHED.tick(); return
        song.latents = latents.detach().float().cpu().numpy()
        song.nar_seconds = time.perf_counter() - t0
        log(f"Synthesis of {song.label} done in {song.nar_seconds:.0f} s ({song.nar_seconds / audio_s:.1f} s per second of audio, "
            f"{song.steps} steps, {client.label()})")
        release()
        if song.cancel.is_set():
            SCHED.finish(song, CANCELLED, "stopped"); return
        song.set_state(RENDER_WAIT, "waiting", engine="remote")
        SCHED.tick()
    except InterruptedError:
        release()
        SCHED.finish(song, CANCELLED, "stopped")
    except Exception as exc:
        release()
        fail(song, exc)


def run_gpu(song):
    """Synthesize on the GPU (MLX). A full-quality song moves to the Neural Engine when the
    scheduler grants it, and pauses between steps whenever the GPU is wanted elsewhere."""
    from yue2.nar_switch import synthesize_switchable
    try:
        pipe, model = acquire_model()
        kwargs, audio_s, _ = synth_common(song, model, pipe)
        kwargs["on_phase"] = lambda text: song.progress(0.0, text)
        song.used_engine = "mlx"
        t0 = time.perf_counter()
        movable = song.wants_ane or remote_can_take(song)
        where = "the Neural Engine" if song.wants_ane else "the iPhone"
        if movable:
            log(f"Synthesizing {song.label} on the GPU until {where} is granted: about {audio_s:.0f} s of audio ({song.steps} steps)")
            song.set_state(SYNTHING, f"on the GPU · moves to {where} when granted", engine="mlx")
            if song.wants_ane:
                precompile(song, model, "for the move")
            SCHED.tick()                                   # the program may already be compiled
        else:
            log(f"Synthesizing {song.label} on the GPU: about {audio_s:.0f} s of audio, {song.quality} quality ({song.steps} steps)")
            song.set_state(SYNTHING, "preparing", engine="mlx")
        idle_text = f"on the GPU · moves to {where} when granted" if movable else "on the GPU"
        paused = [False]
        def should_wait():
            # Tokenizing, rendering and the Neural Engine song's prefill own the GPU: MLX's long
            # kernels would starve them, so this song sits out until they are done.
            busy = SCHED.gpu_wanted_elsewhere()
            if busy != paused[0]:
                paused[0] = busy
                if busy:
                    what = "tokenizing" if SCHED.token_batch is not None else "rendering" if SCHED.render is not None else "prefilling for the Neural Engine"
                    log(f"{song.label} pauses: the GPU is {what}")
                    song.set_state(SYNTHING, f"paused · the GPU is busy {what}", engine="mlx")
                else:
                    log(f"{song.label} resumes on the GPU")
                    song.set_state(SYNTHING, idle_text, engine="mlx")
            return busy
        def on_switch(step, kind):
            moved_to = "the Neural Engine" if kind == "ane" else "the iPhone"
            song.migrated = True; song.used_engine = kind
            log(f"{song.label} moved to {moved_to} at solver step {step}/{song.steps}")
            song.set_state(SYNTHING, f"moved to {moved_to} at step {step}", engine=kind)
            with SCHED.cv:
                if SCHED.gpu_synth is song:
                    SCHED.gpu_synth = None                   # the GPU is free for the next song
            trim_gpu_memory(model, f"moved to {moved_to}")
            SCHED.tick()
        latents, used, switched_at = synthesize_switchable(model, song.plan.prefix, song.codec, song.seed,
                                                           may_switch=(lambda: song.migrate) if movable else None,
                                                           should_wait=should_wait, on_switch=on_switch, remote=REMOTE, **kwargs)
        song.used_engine = f"mlx+{used}" if switched_at is not None else "mlx"
        song.latents = latents.detach().float().cpu().numpy()
        song.nar_seconds = time.perf_counter() - t0
        log(f"Synthesis of {song.label} done in {song.nar_seconds:.0f} s ({song.nar_seconds / audio_s:.1f} s per second of audio, "
            f"{song.steps} steps, {song.used_engine})")
        with SCHED.cv:
            if SCHED.gpu_synth is song:
                SCHED.gpu_synth = None
            if SCHED.ane_synth is song:
                SCHED.ane_synth = None
            if SCHED.remote_synth is song:
                SCHED.remote_synth = None
        trim_gpu_memory(model, "GPU synthesis finished")
        if song.cancel.is_set():
            SCHED.finish(song, CANCELLED, "stopped"); return
        song.set_state(RENDER_WAIT, "waiting", engine=song.used_engine)
        SCHED.tick()
    except InterruptedError:
        SCHED.finish(song, CANCELLED, "stopped")
    except Exception as exc:
        fail(song, exc)


def run_render(song):
    """Decode the latents to a waveform on the GPU and write the song's files."""
    from yue2.pipeline import SemanticResult, SongResult
    from yue2.storage import identity
    try:
        pipe, _ = acquire_model()
        song.set_state(RENDERING, "decoding waveform", engine=song.used_engine)
        rate, seen = Rate(), [0]
        rate.add(0)
        def on_tile(done, total):
            gflops = rate.add((done - seen[0]) * pipe.vae_core_frames * VAE_GFLOP_PER_FRAME * 1e9); seen[0] = done
            song.progress(done / max(total, 1), f"decoding tile {done}/{total}", gflops)
            if song.cancel.is_set():
                raise InterruptedError("stopped")
            TORCH_LOCK.yield_turn()                  # let a waiting token step or prefill in between tiles
        t1 = time.perf_counter()
        with TORCH_LOCK:
            audio = pipe.decode(song.latents, on_progress=on_tile)
        if song.cancel.is_set():
            raise InterruptedError("stopped")
        plan = song.plan
        config = pipe.effective_config(plan.request)
        config.update({"execution": "eager_batched", "nar_engine": song.used_engine, "ode_steps": song.steps, "quality": song.quality})
        semantic = SemanticResult(plan, song.codec, song.timing or {}, song.truncated)
        result_song = SongResult(audio, 48000, semantic, song.latents, config, pipe.weights,
                                 {"semantic": song.timing or {}, "nar_seconds": song.nar_seconds, "vae_seconds": time.perf_counter() - t1},
                                 identity({"request": plan.request.to_dict(), "config": config, "weights": pipe.weights}))
        directory = song.directory
        if song.quality == "full" and (directory / "audio.flac").exists() and \
                json.loads((directory / "result.json").read_text()).get("quality") == "draft":
            (directory / "audio.flac").replace(directory / "draft.flac")      # keep the preview beside the final render
        result = result_song.save_artifacts(directory)
        result.update({"quality": song.quality, "ode_steps": song.steps, "nar_engine": song.used_engine, "priority": song.priority,
                       "title": song.title})
        (directory / "result.json").write_text(json.dumps(result, indent=2))
        length = len(audio) / 48000
        log(f"Saved {directory / 'audio.flac'} ({length:.1f} s, {song.quality})")
        emit(event="song", index=song.index, path=song.path, score=plan.abc or "", seconds=round(length, 1), seed=song.seed,
             truncated=bool(song.truncated or plan.truncated), quality=song.quality, steps=song.steps, engine=song.used_engine,
             title=song.title)
        SCHED.finish(song, DONE)
    except InterruptedError:
        SCHED.finish(song, CANCELLED, "stopped")
    except Exception as exc:
        fail(song, exc)
    finally:
        song.latents = None


# ── Requests ─────────────────────────────────────────────────────────────────

STAMPS_LOCK = threading.Lock()
STAMPS = set()                                            # run folders handed out this session


def slug(title, limit=40):
    """A filename-safe form of a title for the run folder: letters, digits and single dashes."""
    import re, unicodedata
    text = unicodedata.normalize("NFKD", title).encode("ascii", "ignore").decode()
    text = re.sub(r"[^A-Za-z0-9]+", "-", text).strip("-")
    return text[:limit].rstrip("-")


def steps_for(quality, req):
    if quality == "draft":
        return max(1, min(int(req.get("draft_steps", 8)), 32))
    from yue2.protocol import GenerationConfig
    return (PIPE.generation_config if PIPE is not None else GenerationConfig()).ode_steps


def submit_generate(req):
    from yue2.protocol import SongRequest
    n = int(req.get("batch", 1)); mode = req.get("cot", "full")
    style, lyrics = req["style"].strip(), req["lyrics"].strip()
    instrumental = bool(req.get("instrumental"))
    if instrumental:
        from yue2.instrumental import instrumental_tags, structure_only
        style, lyrics = instrumental_tags(style), structure_only(lyrics)
        if mode == "off":
            mode = "full"                      # the vocal voice can only be silenced in a planned score
    quality = "draft" if req.get("quality", "draft") == "draft" else "full"
    allow_ane = req.get("engines", "gpu+ane" if quality == "full" else "gpu") != "gpu"
    base = int(time.time()) % 10_000_000 if req.get("random_seed") else int(req.get("seed", 831001))
    seeds = [base + i for i in range(n)]
    abc = (req.get("abc") or "").strip() or None
    abc_open = bool(req.get("abc_open")) and abc is not None
    if abc_open:
        from yue2.hum import hum_opening
        abc = hum_opening(abc)                       # the hummed line as the Vocal voice, ending on its last note
    title = " ".join(str(req.get("title", "")).split())[:120]
    with STAMPS_LOCK:                                    # two jobs submitted in the same second must not share a folder
        stamp = dt.datetime.now().strftime("%Y%m%d-%H%M%S") + (f"-{slug(title)}" if slug(title) else "")
        out_root = OUTPUT_DIR / stamp
        with SCHED.cv:
            taken = {s.directory.parent for s in SCHED.songs} | STAMPS
        while out_root in taken or out_root.exists():
            stamp += "b"; out_root = OUTPUT_DIR / stamp
        STAMPS.add(out_root)
    steps = steps_for(quality, req)
    limit = max(1, min(int(req.get("max_tokens", 9000)), 9000))
    songs = []
    for i, seed in enumerate(seeds):
        request = SongRequest(style=style, lyrics=lyrics, cot=mode, seed=seed, abc=abc, abc_open=abc_open,
                              id=f"song{i + 1}", **({"cfg_scale": 1.0} if mode == "off" else {}))
        songs.append(Song(stamp, i + 1, seed, request, out_root / f"song{i + 1}", quality, steps, limit, instrumental, allow_ane, title))
    emit(event="started", job=stamp, output=str(out_root), title=title,
         songs=[{"index": s.index, "seed": s.seed, "path": s.path, "priority": s.priority} for s in songs])
    log(f"Queued {stamp}: {n} song(s), {quality} quality ({steps} steps, {'GPU + Neural Engine' if allow_ane else 'GPU only'}), "
        f"seeds {seeds}, priorities {[s.priority for s in songs]}" + (", instrumental" if instrumental else ""))
    SCHED.submit(songs)


def submit_render(req):
    """Synthesize a song from its saved tokens: a full-quality render of a draft, or a song whose
    tokens were saved but never synthesized. Same tokens, seed and noise."""
    import numpy as np
    from yue2.pipeline import SymbolicPlan
    directory = Path(req["path"])
    if not directory.is_dir():
        # The app addresses songs by their audio.flac path even when the file
        # doesn't exist yet (a stalled song) — strip to the directory either way.
        directory = directory.parent
    if SCHED.find(str(directory / "audio.flac")) is not None:
        emit(event="error", message=f"{directory.name} is already queued"); return
    quality = "draft" if req.get("quality", "full") == "draft" else "full"
    allow_ane = req.get("engines", "gpu+ane" if quality == "full" else "gpu") != "gpu"
    plan = SymbolicPlan.load(directory)
    codec = np.load(directory / "semantic.npy", allow_pickle=False).astype(int).tolist()
    previous = {}
    for name in ("result.json", "tokens.json"):
        if (directory / name).exists():
            previous = json.loads((directory / name).read_text()); break
    truncated = previous.get("truncated")
    truncated = bool(truncated.get("semantic", False) if isinstance(truncated, dict) else truncated)
    index = int(directory.name[4:]) if directory.name.startswith("song") and directory.name[4:].isdigit() else 1
    song = Song(directory.parent.name, index, plan.request.seed, plan.request, directory, quality, steps_for(quality, req), 9000,
                allow_ane=allow_ane, title=str(previous.get("title", "")))
    song.plan, song.codec, song.truncated = plan, codec, truncated
    song.timing = previous.get("timing", {}) if "frames" in previous else {}
    song.decide_engine()
    song.state = SYNTH_WAIT
    emit(event="started", job=directory.parent.name, output=str(directory.parent), title=song.title,
         songs=[{"index": index, "seed": song.seed, "path": song.path, "priority": song.priority}])
    log(f"Queued {song.label} for {quality} synthesis ({song.steps} steps, {'GPU + Neural Engine' if allow_ane else 'GPU only'}, "
        f"priority {song.priority}): about {len(codec) / 25:.0f} s of audio")
    SCHED.submit([song])


def submit(req):
    try:
        (submit_render if req.get("cmd") == "render" else submit_generate)(req)
    except Exception as exc:
        traceback.print_exc(file=sys.stderr)
        log(f"Error: {type(exc).__name__}: {exc}"); emit(event="error", message=f"{type(exc).__name__}: {exc}")
        # The app marks the row queued before sending — flip it to failed so a
        # submit that never reached the scheduler doesn't sit on "Queued" forever.
        emit(event="failed", path=req.get("path", ""), message=f"{type(exc).__name__}: {exc}")


# ── Transcription (SheetSage2) ───────────────────────────────────────────────

def run_transcribe(req):
    """Run SheetSage2 in its own environment and forward its JSON progress lines."""
    rid = req.get("id", "")
    if not TRANSCRIBE_LOCK.acquire(blocking=False):
        emit(event="transcribe", id=rid, stage="failed", code="busy",
             message="a transcription is already running")
        return
    try:
        python = Path(os.environ.get("YUE2_SHEETSAGE_PYTHON")
                      or ROOT / ".venv-sheetsage2" / "bin" / "python")
        if not python.is_file():
            emit(event="transcribe", id=rid, stage="failed", code="no_env",
                 message="SheetSage2 environment not installed")
            return
        tool = Path(os.environ.get("YUE2_TRANSCRIBE_TOOL")
                    or Path(__file__).with_name("transcribe_sheetsage.py"))
        audio = Path(req.get("audio", ""))
        stamp = dt.datetime.now().strftime("%Y%m%d-%H%M%S")
        out = OUTPUT_DIR / "transcriptions" / f"{audio.stem}-{stamp}"   # song rescans never look here
        cmd = [str(python), "-u", str(tool), str(audio), "--output", str(out),
               "--task", req.get("task", "melody-full"), "--device", "cpu", "--dtype", "fp32",
               "--threads", str(min(8, os.cpu_count() or 4))]
        if req.get("offline"):
            cmd.append("--offline")
        log(f"Transcribing '{audio.name}' with SheetSage2")
        proc = subprocess.Popen(cmd, stdout=subprocess.PIPE, text=True)   # stderr inherited
        TRANSCRIBE_PROC[0] = proc
        settled = False
        for line in proc.stdout:
            line = line.strip()
            if not line:
                continue
            try:
                obj = json.loads(line)
            except json.JSONDecodeError:
                log(line)
                continue
            if obj.get("stage") in ("done", "failed"):
                settled = True
            emit(event="transcribe", id=rid, **obj)
        code = proc.wait()
        if code < 0:                                  # killed by transcribe_cancel
            emit(event="transcribe", id=rid, stage="cancelled")
            log(f"Transcription of '{audio.name}' cancelled")
        elif not settled:
            emit(event="transcribe", id=rid, stage="failed", code="crash",
                 message=f"transcriber exited with status {code}")
    except Exception as exc:
        traceback.print_exc(file=sys.stderr)
        emit(event="transcribe", id=rid, stage="failed", code="crash",
             message=f"{type(exc).__name__}: {exc}")
    finally:
        TRANSCRIBE_PROC[0] = None
        TRANSCRIBE_LOCK.release()


def cancel_transcribe():
    proc = TRANSCRIBE_PROC[0]
    if proc is None:
        emit(event="error", message="no transcription running")
        return
    proc.terminate()

    def kill_after_grace(p=proc):
        try:
            p.wait(3)
        except subprocess.TimeoutExpired:
            p.kill()

    threading.Thread(target=kill_after_grace, daemon=True).start()


# ── Memory ───────────────────────────────────────────────────────────────────

def footprint_mb():
    """Physical footprint of this process (macOS), or None."""
    try:
        import subprocess
        out = subprocess.run(["/usr/bin/footprint", "-p", str(os.getpid())], capture_output=True, text=True, timeout=10).stdout
        for line in out.splitlines():
            if "phys_footprint:" in line:
                value, unit = line.split()[1:3]
                return float(value) * {"KB": 1 / 1024, "MB": 1, "GB": 1024}.get(unit, 1)
    except Exception:
        return None


def release(deep=False):
    """Free GPU/engine memory held between jobs: MLX weights and cache, ANE weight surfaces and
    program mappings, the VAE and the MPS cache. deep=True also drops the model (lean reload ~1 s)
    and the compiled ANE programs. Call under MODEL_LOCK with no live songs."""
    global PIPE
    import gc, torch
    before = footprint_mb()
    if PIPE is not None and PIPE._model is not None:
        model = PIPE._model
        model._yue2_mlx_weights = None
        model._yue2_ar_mlx = None
        programs = getattr(model, "_yue2_ane_programs", None)
        if programs is not None:                 # unmap programs before freeing the surfaces they bind
            with programs.lock:
                for key in list(programs.loaded):
                    for program in programs.loaded[key]:
                        if deep:
                            program.free()
                        elif program.resident:
                            program.unload()
                if deep:
                    programs.loaded.clear()
        weights = getattr(model, "_yue2_ane_weights", None)
        if weights is not None:
            weights.close(); model._yue2_ane_weights = None
        PIPE._vae = None
        if deep:
            PIPE._model = None
        del model, programs, weights             # no local may keep the model alive through the collection below
    try:
        import mlx.core as mx
        (getattr(mx, "clear_cache", None) or mx.metal.clear_cache)()
    except Exception:
        pass
    gc.collect()
    if torch.backends.mps.is_available():
        torch.mps.empty_cache()
    after = footprint_mb()
    if before is not None and after is not None:
        log(f"{'Unloaded the model' if deep else 'Released working memory'}: {before / 1024:.1f} GB -> {after / 1024:.1f} GB")


def idle_watch():
    while True:
        time.sleep(30)
        if PIPE is not None and PIPE._model is not None and time.time() - LAST_ACTIVE[0] > IDLE_UNLOAD_S:
            with MODEL_LOCK:
                if SCHED.songs or time.time() - LAST_ACTIVE[0] <= IDLE_UNLOAD_S or PIPE._model is None:
                    continue
                with TORCH_LOCK:
                    release(deep=True)


# ── Main loop ────────────────────────────────────────────────────────────────

def main():
    emit(event="ready", root=str(ROOT), concurrent=CONCURRENT, max_batch=MAX_BATCH)
    threading.Thread(target=idle_watch, name="idle_watch", daemon=True).start()
    for line in sys.stdin:
        line = line.strip()
        if not line:
            continue
        try:
            req = json.loads(line)
        except json.JSONDecodeError:
            emit(event="error", message="bad json"); continue
        cmd = req.get("cmd")
        if cmd == "ping":
            emit(event="pong")
        elif cmd == "quit":
            SCHED.stop(); break
        elif cmd == "stop":
            SCHED.stop()
        elif cmd == "cancel":
            song = SCHED.find(str(Path(req.get("path", ""))))
            if song is None:
                emit(event="error", message="not queued")
            else:
                SCHED.cancel(song)
        elif cmd in ("generate", "render"):
            threading.Thread(target=submit, args=(req,), daemon=True).start()     # never block the command loop on a model load
        elif cmd == "remote":
            threading.Thread(target=set_remote, args=(req,), daemon=True).start()
        elif cmd == "transcribe":
            threading.Thread(target=run_transcribe, args=(req,), daemon=True).start()
        elif cmd == "transcribe_cancel":
            cancel_transcribe()
        else:
            emit(event="error", message=f"unknown command {cmd}")


if __name__ == "__main__":
    main()
