"""Run YuE2 acoustic flow matching on the Apple Neural Engine (private route).

One compiled program per decoder layer per length bucket (S NAR rows, P prefix
keys). Layer outputs chain through IOSurfaces with no copies; the tiny input
embedding and output projection run on the CPU in fp32.
"""
from __future__ import annotations
import ctypes, hashlib, json, math, os, threading, time
from pathlib import Path
import numpy as np
import torch

from .mil import MIL_VERSION, NEG, WEIGHT_SHAPES, arrays_from_state, build_program, weight_input_shapes
from ..lean import nar_layer_state

LIB = Path(os.environ.get("YUE2_ANE_LIB", Path(__file__).with_name("libyue2ane.dylib")))
CACHE = Path(os.environ.get("YUE2_ANE_CACHE", Path.home() / ".cache" / "yue2-ane"))
S_STEP, P_STEP = 512, 1024
QBLK, KCHUNK = int(os.environ.get("YUE2_ANE_QBLK", 1024)), int(os.environ.get("YUE2_ANE_KCHUNK", 2048))
# Room for the bridge's error text: an NSError description plus its chain of underlying errors
# (the compiler's reason for a rejection is in there), written whole so the app's log can show it.
ERR_BYTES = 8192
MAX_RESIDENT = int(os.environ.get("YUE2_ANE_RESIDENT", 28))
LAYER_LIMIT = int(os.environ.get("YUE2_ANE_LAYERS", 0))      # diagnostics: run only the first N layers
LAYERS_PER_PROGRAM = int(os.environ.get("YUE2_ANE_GROUP", 2))  # layers per program call
# One program per bucket serves every layer: the weights are program inputs (IOSurfaces) rather than
# baked constants, so a bucket costs one compile instead of one per layer pair (14x less), and the
# engine only holds one program's working arena. YUE2_ANE_WEIGHT_POOL=n keeps only n layers' weight
# surfaces mapped (rewritten before each call) instead of all 28 (~2.8 GB) — a memory fallback.
WEIGHT_POOL = int(os.environ.get("YUE2_ANE_WEIGHT_POOL", 0))


def physical_memory_gib():
    try:
        return float(os.environ.get("YUE2_PHYSICAL_GIB") or os.sysconf("SC_PAGE_SIZE") * os.sysconf("SC_PHYS_PAGES") / 2**30)
    except (ValueError, OSError):
        return 32.0


# Compiled programs of buckets other than the one in use are kept (compiled, unloaded) so a
# batch of songs of different lengths never recompiles; on small machines they are freed instead.
KEEP_BUCKETS = int(os.environ.get("YUE2_ANE_KEEP_BUCKETS", 8 if physical_memory_gib() >= 24 else 1))


def bucket(n, step):
    return max(step, int(math.ceil(n / step)) * step)


def available():
    return LIB.exists() and torch.backends.mps.is_available()


def buckets_for(engine_or_lengths):
    """(S, P) bucket for a CachedNAR engine or a (nar_length, ar_length) pair."""
    n, p = (engine_or_lengths.nar_length, engine_or_lengths.ar_length) if hasattr(engine_or_lengths, "nar_length") else engine_or_lengths
    return bucket(n, S_STEP), bucket(p, P_STEP)


class Lib:
    _instance = None

    def __init__(self):
        if not LIB.exists():
            raise FileNotFoundError(f"{LIB} missing; build with clang (see libyue2ane.m)")
        lib = ctypes.CDLL(str(LIB))
        lib.ane_surface_create.restype = ctypes.c_void_p; lib.ane_surface_create.argtypes = [ctypes.c_size_t]
        lib.ane_surface_write.argtypes = [ctypes.c_void_p, ctypes.c_void_p, ctypes.c_size_t]
        lib.ane_surface_read.argtypes = [ctypes.c_void_p, ctypes.c_void_p, ctypes.c_size_t]
        lib.ane_surface_free.argtypes = [ctypes.c_void_p]
        lib.ane_program_load.restype = ctypes.c_void_p
        lib.ane_program_load.argtypes = [ctypes.c_char_p, ctypes.POINTER(ctypes.c_double), ctypes.c_char_p, ctypes.c_int]
        lib.ane_program_compile.restype = ctypes.c_void_p
        lib.ane_program_compile.argtypes = [ctypes.c_char_p, ctypes.c_int, ctypes.POINTER(ctypes.c_double), ctypes.c_char_p, ctypes.c_int]
        lib.ane_program_eval.restype = ctypes.c_int
        lib.ane_program_eval.argtypes = [ctypes.c_void_p, ctypes.POINTER(ctypes.c_void_p), ctypes.c_int,
                                         ctypes.POINTER(ctypes.c_void_p), ctypes.c_int, ctypes.c_char_p, ctypes.c_int]
        lib.ane_program_free.argtypes = [ctypes.c_void_p]
        lib.ane_program_unload.restype = ctypes.c_int; lib.ane_program_unload.argtypes = [ctypes.c_void_p, ctypes.c_char_p, ctypes.c_int]
        lib.ane_program_reload.restype = ctypes.c_int; lib.ane_program_reload.argtypes = [ctypes.c_void_p, ctypes.c_char_p, ctypes.c_int]
        self.lib = lib

    @classmethod
    def get(cls):
        if cls._instance is None:
            cls._instance = cls()
        return cls._instance


class Surface:
    def __init__(self, nbytes):
        self.lib = Lib.get().lib
        self.nbytes = nbytes
        self.handle = self.lib.ane_surface_create(nbytes)
        if not self.handle:
            raise RuntimeError("IOSurface creation failed")

    def write(self, array):
        a = np.ascontiguousarray(array, dtype=np.float16)
        if a.nbytes != self.nbytes:
            raise ValueError(f"surface {self.nbytes} bytes, array {a.nbytes}")
        self.lib.ane_surface_write(self.handle, a.ctypes.data, a.nbytes)

    def read(self, shape):
        a = np.empty(shape, dtype=np.float16)
        if a.nbytes != self.nbytes:
            raise ValueError("shape does not match surface size")
        self.lib.ane_surface_read(self.handle, a.ctypes.data, a.nbytes)
        return a

    def free(self):
        if self.handle:
            self.lib.ane_surface_free(self.handle); self.handle = None


def input_names(directory):
    """Input names in binding order, parsed from the MIL main signature (coremltools sorts them)."""
    import re
    mil = (Path(directory) / "model.mil").read_text()
    sig = re.search(r"func main<[^>]*>\((.*?)\)\s*\{", mil, re.S).group(1)
    return [m.group(1) for m in re.finditer(r">\s*(\w+)\s*(?:,|$)", sig)]


class Program:
    def __init__(self, directory, load=True):
        """Compile the program; load=False leaves it unloaded (no engine mappings) until reload()."""
        self.lib = Lib.get().lib
        self.inputs = input_names(directory)
        err = ctypes.create_string_buffer(ERR_BYTES)
        seconds = ctypes.c_double(0)
        if os.environ.get("YUE2_ANE_FAIL_COMPILE"):          # test hook: behave like a compiler rejection
            raise RuntimeError("compile: simulated ANECCompile() FAILED")
        self.handle = self.lib.ane_program_compile(str(directory).encode(), 1 if load else 0, ctypes.byref(seconds), err, len(err))
        if not self.handle:
            raise RuntimeError(err.value.decode(errors="replace"))
        self.compile_seconds = seconds.value
        self.resident = bool(load)

    def __call__(self, inputs, outputs):
        """inputs: {name: Surface} bound in the program's declared order; outputs: [Surface]."""
        err = ctypes.create_string_buffer(ERR_BYTES)
        ordered = [inputs[name] for name in self.inputs]
        ins = (ctypes.c_void_p * len(ordered))(*[s.handle for s in ordered])
        outs = (ctypes.c_void_p * len(outputs))(*[s.handle for s in outputs])
        if self.lib.ane_program_eval(self.handle, ins, len(inputs), outs, len(outputs), err, len(err)):
            raise RuntimeError(err.value.decode(errors="replace"))

    def unload(self):
        err = ctypes.create_string_buffer(ERR_BYTES)
        if self.lib.ane_program_unload(self.handle, err, len(err)):
            raise RuntimeError(err.value.decode(errors="replace"))
        self.resident = False

    def reload(self):
        err = ctypes.create_string_buffer(ERR_BYTES)
        if self.lib.ane_program_reload(self.handle, err, len(err)):
            raise RuntimeError(err.value.decode(errors="replace"))
        self.resident = True

    def free(self):
        if self.handle:
            self.lib.ane_program_free(self.handle); self.handle = None


class LayerPrograms:
    """Generates (once, on disk) and compiles (once per process) the shared program of a bucket.

    The program computes LAYERS_PER_PROGRAM layers with their weights as inputs; the 28 layers run
    as 14 calls of the same program with different weight surfaces bound (see WeightSurfaces).
    """

    def __init__(self, model, identity):
        self.model, self.identity = model, identity
        self.loaded = {}            # (S, P) -> [Program]  (compiled; resident or unloaded)
        self.timing = {}
        self.lock = threading.RLock()

    def directory(self, S, P):
        tile = "" if (QBLK, KCHUNK) == (1024, 2048) else f"_q{QBLK}k{KCHUNK}"
        return CACHE / self.identity / MIL_VERSION / f"S{S}_P{P}{tile}_g{LAYERS_PER_PROGRAM}" / "shared"

    def precompile(self, S, P, on_progress=None):
        """Compile a bucket's program in the background without loading it: a second program
        mapped beside the running one overflows the engine's address window."""
        with self.lock:
            if (S, P) in self.loaded:
                return
            self.loaded[(S, P)] = self._build(S, P, on_progress, load=False)

    def ensure(self, S, P, on_progress=None):
        key = (S, P)
        with self.lock:
            for other in list(self.loaded):      # one bucket resident (mapped) at a time; others stay compiled but unloaded
                if other != key:
                    for p in self.loaded[other]:
                        if p.resident:
                            p.unload()
            others = [k for k in self.loaded if k != key]
            for other in others[:max(0, len(others) + 1 - KEEP_BUCKETS)]:   # this bucket plus at most KEEP_BUCKETS-1 others
                for p in self.loaded.pop(other):  # oldest first
                    p.free()
            if key in self.loaded:
                for p in self.loaded[key]:
                    if not p.resident:
                        p.reload()
                return self.loaded[key]
            programs = self._build(S, P, on_progress)
            self.loaded[key] = programs
            return programs

    def _build(self, S, P, on_progress=None, load=True):
        cfg = self.model.config
        d = self.directory(S, P)
        gen_s = 0.0
        if not (d / "meta.json").exists():
            t0 = time.perf_counter()
            build_program([None] * LAYERS_PER_PROGRAM, d, S=S, P=P, D=cfg.hidden_size, H=cfg.num_attention_heads,
                          KV=cfg.num_key_value_heads, HD=cfg.head_dim, eps=cfg.rms_norm_eps, qblk=QBLK, kchunk=KCHUNK,
                          weights_as_inputs=True)
            gen_s = time.perf_counter() - t0
        program = Program(d, load=load)
        program.layers = LAYERS_PER_PROGRAM
        if on_progress is not None:
            on_progress(1, 1)
        self.timing[(S, P)] = {"generate_seconds": gen_s, "compile_seconds": program.compile_seconds}
        return [program]


class WeightSurfaces:
    """The NAR-path weights of every layer as fp16 IOSurfaces, created once per process.

    With WEIGHT_POOL > 0 only that many layers' surfaces exist and ``bind`` rewrites them.
    """

    def __init__(self, model):
        cfg = model.config
        self.model = model
        self.n_layers = len(model.model.layers)
        self.shapes = weight_input_shapes(cfg.hidden_size, cfg.num_attention_heads, cfg.num_key_value_heads, cfg.head_dim,
                                          cfg.intermediate_size)
        self.pool = WEIGHT_POOL or self.n_layers
        self.surfaces = [{name: Surface(int(np.prod(shape)) * 2) for name, shape in self.shapes.items()} for _ in range(self.pool)]
        self.holding = [None] * self.pool
        if not WEIGHT_POOL:
            for i in range(self.n_layers):
                self._write(i, i)

    def _write(self, slot, layer):
        arrays = arrays_from_state(nar_layer_state(self.model, layer), self.model.config)
        for name, surface in self.surfaces[slot].items():
            surface.write(np.ascontiguousarray(arrays[name].reshape(self.shapes[name])))
        self.holding[slot] = layer

    def bind(self, first_layer, count, inputs):
        """Bind layers first_layer..first_layer+count-1 as w0_*, w1_*, ... into ``inputs``."""
        for j in range(count):
            layer = first_layer + j
            slot = layer if not WEIGHT_POOL else (first_layer // count * count + j) % self.pool
            if self.holding[slot] != layer:
                self._write(slot, layer)
            for name, surface in self.surfaces[slot].items():
                inputs[f"w{j}_{name}"] = surface

    def close(self):
        for slot in self.surfaces:
            for surface in slot.values():
                surface.free()


def weights_for(model):
    cached = getattr(model, "_yue2_ane_weights", None)
    if cached is None:
        cached = WeightSurfaces(model)
        model._yue2_ane_weights = cached
    return cached


def programs_for(model):
    cached = getattr(model, "_yue2_ane_programs", None)
    if cached is None:
        weights = getattr(model, "_yue2_weight_identity", None) or hashlib.sha256(
            str(sorted((n, tuple(p.shape)) for n, p in model.named_parameters())).encode()).hexdigest()[:16]
        cached = LayerPrograms(model, weights)
        model._yue2_ane_programs = cached
    return cached


class ANEVelocity:
    """ANE counterpart of ``CachedNAR.velocity``/``solve`` for one chunk."""

    def __init__(self, engine, model, on_prepare=None):
        cfg = model.config
        self.model, self.engine = model, engine
        if engine.visible_length != engine.ar_length:
            raise ValueError("ANE engine does not support restricted prefix visibility")
        self.D, self.KV, self.HD, self.H = cfg.hidden_size, cfg.num_key_value_heads, cfg.head_dim, cfg.num_attention_heads
        self.S_real, self.P_real = engine.nar_length, engine.ar_length
        self.S, self.P = bucket(self.S_real, S_STEP), bucket(self.P_real, P_STEP)
        self.K = self.S + self.P
        self.programs = programs_for(model).ensure(self.S, self.P, on_progress=on_prepare)
        f32 = lambda t: t.detach().float().cpu()
        self.vae2llm = (f32(model.vae2llm.weight), f32(model.vae2llm.bias))
        self.llm2vae = (f32(model.llm2vae.weight), f32(model.llm2vae.bias))
        self.final_norm, self.eps = f32(model.model.norm.weight), cfg.rms_norm_eps
        import copy
        self.time_embedder = copy.deepcopy(model.time_embedder).to("cpu").eval()   # tiny; keeps the solve off the GPU stream
        self.pos_emb = f32(engine.pos_emb[0])                                 # [S_real, D] (bf16-rounded like torch)
        # Surfaces: residual ping-pong, per-layer prefix K/V, rotary tables, key bias.
        self.x = [Surface(self.S * self.D * 2), Surface(self.S * self.D * 2)]
        # Per-layer prefix K/V stay in host memory; only a small pool of surfaces is mapped for the
        # engine (its per-process address window is ~3.5 GiB and every mapping and program arena counts).
        self.kv_host = []
        for k, v in engine.cache:
            padded_k = np.zeros((1, self.KV, self.P, self.HD), dtype=np.float16)
            padded_v = np.zeros((1, self.KV, self.P, self.HD), dtype=np.float16)
            padded_k[0, :, :self.P_real] = f32(k).numpy().transpose(1, 0, 2)
            padded_v[0, :, :self.P_real] = f32(v).numpy().transpose(1, 0, 2)
            self.kv_host.append((padded_k, padded_v))
        engine.cache = []                       # the GPU copy of the prefix K/V is no longer needed
        if torch.backends.mps.is_available():
            torch.mps.empty_cache()
        self.kv_pool = [(Surface(self.KV * self.P * self.HD * 2), Surface(self.KV * self.P * self.HD * 2))
                        for _ in range(max(p.layers for p in self.programs))]
        self.weights = weights_for(model)
        self.n_layers = len(self.kv_host)
        positions = np.arange(self.P_real, self.P_real + self.S, dtype=np.float32)
        inv = 1.0 / (cfg.rope_theta ** (np.arange(0, self.HD, 2, dtype=np.float32) / self.HD))
        angles = positions[:, None] * inv[None]
        self.cos, self.sin = Surface(self.S * self.HD), Surface(self.S * self.HD)
        self.cos.write(np.cos(angles).reshape(1, 1, self.S, self.HD // 2)); self.sin.write(np.sin(angles).reshape(1, 1, self.S, self.HD // 2))
        bias = np.zeros((1, 1, 1, self.K), dtype=np.float16)
        bias[..., self.P_real:self.P] = NEG
        bias[..., self.P + self.S_real:] = NEG
        self.bias = Surface(self.K * 2); self.bias.write(bias)
        self.layer_seconds = []

    @torch.inference_mode()
    def _time_embedding(self, raw_t):
        shifted = self.model._shift_t_value(raw_t, torch.device("cpu"), torch.bfloat16)
        return self.time_embedder(shifted.expand(1)).float()[0]

    @torch.inference_mode()
    def velocity(self, state, raw_t):
        """state [T_lat, 64] fp32 CPU tensor -> velocity [T_lat, 64] fp32 CPU tensor."""
        x_nar = torch.nn.functional.pad(state, (0, 0, 1, 1))                   # [S_real, 64]
        x = x_nar @ self.vae2llm[0].T + self.vae2llm[1] + self._time_embedding(raw_t) + self.pos_emb
        x0 = np.zeros((self.S, self.D), dtype=np.float16)
        x0[:self.S_real] = x.numpy().astype(np.float16)
        self.x[0].write(x0)
        t0 = time.perf_counter()
        program = self.programs[0]                       # one shared program; weights bound per call
        if not program.resident:
            program.reload()
        calls = 0
        for first in range(0, self.n_layers, program.layers):
            self.current_layer = first
            count = min(program.layers, self.n_layers - first)
            inputs = {"x": self.x[calls % 2], "cos": self.cos, "sin": self.sin, "bias": self.bias}
            for j in range(count):
                pk, pv = self.kv_pool[j]
                hk, hv = self.kv_host[first + j]
                pk.write(hk); pv.write(hv)
                inputs[f"pk{j}"], inputs[f"pv{j}"] = pk, pv
            self.weights.bind(first, count, inputs)
            try:
                program(inputs, [self.x[(calls + 1) % 2]])
            except RuntimeError as exc:
                if "Inference error" not in str(exc):
                    raise
                # A transient engine failure (its address window was briefly oversubscribed): the
                # inputs are intact, so reload the program and run this call again.
                program.unload(); program.reload()
                program(inputs, [self.x[(calls + 1) % 2]])
            calls += 1
        self.layer_seconds.append(time.perf_counter() - t0)
        h = torch.from_numpy(self.x[calls % 2].read((self.S, self.D)).astype(np.float32))[:self.S_real]
        h = h * torch.rsqrt(h.pow(2).mean(-1, keepdim=True) + self.eps) * self.final_norm
        return (h @ self.llm2vae[0].T + self.llm2vae[1])[1:-1]

    def solve(self, noise, steps=32, cancelled=None, on_progress=None):
        state = noise.to(torch.float32).clone()
        dt = 1.0 / steps
        for step in range(steps):
            if cancelled is not None and cancelled():
                raise InterruptedError("Cancelled during acoustic flow matching")
            t = 1.0 - step * dt
            raw = float(torch.logit(torch.tensor(t, dtype=torch.float64)).clamp(-20, 20))
            first = self.velocity(state, raw)
            mid = state - first * (dt / 2)
            raw_mid = float(torch.logit(torch.tensor(t - dt / 2, dtype=torch.float64)).clamp(-20, 20))
            state = state - self.velocity(mid, raw_mid) * dt
            if on_progress is not None:
                on_progress(step + 1, int(steps))
        if not torch.isfinite(state).all():
            raise FloatingPointError("Acoustic flow matching produced non-finite latents")
        return state

    def close(self):
        for s in self.x + [t for kv in self.kv_pool for t in kv] + [self.cos, self.sin, self.bias]:
            s.free()
