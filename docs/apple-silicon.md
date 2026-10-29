# YuE2 on Apple Silicon (unofficial)

The README targets Linux + NVIDIA, but the torch backend runs on Apple Silicon
through PyTorch MPS. Measured on an M4 with 32 GB unified memory.

## Setup

```bash
uv venv --python 3.12 .venv && source .venv/bin/activate
uv pip install -e .
yue2 doctor                      # should report mps_available: true
yue2 generate --request examples/song.json --device mps --output outputs/first-song
```

The example song (20 s of audio) takes about 2.5 minutes: ~14 tokens/s for the
score plan and song tokens, then ~60 s of audio synthesis and a few seconds of
VAE decoding. `examples/generate.py` hard-codes `device="cuda"`; use the CLI or
pass `device="mps"` to `YuE2Pipeline.from_pretrained`.

## Batched decoding

Token generation is memory-bandwidth bound: every step streams ~5.3 GB of
weights for one token. `yue2.batched.generate_tokens_batched` decodes several
requests in one batch so that weight read is shared. Two MPS-specific patches
(`yue2.batched.mps_decode_patches`) are needed for the batch to actually scale:

- PyTorch MPS treats a `[batch, 1, hidden]` input to `nn.Linear` as a batched
  matmul and re-reads the weight once per row. Flattening to 2D fixes it.
- The grouped-query attention fallback copies K/V per row and the masked
  single-query SDPA kernel is slow. A broadcast-matmul attention replaces it.

Measured per-step cost with 200 fixed steps (`tools/bench_batched.py`):

| Songs per batch | ms per step | aggregate tokens/s |
|---|---|---|
| 1 (library loop) | 65 | 14 |
| 1 | 53 | 16 |
| 2 | 52 | 31 |
| 4 | 56 | 48 |
| 8 | 62 | 73 |

Audio synthesis is compute-bound and still runs per song. Seeded outputs from
the batched path are not bit-identical to the single-request loop.

## Web UI

```bash
python tools/yue2_ui.py          # http://127.0.0.1:7860
```

Enter a style and lyrics, pick a planning mode, and set "Songs per run". All
songs in a run share one batched planning pass and one batched token pass, then
are synthesized in turn. Artifacts land in `outputs/ui/<timestamp>/song<N>/`.

## Synthesis attention on MPS

Audio synthesis (the 32-step flow-matching solver) is compute-bound and runs
per song. On MPS the fused `scaled_dot_product_attention` kernel runs long
non-causal attention at under 1 TFLOP/s, three times slower than the dense
matmuls in the same layer. `yue2.nar.attention` therefore defaults to a
`matmul` path on MPS: contiguous `[KV, groups, S, D]` batched matmuls with a
fused bf16 softmax, no K/V head copies. Measured at 6,000 frames x 12,800 keys:
731 ms -> 351 ms per layer (2.1x); at 1,500 x 2,300: 35 -> 18 ms. Outputs match
the fused path to bf16 precision (relative RMS ~0.8%, latent correlation
0.99997 on a full song). Pass `attention="sdpa-legacy"` to `synthesize` for the
original path. The remaining ceiling is the GPU's ~3.2 TFLOP/s dense rate.

## MLX synthesis engine

`yue2.nar.synthesize(engine="mlx")` (the default on Apple Silicon when `mlx` is
installed; `YUE2_NAR_ENGINE=torch` restores PyTorch) keeps the AR prefix
prefill in PyTorch and runs the 64 velocity evaluations in MLX with its fused
attention kernel. The NAR-path weights are converted once per process (2.7 GB
bf16). Measured on a 3 min 51 s song (5,777 frames): legacy 2,643 s, PyTorch
matmul attention 1,533 s, MLX 1,080 s; latents correlate 0.99995 with the
legacy path. Do not run two model-holding processes at once on a 32 GB machine;
the extra 7 GB copy pushes the system into swap and multiplies run times.

## Neural Engine synthesis engine

`yue2.nar.synthesize(engine="ane")` runs the 28 decoder layers of the flow-matching
solver on the Apple Neural Engine through the private in-memory program API
(`src/yue2/ane/libyue2ane.m`, built with clang; no Core ML at run time). One
program per layer per length bucket (S rows in steps of 512, prefix keys P in
steps of 1024) is generated from the real weights by `src/yue2/ane/mil.py`
using coremltools offline, cached under `~/.cache/yue2-ane/`, and compiled on
first use in a process (about 4 s per layer). Attention is key-chunked with a
streaming softmax (1024 x 2048 tiles) and bucket padding is masked with a bias
input. The input embedding and output projection run on the CPU in fp32.
Measured on the 20 s example song: velocity correlation 0.99974 with PyTorch,
latent correlation 0.9999 with the legacy path, 387 ms per pass, 25 s per
solve (PyTorch 47 s, legacy 60 s).

### If the Neural Engine compiler fails on your Mac

The compiler rejects some shapes on some machines (reported on an M5 Max with
macOS 27: `Error Domain=com.apple.appleneuralengine.compiler Code=22 "Compiler
internal error: Couldn't do topological sort"`). When it does, the app falls
back to the GPU for that song by itself: the song row says "waiting for the
GPU", and the log pane carries the complete compiler error under a "Neural
Engine compiler said:" line (the one-line summary beside it shows only the
first line). Nothing is retried automatically.

When running from source, the attention tiling can be changed through the
environment before the worker (or `yue2 generate`) starts: `YUE2_ANE_QBLK`
(query rows per block, default 1024) and `YUE2_ANE_KCHUNK` (keys per chunk,
default 2048). Setting `YUE2_ANE_KCHUNK=16384` made the failing shapes compile
on that M5 Max, because a key chunk that covers the whole bucket removes the
multi-chunk streaming attention from the program; the same programs then ran
about 8x slower on its Neural Engine than its GPU path, so this is a
diagnostic, not a recommended setting, and the GPU fallback remains the
default.

## Native app

`app/YuEStudio` is a SwiftUI macOS app (`swift build -c release`, then run
`.build/release/YuEStudio`). It launches `tools/yue2_worker.py` from the
project's `.venv` as a long-lived worker and exchanges JSON lines with it, so
the model loads once. The log pane shows every stage: model load, score
planning, song tokens, ANE program compilation, solver steps, decoding, and
saved files. Songs play in place and open in Finder. Outputs go to
`outputs/app/<timestamp>/song<N>/`.

Four-minute song (5,777 frames, largest bucket 6144 x 8192): 843 s on the ANE
versus 1,080 s MLX and 2,643 s legacy, latent correlation 0.9996. Two constraints
shaped the runtime: the engine's ~3.5 GiB per-process address window (each
prepared program keeps a working-buffer arena, so layers are grouped two per
program and only a small pool of prefix-key surfaces is mapped), and the
per-process compile cost (~5 ms per MIL op; about 6 minutes for the largest
bucket, 20 s for a 20-second song). `engine="auto"` selects the ANE.

## Memory (16 GB Macs)

Measured with `footprint` on the worker (20 s song, bucket 512 x 2048):

| stage | before | after |
| --- | --- | --- |
| model loaded | 8.4 GB | 5.3 GB |
| end of a 500-token song stage | 9.7 GB (+2.6 MB per token, unbounded) | 5.5 GB |
| ANE programs compiled, synthesizing | ~11.3 GB | 6.7 GB |
| VAE decode | ~12 GB plus a 7 GB CPU copy of the model | 9.4 GB (1024-frame tiles), less with 512 |

Four changes:

- **Lean model** (`yue2/lean.py`, `YuE2Pipeline(lean=True)`, on by default in the
  worker on Apple Silicon): the model is built on the meta device and only the
  AR path is streamed from the safetensors file to the GPU (4.0 GiB instead of
  6.8; load 2 s instead of 7). The NAR-path layer weights are read from the
  checkpoint on demand by the ANE program builder and by the MLX engine
  (`nar_layer_state`), so nothing is lost; only the pure-PyTorch synthesis engine
  needs the full model.
- **Bounded kernel shapes in token decoding** (`BucketedKVCache`): on MPS every
  distinct attended length is a new compiled graph that stays cached for the
  life of the process (about 2.6 MB per token; gigabytes for a long song, and
  the compile cost about 11 ms per step). Views of the static cache are now
  rounded up to 256 slots, with the slack masked. Steps drop from 50 to 39 ms and
  memory no longer grows with the song. Numerically identical apart from bf16
  accumulation order, the same class of noise as changing the batch composition.
- **No CPU round trip around the VAE**: `decode()` no longer parks the LM on the
  CPU on MPS (unified memory; the copy cost 7 GB and a compressor storm). The GPU
  copy of the prefix K/V is released as soon as the ANE solver has it.
- **ANE program memory**: weight blobs are memory-mapped instead of read into
  private memory, the temp copy the framework needs is an APFS clone, and on
  machines under 24 GB only the current bucket's compiled programs are kept
  (`YUE2_ANE_KEEP_BUCKETS`). The worker also picks 512-frame VAE tiles there.

A retained temp directory does not make a later ANE compile faster
(28.8 s vs 26.8 s for the largest program); the per-process compile remains.

## Titles, playback and the form

A run can be given a **Title**: it names the run folder (`<timestamp>-<title
slug>`), is stored in each song's `tokens.json` and `result.json`, and heads
the run's section and rows in the app. Left empty, it is chosen from the
lyrics by the on-device language model (Apple's Foundation Models framework,
macOS 26 with Apple Intelligence; weak-linked, so on macOS 14 and 15 the first
lyric line is used instead) when Generate is pressed; a title the model chose
is replaced on the next run unless you edit it. Beside the Lyrics label,
**Write lyrics** (shown only when that model is available) writes lyrics for
the style and title with guided generation (one described field per section,
temperature 0.9: free-form prompting made the small model repeat itself) after a
sheet asks what the song is about; a spinner covers the lyrics box meanwhile. It
refuses with an alert while songs are queued or generating, since the model
shares the chip with the engines, and asks before replacing lyrics you typed. The transport bar under the song list
plays the loaded song with pause, back to start, 10-second skips and a
scrubbing slider (space toggles play). The Style editor has a grab bar to make
it taller; the height persists.

## Quality and engines

Synthesis is the expensive stage for long songs, and most of its cost is
independent of whether the song is any good. The app's Quality menu therefore
offers four modes: **Draft (GPU)**, **Draft (GPU + Neural Engine)**, **Full
(GPU)** and **Full (GPU + Neural Engine)**. Draft runs 8 midpoint steps
instead of 32 with the same score, tokens, seed and noise. Measured on a 20 s
song against the 32-step reference (`tools/bench_draft.py`):

| steps | time | latent corr | rel. RMS error |
| --- | --- | --- | --- |
| 32 | 52 s | 1 | 0 |
| 16 | 40 s | 0.9999 | 0.012 |
| 12 | 28 s | 0.9998 | 0.018 |
| 8 | 18 s | 0.9996 | 0.030 |

The engine part (`engines: gpu|gpu+ane` in the worker request) says whether the
scheduler may give the song to the Neural Engine. With "GPU" only, the Neural
Engine's 2.8 GB of weight surfaces are never mapped, the lowest-memory setting
and the right one on 16 GB Macs. Drafts on the Neural Engine were pointless
while a bucket's compile took minutes; with one shared program per bucket it is
seconds to a minute, so they are offered. The Neural Engine is used only for
lengths its compiler accepts (up to 12288 NAR rows with MIL v7); longer songs
stay on the GPU whatever the mode.

Each draft row has **Render full quality**, which re-synthesizes from the saved
tokens with the same seed and noise at 32 steps (worker command `render`) with
the engine choice current in the menu, keeps the preview as `draft.flac`, and
replaces `audio.flac`. `result.json` records `quality`, `ode_steps` and
`nar_engine`.

## One program per bucket (weights as inputs)

The compiler used to be run 14 times per bucket, once per pair of layers,
because each program had its layers' 100 MB of weights baked in as constants.
Since all 28 layers have the same shape, the program now takes the weights as
inputs (`build_program(..., weights_as_inputs=True)`, MIL v6): one compile per
bucket, and the 28 layers run as 14 calls of the same program with each layer's
weight surfaces bound by name (`WeightSurfaces`, created once per process,
~2.8 GB of fp16 IOSurfaces; `YUE2_ANE_WEIGHT_POOL=n` keeps only n layers'
surfaces and rewrites them per call).

| bucket | compile before | compile now | pass before | pass now | output |
| --- | --- | --- | --- | --- | --- |
| 512 x 2048 (20 s) | 14 x 1.8 s | 0.9 s | 27 ms / 2 layers | 32 ms | identical |
| 6144 x 8192 (4 min) | 14 x 25 s = 354 s | 25 s | 871 ms / 2 layers | 902 ms | identical |

The ANE compiler itself cannot be parallelised: threads and even separate
processes all queue on one single-threaded `ANECompilerService`.

**Song-length ceiling (MIL v7).** The compiler refused any bucket above 8192
frames. Probing one op at a time showed the per-head RMSNorm (`q_norm`,
`k_norm`, a reduce over `[.., S, 128]`) as the only op that fails past 8192
rows, in any layout. It is now computed in the flat `[1,1,S,H*128]` layout
with two small constant matmuls (block sums, then spread back; block *sums*,
not means, so the products stay out of fp16's subnormal range). The query heads
of a K/V group also moved from a stacked `2S` row axis to the batch axis. With
that, 8704 x 12288, 9216 x 14336, 12288 x 4096 and 8192 x 18432 all compile;
12288 x 14336 still does not, so the worker tries the ANE up to 12288 frames and
falls back to MLX on a refusal. Accuracy is unchanged (velocity 0.9997 on the
20 s song, 0.9995 on the 4-minute song; latents 0.9999); the batch-axis layout
roughly doubles the op count, so the 4-minute bucket compiles in ~47 s.

**8-bit token decoding in MLX.** The token loop is bound by the bytes of
weights it streams per step (4.04 GB in bf16 at about 109 GB/s: 37 ms), so
`src/yue2/ar_mlx.py` runs it in MLX with the AR weights stored as 8-bit integers
plus a 16-bit scale and bias per group of 64 (2.2 GB), expanded inside MLX's
quantized matmul. Measured on the M4: 18.5 ms per step against 33 to 37 for
PyTorch, a 1.8x speedup at batch sizes 1, 2 and 4, with the same top-1 first
token and a 0.9996 logit correlation; greedy decoding diverges from PyTorch at
the same point as the unquantized MLX port does, so the drift is bf16
accumulation order, not the quantization. The worker uses it by default on
machines with 24 GB or more (`YUE2_AR_ENGINE=torch` restores the PyTorch loop);
the 8-bit copy is quantized before each batch (about a second) and dropped
after it, and the MLX synthesis weights are dropped whenever no song needs the
GPU for synthesis (`trim_gpu_memory`): with the PyTorch weights, the Neural
Engine surfaces and a long song's activations all resident, the worker
otherwise peaked at 26 GB on the 32 GB machine, and the Neural Engine's
evaluate fails when its mapped memory is squeezed. 16 GB machines keep the
PyTorch loop.
`tools/bench_ar_mlx.py` reproduces the comparison.

**The scheduler.** `tools/yue2_worker.py` treats songs as processes and the
GPU and the Neural Engine as resources. Each song has a priority (its arrival
number) and a state: queued, planning, tokenizing, waiting for synthesis,
synthesizing, waiting to render, rendering, done. One function,
`Scheduler.schedule`, runs after every event and assigns resources by priority:

- *Neural Engine*: the highest-priority song that wants it (full quality, length
  the compiler accepts), including a song already solving on the GPU, which
  moves across at its next step boundary when it outranks everything waiting
  (`nar_switch.synthesize_switchable`: same noise and schedule, the state is
  handed over; the prefix K/V stays on the GPU until the Neural Engine solver
  is built from it). While a song solves on the Neural Engine the next Neural
  Engine song's program compiles in the background.
- *GPU, tokenizing*: the highest-priority queued song, taking every other queued
  song of the same kind (same planning mode) along in one batch of up to
  `YUE2_MAX_BATCH` (4 on 24 GB+, else 2), across jobs: one row or four cost the
  same per step. Rows have their own token budgets and leave the batch the
  moment they end (`generate_tokens_batched(on_row_done=..., limits=...)`). A
  queued song that outranks the song synthesizing on the GPU starts its batch
  anyway, and the synthesis pauses (preemption).
- *GPU, synthesis*: drafts, and full-quality songs while the Neural Engine is
  busy; only when no batch runs and no queued song outranks the candidate. It
  pauses between steps whenever tokenizing, rendering or the Neural Engine
  song's prefix prefill are on the GPU (MLX's long kernels would starve them:
  25x slower token steps measured).
- *Rendering*: always admitted; its tiles interleave with token steps on the
  device lock.

PyTorch's Metal backend cannot be driven from two threads at once (it encodes
into one command buffer and asserts), so every PyTorch GPU section takes turns
on `locks.FairLock`: each token step, the synthesis prefill and weight copies,
each decode tile. The Neural Engine and MLX solvers run outside the lock, which
is where the overlap comes from. Below 24 GB the resources take turns
(`YUE2_PIPELINE=1` overrides). Songs enter with a `started` event and report
`stage` (with `priority`) and per-song `progress`; `idle` fires when the last
song finishes. Tokens are written to the song folder as soon as they exist, so
a song whose synthesis never ran can be synthesized later with the `render`
command (the app lists such songs as "tokens only").

**Background compiles must not load.** The bridge's original load function
compiled a program and loaded it into the engine (mapping its arena in the
process's ~3.5 GiB address window) before `precompile` unloaded it again. With
a long song's program resident next to the 2.8 GB of weight surfaces, that
transient second mapping overflowed the window and the *running* inference
failed ("Program Inference error", status 0x2) within seconds of "programs
ready". `ane_program_compile(dir, do_load=0)` now compiles without loading;
programs are loaded only by `ensure`, after the previous bucket is unloaded. A
pass that still fails with an inference error is retried once after reloading
the program (the solver state is on the CPU, so nothing is lost).

**Memory between jobs.** When the queue drains the worker releases its
working memory (MLX weights and cache, ANE weight surfaces and program
mappings, the VAE, the MPS cache): about 11 GB -> 5 GB resident with the lean
model kept. After `YUE2_IDLE_UNLOAD_S` (default 600 s) idle it drops the model
too (-> 0.6 GB); the next job reloads it in about a second. Compiled ANE
programs survive the first level (they hold no weights now) and are freed by
the second.

## iPhone companion (YuE Remote)

`app/YuERemote` is an iOS app (Xcode project generated by `xcodegen generate`, signed with an
Apple Development identity) that turns an iPhone into a second Neural Engine for synthesis.
The Mac keeps the solver loop and the tiny CPU-side maths (input projection, time embedding,
final norm, output projection); the phone runs the 28 decoder layers of each pass, two per
Core ML call, through the **public** Core ML API (`MLModel`, `.cpuAndNeuralEngine`). The same
MIL program `src/yue2/ane/mil.py` generates for the Mac's private route is saved as an
iOS 18 `.mlpackage` with weights as inputs, so it has no constants (about 300 KB) and one program
serves all 28 layers.

Measured on an iPhone 17 Pro (A19 Pro, 12 GB), all ops on its Neural Engine:

| bucket (rows × prefix) | per 2-layer call | 28-layer pass |
|---|---|---|
| 512 × 1024 | 44 ms | 0.62 s |
| 2048 × 2048 | 215 ms | 3.0 s |
| 2048 × 4096 | 256 ms | 3.6 s |
| 4096 × 4096 | 498 ms | 7.0 s |

That is at least the M4's pace for the same program. What its compiler accepts differs from
the Mac's (`src/yue2/remote/client.py` bakes the working choices in): query tiles of 512 rows
with 1024-key chunks (1024-row tiles fail outright from 4096 rows), and the plain layer form.
Row-blocking the projections and MLP does not help (the whole program is refused at 6656
rows), and the reduce form of the per-head RMSNorm is refused at any length. At 6656 rows
(a 4-minute song) the plain form compiles except for the 24 head-norm ops (the block-sum
matmul, its spread matmul and the `tile`), which fall to the CPU; a "spread" form of that norm
(one block-diagonal matmul that yields each head's sum already spread across its columns, with
the matrices as program inputs, `head_norm="spread"`) is the candidate under test. Until it
compiles, songs above 4096 rows (about 2 min 40 s) stay on the Mac (`YUE2_REMOTE_MAX_ROWS`).
Compiles are slow on the phone (2 min for 4096 rows, 10–20 min for 6656) but cached there per
bucket. The `com.apple.developer.kernel.increased-memory-limit` entitlement lifts the app's
ceiling from 3.4 GB to about 6 GB on a 12 GB phone; the 2.8 GB of fp16 weights are stored
once in the app's container and memory-mapped.

Measured end to end (`tools/remote_probe.py`, 75-second song, 4 steps): the phone's latents
correlate 0.999989 with the Mac's Neural Engine (0.46% RMS), the weight transfer runs at
34 MB/s over Wi-Fi (85 s, once), and with two draft songs queued the scheduler put one on
each Neural Engine and finished both in 99 s instead of about 130 s.

Protocol (`src/yue2/remote/protocol.py`, `app/YuERemote/Sources/Protocol.swift`): TCP frames of
`u32 header length | JSON header | u64 payload length | payload`. Ops: `hello`, `weights_begin`
/ `weights_layer` / `weights_end` (2.8 GB, once), `program` (the mlpackage files, compiled on the
phone), `open` (bucket and real lengths), `kv` (a layer's prefix keys and values, fp16
unpadded), `tables` (rotary tables and key bias), `velocity` (x in, h out, fp16 `[S, D]`),
`close`, `ping`. Per pass the Mac sends and receives `S × 2048 × 2` bytes (27 MB each way for a
4-minute song); per song it sends about 400 MB of prefix K/V and gets 1.7 MB of latents back.

The Mac app finds the phone with Bonjour (`_yuestudio._tcp`) and hands the worker its address
(`{"cmd": "remote", "host", "port", "name"}`); the worker treats it as a third engine
(`Scheduler.remote_synth`): after the internal Neural Engine has taken the top song that wants
it, the phone takes the next song the user allowed a Neural Engine for, waiting or already on
the GPU (which then hands over mid-trajectory, `nar_switch` with `may_switch()` returning
`"remote"`). A song the phone cannot run (its compiler rejects the length, or the connection
drops) goes back to the queue for the Mac. `engine="remote"` in `yue2.nar.synthesize` and
`RemoteVelocity` in `src/yue2/remote/velocity.py` mirror the `ANEVelocity` contract.
