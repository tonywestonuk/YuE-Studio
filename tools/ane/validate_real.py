"""Validate the ANE synthesis engine against PyTorch on a saved song; report timings."""
import json, sys, time, numpy as np, torch
from yue2 import YuE2Pipeline
from yue2.nar import CachedNAR, song_chunks, synthesize
from yue2.ane import runtime
from yue2.pipeline import SymbolicPlan
d = sys.argv[1] if len(sys.argv) > 1 else "outputs/first-song/city_lights"
steps = int(sys.argv[2]) if len(sys.argv) > 2 else 32
pipe = YuE2Pipeline.from_pretrained("m-a-p/YuE2-3B", device="mps", progress=False)
model = pipe._load_model(for_nar=True)
plan = SymbolicPlan.load(d); tokens = np.load(f"{d}/semantic.npy").tolist(); ref = np.load(f"{d}/latent.npy")
chunk = song_chunks(plan.prefix, tokens, plan.request.seed)[0]
engine = CachedNAR(model, chunk)
state = chunk.noise.to(device=pipe.device, dtype=torch.bfloat16)
raw = -20.0
with torch.inference_mode():          # reference first: the ANE solver releases the GPU prefix cache
    vt = engine.velocity(state, raw).float().cpu().numpy()
t0 = time.perf_counter()
solver = runtime.ANEVelocity(engine, model, on_prepare=lambda i, n: print(json.dumps({"prepare_layer": i, "of": n}), flush=True) if i % 7 == 0 else None)
prep = time.perf_counter() - t0
print(json.dumps({"song": d, "frames": len(tokens), "S_real": solver.S_real, "S": solver.S, "P_real": solver.P_real, "P": solver.P,
                  "prepare_s": round(prep), **{k: (round(v) if isinstance(v, (int, float)) else v) for k, v in runtime.programs_for(model).timing[(solver.S, solver.P)].items()}}), flush=True)
try:
    va = solver.velocity(chunk.noise.float(), raw).numpy()
except Exception as e:
    print(json.dumps({"velocity_failed_at_layer": solver.current_layer, "programs": len(solver.programs), "error": str(e)[-90:]}), flush=True); raise
dd = np.abs(vt - va)
print(json.dumps({"velocity_rel_rms": round(float(np.sqrt((dd**2).mean()) / np.sqrt((vt**2).mean())), 5), "max_abs": round(float(dd.max()), 4),
                  "scale": round(float(np.abs(vt).max()), 3), "corr": round(float(np.corrcoef(vt.ravel(), va.ravel())[0, 1]), 6),
                  "ane_layers_ms": round(1000 * solver.layer_seconds[-1])}), flush=True)
t0 = time.perf_counter()
lat = solver.solve(chunk.noise, steps).numpy()
secs = time.perf_counter() - t0
dd = np.abs(lat - ref)
print(json.dumps({"ane_full_solve_s": round(secs), "mean_layers_ms_per_pass": round(1000 * float(np.mean(solver.layer_seconds[1:]))),
                  "latent_rel_rms_vs_ref": round(float(np.sqrt((dd**2).mean()) / np.sqrt((ref**2).mean())), 4),
                  "corr": round(float(np.corrcoef(lat.ravel(), ref.ravel())[0, 1]), 5), "finite": bool(np.isfinite(lat).all())}), flush=True)
solver.close()
