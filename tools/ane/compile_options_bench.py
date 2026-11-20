#!/usr/bin/env python3
"""Measure how ANE compiler flags change compile time, call time and output (research).

Each variant is a JSON object of compiler flags (CamelCase names, see DEFAULT_COMPILER_FLAGS in
src/yue2/ane/runtime.py). The program directory is copied so every variant compiles afresh in its
own process (YUE2_ANE_COMPILER_FLAGS), the program is evaluated on fixed random inputs, and its
output is compared with the first variant's.

  .venv/bin/python tools/ane/compile_options_bench.py [--program S512_P2048_g2] [--evals 20]
        [--variants baseline,fast,nocp,...] [--no-eval]

Do not run while YuE Studio has a song in flight: it loads engine programs.
"""
import argparse, json, os, re, shutil, subprocess, sys, tempfile, time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
CACHE = Path(os.environ.get("YUE2_ANE_CACHE", Path.home() / ".cache" / "yue2-ane"))

VARIANTS = {
    "baseline": {},
    "fast": {"DisableCpAllocation": True, "MemCacheStrategy": "None", "SpatialSplit": "Auto", "DisableL2BankConflictOpt": True,
             "EnableGlobalCWOptimization": False, "UseWideWorkUnits": True, "EnableL2BatchSplitting": False},
    "nocp": {"DisableCpAllocation": True},
    "memcache_none": {"MemCacheStrategy": "None"},
    "lowcp": {"EnableLowEffortCPAllocation": True},
    "mt8": {"CompilerMultithreading": 8},
    "service": None,                      # the compiler service route (YUE2_ANE_INPROCESS=0)
}


def find_program(name):
    hits = sorted(CACHE.glob(f"*/v*/{name}/shared/model.mil"))
    if not hits:
        sys.exit(f"no cached program {name} under {CACHE}")
    return hits[-1].parent


def signature(mil_text):
    sig = re.search(r"func main<[^>]*>\((.*?)\)\s*\{", mil_text, re.S).group(1)
    inputs = [(m.group(2), json.loads(m.group(1))) for m in re.finditer(r"tensor<fp16, (\[[^\]]*\])>\s*(\w+)", sig)]
    out = re.search(r"\}\s*->\s*\((\w+)\);", mil_text).group(1)
    shape = json.loads(re.search(rf"tensor<fp16, (\[[^\]]*\])>\s*{out}\s*=", mil_text).group(1))
    return inputs, (out, shape)


def child(directory, evals, out_path):
    import numpy as np
    sys.path.insert(0, str(ROOT / "src"))
    from yue2.ane import runtime
    inputs, (out_name, out_shape) = signature((Path(directory) / "model.mil").read_text())
    t0 = time.perf_counter()
    program = runtime.Program(directory, load=True)
    total = time.perf_counter() - t0
    rng = np.random.default_rng(1234)
    surfaces = {}
    for name, shape in inputs:
        n = int(np.prod(shape))
        a = rng.normal(0, 0.02, n) if name.startswith("w") else rng.uniform(-1, 1, n) if name in ("cos", "sin") else np.zeros(n) if name == "bias" else rng.normal(0, 1, n)
        s = runtime.Surface(n * 2); s.write(a.astype(np.float16).reshape(shape)); surfaces[name] = s
    out = runtime.Surface(int(np.prod(out_shape)) * 2)
    times = []
    for _ in range(max(1, evals)):
        t = time.perf_counter(); program(surfaces, [out]); times.append(time.perf_counter() - t)
    result = out.read(out_shape).astype(np.float32)
    np.save(out_path, result)
    times.sort()
    print(json.dumps({"route": program.route, "compile_s": round(program.compile_seconds, 2), "open_s": round(total, 2),
                      "eval_ms_median": round(1000 * times[len(times) // 2], 2), "nan": bool(np.isnan(result).any())}))


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--program", default="S512_P2048_g2")
    ap.add_argument("--variants", default="baseline,fast")
    ap.add_argument("--evals", type=int, default=20)
    ap.add_argument("--no-eval", action="store_true")
    ap.add_argument("--child", help=argparse.SUPPRESS)
    ap.add_argument("--out", help=argparse.SUPPRESS)
    args = ap.parse_args()
    if args.child:
        return child(args.child, 0 if args.no_eval else args.evals, args.out)
    import numpy as np
    source = find_program(args.program)
    print(f"program {source}  ({json.loads((source / 'meta.json').read_text()).get('mil_ops')} MIL ops)")
    work = Path(tempfile.mkdtemp(prefix="ane-flags-"))
    baseline = None
    for name in args.variants.split(","):
        flags = VARIANTS[name] if name in VARIANTS else json.loads(name)
        copy = work / name / "shared"
        shutil.copytree(source, copy, ignore=shutil.ignore_patterns("hwx-*"))
        env = dict(os.environ)
        if flags is None:
            env["YUE2_ANE_INPROCESS"] = "0"
        else:
            env["YUE2_ANE_COMPILER_FLAGS"] = json.dumps(flags)
        out = work / f"{name}.npy"
        cmd = [sys.executable, __file__, "--child", str(copy), "--out", str(out), "--evals", str(args.evals)] + (["--no-eval"] if args.no_eval else [])
        proc = subprocess.run(cmd, env=env, capture_output=True, text=True)
        line = (proc.stdout.strip().splitlines() or [""])[-1]
        if proc.returncode or not line.startswith("{"):
            print(f"{name:14s} FAILED: {(proc.stderr.strip().splitlines() or ['?'])[-1][:300]}")
            continue
        r = json.loads(line)
        extra = ""
        if not args.no_eval:
            a = np.load(out)
            if baseline is None:
                baseline = a
            else:
                extra = f"  vs first: max|diff| {float(np.nanmax(np.abs(a - baseline))):.4g}"
        print(f"{name:14s} {r['route']:10s} compile {r['compile_s']:7.2f} s  open {r['open_s']:6.2f} s"
              + ("" if args.no_eval else f"  eval {r['eval_ms_median']:.2f} ms") + f"  nan={r['nan']}{extra}")
    print(f"work kept in {work}")


if __name__ == "__main__":
    main()
