# Neural Engine compiler probes (research)

Small Objective-C programs used to find out how the private Neural Engine route compiles and
caches programs (November 2026). Build any of them with
`clang -O1 -fobjc-arc -framework Foundation <name>.m -o <name>`. They touch private frameworks and
exist only to answer questions; nothing in the app depends on them.

- `anecc.m` — calls `ANECCompile` from ANECompiler.framework in-process:
  `anecc <program dir> <out dir> h16g [--flag Key=value ...]` writes `model.hwx` and prints the
  compile time. The `--flag` dictionary is the one the compiler honours (the options plist that the
  in-memory descriptor accepts is written to disk but ignored by the service).
- `devprobe.m` — prints `_ANEDeviceInfo` (engine generation `aneSubTypeAndVariant`, OS build) and the
  `_ANEStrings` table (file names, cache directories, entitlements).
- `cacheprobe.m` — compiles a program through the service and reports `compiledModelExists`,
  `modelURL`, `localModelPath` and `purgeCompiledModel` around compile, load and purge: the daemon
  keeps compiled programs by model identity beyond the compiling process.
- `existsprobe.m` — asks the daemon which cached program directories it still holds compiled.
- `idprobe.m` — shows how the options plist changes the in-memory model's identity hash.

Findings are summarised in `docs/apple-silicon.md` ("Compiling in-process"). The compile-time
benchmark is `tools/ane/compile_options_bench.py`.
