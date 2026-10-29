import Foundation
import Combine

/// On-demand installer for the SheetSage2 transcription environment: its pins (torch 2.8,
/// transformers 4.45, numpy 1.24) conflict with the YuE environment, so it gets its own venv
/// and downloads only when the user first asks for a transcription.
@MainActor
final class SheetSageInstaller: ObservableObject {
    enum State: Equatable { case unknown, needed, running, ready, failed(String) }
    struct Step: Identifiable { let id: Int; let title: String; var done = false }
    static let recipe = "1"      // bump when the venv recipe or pins change to force a reinstall
    @Published var state: State = .unknown
    @Published var steps: [Step] = [Step(id: 0, title: "Install Python 3.11"), Step(id: 1, title: "Download SheetSage2 (about 2 GB)"),
                                     Step(id: 2, title: "Create environment"), Step(id: 3, title: "Install PyTorch"),
                                     Step(id: 4, title: "Install SheetSage2 packages"), Step(id: 5, title: "Finish")]
    @Published var current = 0
    @Published var progress = 0.0
    @Published var detail = ""
    @Published var log: [LogLine] = []
    private var task: Task<Void, Never>?
    private var running: Process?
    private var lastRateLog = Date.distantPast

    func cancel() { running?.terminate(); task?.cancel(); if state == .running { state = .needed } }

    func check() {
        let fm = FileManager.default
        guard Paths.packaged else { state = fm.fileExists(atPath: Paths.sheetsagePython.path) ? .ready : .needed; return }
        let marker = (try? JSONSerialization.jsonObject(with: Data(contentsOf: Paths.sheetsageMarker)) as? [String: String])?["recipe"]
        state = (marker == Self.recipe && fm.fileExists(atPath: Paths.sheetsagePython.path) && Self.modelPresent()) ? .ready : .needed
    }

    /// SheetSage2 is complete only with its MERT-v2 backbone cached at the revision its
    /// config.json pins: the model's remote code resolves the parent at that revision alone,
    /// so an install that fetched MERT's `main` after it moved on (September 2026) cannot
    /// load offline (issues #8, #11) and is reinstalled, which now pulls the pinned revision.
    static func modelPresent() -> Bool {
        let fm = FileManager.default
        let hub = Paths.models.appendingPathComponent("hub")
        let snapshots = hub.appendingPathComponent("models--m-a-p--SheetSage2/snapshots")
        guard let dirs = try? fm.contentsOfDirectory(at: snapshots, includingPropertiesForKeys: nil) else { return false }
        return dirs.contains { snapshot in
            guard let data = try? Data(contentsOf: snapshot.appendingPathComponent("config.json")),
                  let config = try? JSONSerialization.jsonObject(with: data) as? [String: Any],
                  let parent = config["base_model_name_or_path"] as? String, !parent.isEmpty,
                  let revision = config["base_model_revision"] as? String, !revision.isEmpty else { return false }
            let parentDir = "models--" + parent.replacingOccurrences(of: "/", with: "--")
            return fm.fileExists(atPath: hub.appendingPathComponent("\(parentDir)/snapshots/\(revision)/model.safetensors").path)
        }
    }

    func append(_ message: String) {
        let f = DateFormatter(); f.dateFormat = "HH:mm:ss"
        log.append(LogLine(time: f.string(from: Date()), message: message))
    }

    func install() {
        guard let payload = Paths.payload else { return }   // repo mode installs by hand (docs/covers.md)
        state = .running; progress = 0; current = 0
        for i in steps.indices { steps[i].done = false }
        let uv = payload.appendingPathComponent("uv").path
        let support = Paths.support
        let env: [String: String] = ["UV_PYTHON_INSTALL_DIR": support.appendingPathComponent("python").path,
                                     "UV_CACHE_DIR": support.appendingPathComponent("uv-cache").path,
                                     "HF_HOME": Paths.models.path, "HF_HUB_DISABLE_TELEMETRY": "1",
                                     "PATH": "/usr/bin:/bin:/usr/sbin:/sbin", "HOME": NSHomeDirectory()]
        task = Task { [weak self] in
            guard let self else { return }
            do {
                try await step(0) { try await self.run(uv, ["python", "install", "3.11"], env) }
                try await step(1) {
                    // The main env is guaranteed installed by now; its download script takes repo args
                    // and follows SheetSage2's config.json to the MERT-v2 backbone at the revision it pins.
                    try await self.run(Paths.python.path, [Paths.src.appendingPathComponent("tools/download_models.py").path,
                                                          "m-a-p/SheetSage2"], env) { [weak self] line in
                        guard let p = downloadProgress(line) else { return }
                        Task { @MainActor in
                            guard let self else { return }
                            self.progress = p.fraction; self.detail = p.detail
                            if Date().timeIntervalSince(self.lastRateLog) > 15 && p.bytes > 0 {
                                self.lastRateLog = Date(); self.append(String(format: "Downloaded %.2f GB at %.0f MB/s", p.bytes / 1e9, p.rate))
                            }
                        }
                    }
                }
                try await step(2) { try await self.run(uv, ["venv", Paths.sheetsageEnv.path, "--python", "3.11", "--clear"], env) }
                // Plain PyPI wheels: on macOS these are the CPU/MPS builds (covers.md's cu126 index is for Linux).
                try await step(3) { try await self.run(uv, ["pip", "install", "--python", Paths.sheetsagePython.path,
                                                            "torch==2.8.0", "torchaudio==2.8.0"], env) }
                try await step(4) {
                    let snapshots = Paths.models.appendingPathComponent("hub/models--m-a-p--SheetSage2/snapshots")
                    guard let snapshot = (try? FileManager.default.contentsOfDirectory(at: snapshots, includingPropertiesForKeys: nil))?.first else {
                        throw NSError(domain: "YuEStudio", code: 1, userInfo: [NSLocalizedDescriptionKey: "SheetSage2 snapshot not found after download"])
                    }
                    try await self.run(uv, ["pip", "install", "--python", Paths.sheetsagePython.path,
                                            "-r", snapshot.appendingPathComponent("requirements.txt").path, "soundfile"], env)
                }
                try await step(5) {
                    try? FileManager.default.removeItem(at: support.appendingPathComponent("uv-cache"))
                    let data = try JSONSerialization.data(withJSONObject: ["recipe": Self.recipe])
                    try data.write(to: Paths.sheetsageMarker)
                }
                state = .ready
            } catch {
                append("Install failed: \(error.localizedDescription)")
                state = .failed(error.localizedDescription)
            }
        }
    }

    private func step(_ i: Int, _ body: () async throws -> Void) async throws {
        current = i; detail = ""; append("Step \(i + 1): \(steps[i].title)")
        try await body()
        steps[i].done = true; progress = Double(i + 1) / Double(steps.count)
    }

    private func run(_ exe: String, _ args: [String], _ env: [String: String], onLine: (@Sendable (String) -> Void)? = nil) async throws {
        try await runStreaming(exe, args, env, register: { running = $0 }, append: { [weak self] in self?.append($0) }, onLine: onLine)
    }
}
