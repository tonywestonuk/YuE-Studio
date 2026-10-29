import Foundation

struct Song: Identifiable, Equatable {
    /// In-flight stages come from the worker; ready/stalled/failed are settled states.
    enum Status: Equatable { case queued, planning, tokens, synth, decode, ready, stalled, failed }
    var id: String { path }                      // the audio file path: stable across restarts
    let run: String                              // output folder name (timestamp)
    let index: Int
    let path: String
    var score: String
    var seconds: Double
    let seed: Int
    var truncated: Bool
    var status: Status
    var quality = "full"                         // "draft" (8-step MLX preview) or "full" (32 steps)
    var detail = ""                              // what the current stage is doing, or the failure
    var fraction: Double? = nil                  // stage progress, when known
    var gflops: Double? = nil                    // rough throughput of the current stage, GFLOP/s
    var engine = ""                              // synthesis engine (ane / mlx / torch)
    var priority = 0                             // scheduling order (1 = first added); 0 = unknown
    var title = ""                               // the run's title, if the user gave one
    var directory: URL { URL(fileURLWithPath: path).deletingLastPathComponent() }
    var inFlight: Bool { [.queued, .planning, .tokens, .synth, .decode].contains(status) }
    var runLabel: String {
        let f = DateFormatter(); f.dateFormat = "yyyyMMdd-HHmmss"
        guard let d = f.date(from: String(run.prefix(15))) else { return run }
        return d.formatted(date: .abbreviated, time: .standard)
    }
    /// Section header: the title when there is one, then the time.
    var runHeader: String { title.isEmpty ? runLabel : "\(title) · \(runLabel)" }
    var rowName: String { title.isEmpty ? "Song \(index)" : "\(title) · song \(index)" }
    var engineLabel: String {
        switch engine {
        case "ane": return "Neural Engine"; case "mlx": return "GPU (MLX)"; case "torch": return "GPU (PyTorch)"
        case "mlx+ane": return "GPU, then Neural Engine"; case "remote": return "iPhone"; case "mlx+remote": return "GPU, then iPhone"
        default: return ""
        }
    }
    /// Position on the stage track: completed stages plus progress within the current one (0 queued ... 4 done).
    var trackProgress: Double {
        let within = min(1, max(0, fraction ?? 0))
        switch status {
        case .queued: return 0
        case .planning: return within
        case .tokens: return 1 + within
        case .synth: return 2 + within
        case .decode: return 3 + within
        case .ready: return 4
        case .stalled: return 2
        case .failed: return 0
        }
    }
    var throughput: String {
        guard let g = gflops, g > 0 else { return "" }
        return g >= 1000 ? String(format: " · %.2f TFLOP/s", g / 1000) : String(format: " · %.0f GFLOP/s", g)
    }
    var statusLine: String {
        switch status {
        case .queued: return "Queued" + (detail.isEmpty ? "" : " · \(detail)")
        case .planning: return "Planning the score on GPU" + (detail.isEmpty ? "" : " · \(detail)") + throughput
        case .tokens: return "Tokenizing on GPU" + (detail.isEmpty ? "" : " · \(detail)") + throughput
        case .synth: return "Synthing" + (engineLabel.isEmpty ? "" : " using \(engineLabel)") + (detail.isEmpty ? "" : " · \(detail)") + throughput
        case .decode: return "Rendering on GPU" + (detail.isEmpty ? "" : " · \(detail)") + throughput
        case .ready: return "Done"
        case .stalled: return "Tokens saved · not synthesized yet"
        case .failed: return "Failed · \(detail)"
        }
    }

    /// The generation request the worker saved beside the song (style, lyrics, cot, seed, abc, abc_open,
    /// cfg_scale, id), for reusing its materials in the form. Nil when the folder has no request.json.
    func request() -> [String: Any]? {
        guard let data = try? Data(contentsOf: directory.appendingPathComponent("request.json")) else { return nil }
        return (try? JSONSerialization.jsonObject(with: data)) as? [String: Any]
    }
    var hasRequest: Bool { FileManager.default.fileExists(atPath: directory.appendingPathComponent("request.json").path) }

    /// Songs on disk under the output folder: <run>/song<N>/ with audio.flac (ready) or only saved
    /// tokens (stalled: can be synthesized later) plus their sidecar files.
    nonisolated static func scan(_ root: URL) -> [Song] {
        let fm = FileManager.default
        guard let runs = try? fm.contentsOfDirectory(at: root, includingPropertiesForKeys: nil) else { return [] }
        var songs: [Song] = []
        for run in runs {
            guard let dirs = try? fm.contentsOfDirectory(at: run, includingPropertiesForKeys: nil) else { continue }
            for dir in dirs where dir.lastPathComponent.hasPrefix("song") {
                let audio = dir.appendingPathComponent("audio.flac")
                let hasAudio = fm.fileExists(atPath: audio.path)
                let hasTokens = fm.fileExists(atPath: dir.appendingPathComponent("semantic.npy").path) && fm.fileExists(atPath: dir.appendingPathComponent("plan_manifest.json").path)
                guard hasAudio || hasTokens else { continue }
                let json = { (name: String) -> [String: Any]? in (try? JSONSerialization.jsonObject(with: Data(contentsOf: dir.appendingPathComponent(name)))) as? [String: Any] }
                let request = json("request.json"), result = hasAudio ? json("result.json") : nil, tokens = json("tokens.json")
                let truncated = (result?["truncated"] as? [String: Bool])?.values.contains(true) ?? (tokens?["truncated"] as? Bool ?? false)
                songs.append(Song(run: run.lastPathComponent, index: Int(dir.lastPathComponent.dropFirst(4)) ?? 0, path: audio.path,
                                  score: (try? String(contentsOf: dir.appendingPathComponent("score.abc"), encoding: .utf8)) ?? "",
                                  seconds: result?["audio_seconds"] as? Double ?? ((tokens?["frames"] as? Double ?? 0) / 25),
                                  seed: request?["seed"] as? Int ?? 0, truncated: truncated, status: hasAudio ? .ready : .stalled,
                                  quality: result?["quality"] as? String ?? "full",
                                  detail: hasAudio ? "" : "tokens saved · not synthesized yet",
                                  title: (result?["title"] as? String) ?? (tokens?["title"] as? String) ?? ""))
            }
        }
        return songs
    }
}
