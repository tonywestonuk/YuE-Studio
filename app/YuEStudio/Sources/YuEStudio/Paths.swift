import Foundation

/// Where things live. Packaged app: payload in the bundle, runtime under Application Support.
/// Development (swift run from the repo): the repo's .venv and tools/ are used directly.
struct Paths {
    static let support: URL = {
        if let o = ProcessInfo.processInfo.environment["YUE_STUDIO_SUPPORT"] { return URL(fileURLWithPath: o) }
        return FileManager.default.urls(for: .applicationSupportDirectory, in: .userDomainMask)[0].appendingPathComponent("YuE Studio")
    }()
    static let payload: URL? = {
        guard let r = Bundle.main.resourceURL?.appendingPathComponent("payload"),
              FileManager.default.fileExists(atPath: r.appendingPathComponent("uv").path) else { return nil }
        return r
    }()
    static var packaged: Bool { payload != nil }
    static let repoRoot: URL = { var u = URL(fileURLWithPath: #filePath); for _ in 0..<5 { u.deleteLastPathComponent() }; return u }()
    static var python: URL { packaged ? support.appendingPathComponent("env/bin/python") : repoRoot.appendingPathComponent(".venv/bin/python") }
    static var worker: URL {
        if let o = ProcessInfo.processInfo.environment["YUE_STUDIO_WORKER"] { return URL(fileURLWithPath: o) }   // tests
        return packaged ? support.appendingPathComponent("src/tools/yue2_worker.py") : repoRoot.appendingPathComponent("tools/yue2_worker.py")
    }
    static var src: URL { support.appendingPathComponent("src") }
    /// Model weights (an HF_HOME): the environment override, then the folder the user chose (an external
    /// SSD, say; a path string in UserDefaults), then the default under Application Support.
    static let modelsFolderKey = "modelsFolder"
    static var defaultModels: URL { support.appendingPathComponent("models") }
    static var chosenModels: URL? {
        guard let path = UserDefaults.standard.string(forKey: modelsFolderKey), !path.isEmpty else { return nil }
        return URL(fileURLWithPath: path)
    }
    static var models: URL {
        if let o = ProcessInfo.processInfo.environment["YUE_STUDIO_HF_HOME"] { return URL(fileURLWithPath: o) }
        return chosenModels ?? defaultModels
    }
    /// Whether a folder holds downloaded weights (a Hub cache).
    static func holdsModels(_ folder: URL) -> Bool { FileManager.default.fileExists(atPath: folder.appendingPathComponent("hub").path) }
    static var aneCache: URL { support.appendingPathComponent("ane-cache") }
    static var output: URL {
        packaged ? FileManager.default.urls(for: .musicDirectory, in: .userDomainMask)[0].appendingPathComponent("YuE Studio")
                 : repoRoot.appendingPathComponent("outputs/app")
    }
    static var installedMarker: URL { support.appendingPathComponent("installed.json") }
    static var bundledVersion: String { (try? String(contentsOf: payload!.appendingPathComponent("version.txt"), encoding: .utf8))?.trimmingCharacters(in: .whitespacesAndNewlines) ?? "dev" }
    // SheetSage2 transcription lives in a second environment (its pins conflict with YuE's).
    static var sheetsageEnv: URL { support.appendingPathComponent("sheetsage-env") }
    static var sheetsagePython: URL { packaged ? sheetsageEnv.appendingPathComponent("bin/python") : repoRoot.appendingPathComponent(".venv-sheetsage2/bin/python") }
    static var sheetsageMarker: URL { support.appendingPathComponent("sheetsage-installed.json") }
    static var workerEnvironment: [String: String] {
        var env = ProcessInfo.processInfo.environment
        env["PYTHONUNBUFFERED"] = "1"; env["TQDM_DISABLE"] = "1"
        env["YUE2_OUTPUT_DIR"] = output.path; env["YUE2_ANE_CACHE"] = aneCache.path
        env["YUE2_SHEETSAGE_PYTHON"] = sheetsagePython.path   // worker cwd differs between modes
        if packaged || chosenModels != nil { env["HF_HOME"] = models.path; env["HF_HUB_DISABLE_TELEMETRY"] = "1" }
        env["PATH"] = "/usr/bin:/bin:/usr/sbin:/sbin"
        return env
    }
}
