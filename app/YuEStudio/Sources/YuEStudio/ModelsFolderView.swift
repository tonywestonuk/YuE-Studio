import SwiftUI
import AppKit

/// Where the model weights live (issue #4): the default under Application Support, or a folder the
/// user chose, such as an external SSD. The installer downloads into the chosen folder; when the old
/// folder already holds the weights they can be moved instead. The worker reads HF_HOME at launch,
/// so in the running app a change applies after a relaunch.
struct ModelsFolderView: View {
    /// Whether to say the change needs a relaunch (true once the worker is running).
    var relaunchNote = true
    /// Called after the folder changes or the models are moved, so the installer can re-check.
    var onChange: () -> Void = {}
    @AppStorage(Paths.modelsFolderKey) private var modelsFolder = ""
    @State private var previous: URL?          // the folder before the last change, when it still holds the models
    @State private var moving = false
    @State private var changed = false
    @State private var error: String?

    private var overridden: Bool { ProcessInfo.processInfo.environment["YUE_STUDIO_HF_HOME"] != nil }

    var body: some View {
        VStack(alignment: .leading, spacing: 6) {
            HStack {
                Text("Models folder").bold()
                Text(Paths.models.path).foregroundStyle(.secondary).lineLimit(1).truncationMode(.middle).help(Paths.models.path)
                Spacer()
                Button("Change…") { choose() }.disabled(moving || overridden)
                if !modelsFolder.isEmpty { Button("Reset to default") { set(nil) }.disabled(moving || overridden) }
            }
            if overridden {
                Text("Set by the YUE_STUDIO_HF_HOME environment variable.").font(.caption).foregroundStyle(.secondary)
            } else if let previous {
                HStack {
                    Text("The previous folder still holds the downloaded models (about 7 GB).").font(.caption).foregroundStyle(.secondary)
                    Button(moving ? "Moving…" : "Move existing models here") { move(from: previous) }.controlSize(.small).disabled(moving)
                    if moving { ProgressView().controlSize(.small) }
                }
            } else if changed && relaunchNote {
                Text("Takes effect after the app is relaunched.").font(.caption).foregroundStyle(.secondary)
            } else if changed {
                Text("The models will be downloaded into this folder.").font(.caption).foregroundStyle(.secondary)
            }
            if let error { Text(error).font(.caption).foregroundStyle(.red) }
        }
    }

    private func choose() {
        let panel = NSOpenPanel()
        panel.canChooseDirectories = true; panel.canChooseFiles = false; panel.canCreateDirectories = true
        panel.allowsMultipleSelection = false
        panel.directoryURL = Paths.models
        panel.message = "Choose the folder that holds the model weights (about 7 GB, plus 2 GB for SheetSage2)"
        panel.prompt = "Use Folder"
        guard panel.runModal() == .OK, let url = panel.url else { return }
        set(url)
    }

    /// Switches to `folder` (nil: the default). Remembers the old folder when it holds models the new one lacks.
    private func set(_ folder: URL?) {
        let old = Paths.models
        let new = folder ?? Paths.defaultModels
        guard new.standardizedFileURL != old.standardizedFileURL else { return }
        error = nil
        modelsFolder = folder?.path ?? ""
        changed = true
        previous = (Paths.holdsModels(old) && !Paths.holdsModels(new)) ? old : nil
        onChange()
    }

    /// Moves the old folder's Hub cache into the current models folder (a copy across volumes).
    private func move(from old: URL) {
        let source = old.appendingPathComponent("hub"), target = Paths.models.appendingPathComponent("hub")
        moving = true; error = nil
        Task.detached {
            let failure: String? = {
                do {
                    try FileManager.default.createDirectory(at: target.deletingLastPathComponent(), withIntermediateDirectories: true)
                    try FileManager.default.moveItem(at: source, to: target)
                    return nil
                } catch { return error.localizedDescription }
            }()
            await MainActor.run {
                moving = false
                if let failure { error = "Could not move the models: \(failure)" } else { previous = nil; onChange() }
            }
        }
    }
}
