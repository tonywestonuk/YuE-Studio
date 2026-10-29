import SwiftUI
import AppKit

@main
struct YuEStudioApp: App {
    @StateObject private var backend = Backend()
    @StateObject private var installer = Installer()
    var body: some Scene {
        WindowGroup("YuE Studio") { RootView().environmentObject(backend).environmentObject(installer) }
            .commands {
                CommandGroup(replacing: .appTermination) { Button("Quit YuE Studio") { installer.cancel(); backend.quit(); NSApp.terminate(nil) }.keyboardShortcut("q") }
                CommandGroup(after: .appSettings) {
                    Button("Open Songs Folder") { NSWorkspace.shared.open(Paths.output) }
                    Button("Repair Installation…") { backend.quit(); installer.repair() }.disabled(!Paths.packaged)
                }
            }
        // Settings… (Cmd-,): where the model weights live, e.g. an external SSD.
        Settings {
            VStack(alignment: .leading, spacing: 12) {
                Text("Storage").font(.headline)
                ModelsFolderView(relaunchNote: true)   // re-checked on the next launch: a missing download starts the installer
            }.padding(20).frame(width: 560)
        }
    }
}
