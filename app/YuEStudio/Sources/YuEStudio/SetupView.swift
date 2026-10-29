import SwiftUI

struct SetupView: View {
    @EnvironmentObject var installer: Installer
    var body: some View {
        VStack(alignment: .leading, spacing: 14) {
            Text("Setting up YuE Studio").font(.title2).bold()
            Text("First run only: this installs a private Python runtime, the YuE music model code, and downloads the model weights. It needs about 10 GB of disk space and an internet connection.")
                .foregroundStyle(.secondary).fixedSize(horizontal: false, vertical: true)
            if installer.state != .running {
                ModelsFolderView(relaunchNote: false) { installer.check() }
            }
            ForEach(installer.steps) { step in
                HStack {
                    Image(systemName: step.done ? "checkmark.circle.fill" : (installer.current == step.id && installer.state == .running ? "arrow.triangle.2.circlepath" : "circle"))
                        .foregroundStyle(step.done ? .green : .secondary)
                    Text(step.title).bold(installer.current == step.id && installer.state == .running)
                    if installer.current == step.id && installer.state == .running && !installer.detail.isEmpty { Text("· " + installer.detail).foregroundStyle(.secondary) }
                }
            }
            ProgressView(value: installer.progress)
            LogTextView(lines: installer.log).frame(minHeight: 160)
            HStack {
                if case .failed(let message) = installer.state {
                    Text(message).foregroundStyle(.red); Spacer()
                    Button("Retry") { installer.install() }.buttonStyle(.borderedProminent)
                } else if installer.state == .needed {
                    Spacer(); Button("Install") { installer.install() }.buttonStyle(.borderedProminent).keyboardShortcut(.defaultAction)
                } else { Spacer(); ProgressView().controlSize(.small) }
            }
        }
        .padding(24).frame(minWidth: 640, minHeight: 520)
    }
}
