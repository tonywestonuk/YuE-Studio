import SwiftUI
import AppKit
import UniformTypeIdentifiers

struct ContentView: View {
    @EnvironmentObject var backend: Backend
    @StateObject private var players = Players()
    @AppStorage("style") private var style = "English, warm piano pop, expressive female voice, acoustic piano, rounded bass and light drums, lyrical memorable melody, unhurried phrasing, 88 BPM"
    @AppStorage("lyrics") private var lyrics = "[Verse]\nNeon fades along the lane\nFootsteps keep the time of rain\nFold the night and leave it here\nMorning has a sky to clear\n\n[Chorus]\nLet the day come into view\nEvery road begins with you\nHold a little room for light\nWe will sing beyond the night"
    @AppStorage("cot") private var cot = "full"
    @AppStorage("seed") private var seed = 831001
    @AppStorage("randomSeed") private var randomSeed = false
    @AppStorage("batch") private var batch = 2
    @AppStorage("maxSeconds") private var maxSeconds = 120.0
    @AppStorage("qualityMode") private var qualityMode = "draft-gpu"   // draft-gpu | draft-gpu-ane | full-gpu | full-gpu-ane
    @AppStorage("useRemote") private var useRemote = true
    @StateObject private var remote = RemoteBrowser()
    private var quality: String { qualityMode.hasPrefix("draft") ? "draft" : "full" }
    private var engines: String { qualityMode.hasSuffix("-ane") ? "gpu+ane" : "gpu" }
    @AppStorage("instrumental") private var instrumental = false
    @AppStorage("abc") private var abc = ""      // a transcribed score survives relaunch
    @AppStorage("abcOpen") private var abcOpen = false   // the score is an opening (a hum) the planner continues
    @State private var humming = false
    @State private var showScore: Song?
    @StateObject private var sheetsage = SheetSageInstaller()
    @State private var transcribeSource: PickedAudio?
    @AppStorage("logPanelHeight") private var logPanelHeight = 130.0
    @State private var logDragStart: Double? = nil
    @AppStorage("title") private var title = ""
    @AppStorage("titleAuto") private var titleAuto = ""    // the last title the model chose: replaced on the next Generate unless edited
    @State private var naming = false
    @State private var writingLyrics = false
    @State private var lyricsAlert: String?                // shown when writing lyrics is refused or fails
    @State private var askAbout = false                    // the "what is the song about?" sheet
    @State private var lyricsVersion = 0                   // bumped on programmatic replacement so the editor rebuilds
    @AppStorage("lyricsAbout") private var lyricsAbout = ""
    @AppStorage("styleHeight") private var styleHeight = 72.0
    @State private var styleDragStart: Double? = nil

    var body: some View {
        HSplitView {
            form.frame(minWidth: 380, idealWidth: 440)
            VStack(spacing: 0) {
                results.frame(minHeight: 160)
                TransportBar(players: players)
                logSplitter
                logView.frame(height: max(64, min(logPanelHeight, 600)))
            }.frame(minWidth: 480)
        }
        .frame(minWidth: 960, minHeight: 640)
        .onAppear {
            backend.rescan(); if backend.process == nil { backend.start() }; remote.start()
            backend.remoteRetry = { if useRemote, let phone = remote.phone { backend.useRemote(phone) } }
        }
        .onChange(of: remote.phone) { _, phone in backend.useRemote(useRemote ? phone : nil) }
        .onChange(of: useRemote) { _, on in backend.useRemote(on ? remote.phone : nil) }
        .onChange(of: backend.connected) { _, up in if up { backend.useRemote(useRemote ? remote.phone : nil) } }
        .onReceive(NotificationCenter.default.publisher(for: NSApplication.didBecomeActiveNotification)) { _ in backend.rescan() }
        .sheet(item: $transcribeSource) { picked in
            TranscribeSheetView(source: picked.url, hum: picked.hum, abc: $abc, abcOpen: $abcOpen, cot: $cot, sheetsage: sheetsage).environmentObject(backend)
        }
        .sheet(isPresented: $humming) {
            HumSheetView { url in
                backend.transcribe = .idle
                DispatchQueue.main.asyncAfter(deadline: .now() + 0.4) { transcribeSource = PickedAudio(url: url, hum: true) }
            }
        }
    }

    private func pickRecording() {
        let panel = NSOpenPanel()
        panel.allowedContentTypes = [.audio]
        panel.allowsMultipleSelection = false
        panel.message = "Choose a recording to transcribe into a melody score"
        if panel.runModal() == .OK, let url = panel.url {
            backend.transcribe = .idle
            transcribeSource = PickedAudio(url: url)
        }
    }

    private var form: some View {
        Form {
            Section("Song") {
                TextField("Title (chosen from the lyrics if left empty)", text: $title)
                Text("Style").font(.caption).foregroundStyle(.secondary)
                TextEditor(text: $style).font(.body).frame(height: max(44, min(styleHeight, 400)))
                resizeHandle(height: $styleHeight, dragStart: $styleDragStart, range: 44...400)
                HStack {
                    Text("Lyrics").font(.caption).foregroundStyle(.secondary)
                    Spacer()
                    if TitleSuggester.modelAvailable {
                        Button(writingLyrics ? "Writing…" : "Write lyrics") { writeLyricsTapped() }
                            .font(.caption).disabled(writingLyrics || naming)
                            .help("Write lyrics for the style and title with the Mac's on-device language model")
                    }
                }
                .alert("Can't write lyrics now", isPresented: Binding(get: { lyricsAlert != nil }, set: { if !$0 { lyricsAlert = nil } })) {
                    Button("OK", role: .cancel) {}
                } message: { Text(lyricsAlert ?? "") }
                .sheet(isPresented: $askAbout) {
                    VStack(alignment: .leading, spacing: 12) {
                        Text("What should the song be about?").font(.headline)
                        TextEditor(text: $lyricsAbout).font(.body).frame(height: 90)
                            .overlay(RoundedRectangle(cornerRadius: 6).stroke(Color.secondary.opacity(0.3)))
                        Text("A theme, a story, a feeling, a place. The style" + ((title.trimmingCharacters(in: .whitespaces).isEmpty || title.trimmingCharacters(in: .whitespaces) == titleAuto) ? " is" : " and the title are") + " used too."
                             + (lyrics.trimmingCharacters(in: .whitespacesAndNewlines).isEmpty ? "" : " The current lyrics will be replaced."))
                            .font(.caption).foregroundStyle(.secondary).fixedSize(horizontal: false, vertical: true)
                        HStack {
                            Spacer()
                            Button("Cancel") { askAbout = false }.keyboardShortcut(.cancelAction)
                            Button("OK") { askAbout = false; Task { await writeLyrics() } }.keyboardShortcut(.defaultAction).buttonStyle(.borderedProminent)
                        }
                    }.padding(20).frame(width: 460)
                }
                TextEditor(text: $lyrics).font(.system(.body, design: .monospaced)).frame(minHeight: 220)
                    .id(lyricsVersion)
                    .disabled(writingLyrics)
                    .overlay {
                        if writingLyrics {
                            VStack(spacing: 8) { ProgressView(); Text("Writing lyrics…").font(.caption).foregroundStyle(.secondary) }
                                .padding(16).background(.regularMaterial, in: RoundedRectangle(cornerRadius: 10))
                        }
                    }
            }
            Section("Run") {
                Stepper("Songs per run: \(batch)", value: $batch, in: 1...8)
                Picker("Planning", selection: $cot) { Text("full").tag("full"); Text("melody").tag("melody"); Text("off").tag("off") }.pickerStyle(.segmented)
                Picker("Quality", selection: $qualityMode) {
                    Text("Draft (GPU)").tag("draft-gpu")
                    Text("Draft (GPU + Neural Engine)").tag("draft-gpu-ane")
                    Text("Full (GPU)").tag("full-gpu")
                    Text("Full (GPU + Neural Engine)").tag("full-gpu-ane")
                }
                Text(qualityCaption).font(.caption).foregroundStyle(.secondary)
                Toggle("Use an iPhone's Neural Engine (YuE Remote app)", isOn: $useRemote)
                Text(useRemote ? (backend.remoteStatus.isEmpty ? remote.detail : backend.remoteStatus) : "off").font(.caption).foregroundStyle(.secondary)
                Toggle("Instrumental (no vocals)", isOn: $instrumental)
                if instrumental {
                    Text("Adds no-vocal tags, keeps only the section markers of the lyrics, and silences the vocal voice in the planned score before the song is tokenized. Planning is used even if set to off. A supplied ABC score has its vocal voice silenced too.").font(.caption).foregroundStyle(.secondary)
                }
                HStack { Text("Max length"); Slider(value: $maxSeconds, in: 30...360, step: 10); Text("\(Int(maxSeconds)) s").monospacedDigit().frame(width: 44) }
                HStack {
                    TextField("Base seed", value: $seed, format: .number).disabled(randomSeed)
                    Toggle("Random", isOn: $randomSeed)
                }
                DisclosureGroup("ABC score (optional)") {
                    TextEditor(text: $abc).font(.system(.caption, design: .monospaced)).frame(height: 100)
                    if !abc.trimmingCharacters(in: .whitespacesAndNewlines).isEmpty {
                        Picker("The score is", selection: $abcOpen) {
                            Text("the whole song").tag(false)
                            Text("an opening the planner continues").tag(true)
                        }
                    }
                    HStack {
                        Button("Transcribe recording…") { pickRecording() }
                            .disabled(!backend.connected || backend.transcribe == .transcribing)
                        Button { humming = true } label: { Label("Hum a melody…", systemImage: "mic") }
                            .disabled(!backend.connected || backend.transcribe == .transcribing)
                    }
                    Text("SheetSage2 melody transcription for covers and hummed tunes · weights CC BY-NC 4.0").font(.caption).foregroundStyle(.secondary)
                }
            }
            Section {
                HStack {
                    Button(action: { Task { await generateTapped() } }) {
                        Label(naming ? "Choosing a title…" : backend.busy ? "Add to queue" : "Generate", systemImage: backend.busy ? "plus" : "play.fill").frame(maxWidth: .infinity)
                    }
                        .buttonStyle(.borderedProminent).keyboardShortcut(.return, modifiers: .command).disabled(!backend.connected || naming)
                    Button(action: { backend.stop() }) { Label("Stop all", systemImage: "stop.fill") }
                        .disabled(!backend.busy)
                }
                pipelineStatus
                if !backend.connected { Text("Worker not connected").foregroundStyle(.red).font(.caption) }
            }
        }
        .formStyle(.grouped)
    }

    /// Writing lyrics runs the on-device model, which shares the chip with the engines: refused while songs are in flight.
    private func writeLyricsTapped() {
        if backend.busy {
            lyricsAlert = "Songs are generating. The on-device language model would take power from the GPU and Neural Engine, so wait until the queue is empty."
            return
        }
        askAbout = true
    }

    private func writeLyrics() async {
        writingLyrics = true
        defer { writingLyrics = false }
        // A title the model chose from the previous lyrics must not steer the new ones: drop it, and
        // let the next Generate name the song from what gets written.
        let typed = title.trimmingCharacters(in: .whitespaces)
        let userTitle = (typed.isEmpty || typed == titleAuto) ? "" : typed
        if userTitle.isEmpty { title = ""; titleAuto = "" }
        switch await TitleSuggester.writeLyrics(style: style, title: userTitle,
                                                about: lyricsAbout.trimmingCharacters(in: .whitespacesAndNewlines)) {
        case .success(let text)?: lyrics = text; lyricsVersion += 1
        case .failure(let error)?: lyricsAlert = "The on-device model declined: \(error.localizedDescription)"
        case nil: lyricsAlert = "The on-device language model is not available on this Mac."
        }
    }

    /// An empty Title (or one the model chose last time) is filled from the lyrics before the run is queued.
    private func generateTapped() async {
        let typed = title.trimmingCharacters(in: .whitespaces)
        if typed.isEmpty || typed == titleAuto {
            naming = true
            let suggested = await TitleSuggester.suggest(lyrics: lyrics, style: style, instrumental: instrumental)
            naming = false
            title = suggested; titleAuto = suggested
        }
        backend.generate(title: title.trimmingCharacters(in: .whitespaces), style: style, lyrics: lyrics, cot: cot, seed: seed, randomSeed: randomSeed, batch: batch,
                         maxTokens: Int(maxSeconds * 25), engine: "auto", abc: abc, abcOpen: abcOpen, quality: quality, engines: engines, instrumental: instrumental)
    }

    private var qualityCaption: String {
        switch qualityMode {
        case "draft-gpu": return "8 solver steps on the GPU: a quick preview of the same song. Render any draft at full quality from its row."
        case "draft-gpu-ane": return "8 solver steps; the Neural Engine takes songs it can compile (a few seconds to minutes per new length), the GPU the rest."
        case "full-gpu": return "32 solver steps on the GPU only. Lowest memory: the Neural Engine's 2.8 GB of weights are never mapped. Best choice on 16 GB Macs."
        default: return "32 solver steps; the Neural Engine and the GPU work on queued songs together. Long songs can take an hour or more."
        }
    }

    /// One line per stage: what it is working on right now.
    private var pipelineStatus: some View {
        let queued = backend.songs.filter { $0.status == .queued }.count
        return VStack(alignment: .leading, spacing: 3) {
            stageLine("Planning · GPU", backend.songs.filter { $0.status == .planning })
            stageLine("Tokenizing · GPU", backend.songs.filter { $0.status == .tokens })
            stageLine("Synthing · " + (backend.songs.first { $0.status == .synth }?.engineLabel ?? "Neural Engine"), backend.songs.filter { $0.status == .synth })
            stageLine("Rendering · GPU", backend.songs.filter { $0.status == .decode })
            HStack { Text("Queued").bold(); Spacer(); Text(queued == 0 ? "—" : "\(queued) song\(queued == 1 ? "" : "s")").foregroundStyle(.secondary) }.font(.caption)
        }
    }

    private func stageLine(_ title: String, _ songs: [Song]) -> some View {
        HStack(alignment: .firstTextBaseline) {
            Text(title).bold()
            Spacer()
            if let first = songs.first {
                let names = songs.map { "song \($0.index)" }.joined(separator: ", ")
                Text("\(names) · \(first.runLabel)" + (first.detail.isEmpty ? "" : " · \(first.detail)")).foregroundStyle(.secondary).lineLimit(1).truncationMode(.middle)
            } else {
                Text("idle").foregroundStyle(.secondary)
            }
        }.font(.caption)
    }

    private var runs: [String] { Array(NSOrderedSet(array: backend.songs.filter { !$0.inFlight }.map(\.run))) as? [String] ?? [] }

    /// In-flight songs are grouped by the stage they are in (in pipeline order), then finished runs, newest first.
    private static let stages: [(Song.Status, String)] = [(.queued, "Queued"), (.planning, "Planning · GPU"), (.tokens, "Tokenizing · GPU"), (.synth, "Synthing"), (.decode, "Rendering · GPU")]

    private var results: some View {
        List {
            ForEach(Self.stages, id: \.1) { status, title in
                let inStage = backend.songs.filter { $0.status == status }
                if !inStage.isEmpty {
                    Section(status == .synth ? "Synthing · " + (inStage.first { !$0.engineLabel.isEmpty }?.engineLabel ?? "Neural Engine / GPU") : title) {
                        ForEach(inStage) { song in row(song) }
                    }
                }
            }
            ForEach(runs, id: \.self) { run in
                Section(backend.songs.first { $0.run == run }?.runHeader ?? run) {
                    ForEach(backend.songs.filter { $0.run == run && !$0.inFlight }) { song in row(song) }
                }
            }
        }
        .overlay { if backend.songs.isEmpty { Text("Songs appear here").foregroundStyle(.secondary) } }
        .safeAreaInset(edge: .top) { HStack { Text("Songs").font(.caption).bold(); Spacer(); Button("Open songs folder") { NSWorkspace.shared.open(Paths.output) }.font(.caption) }.padding(.horizontal, 8).padding(.vertical, 4).background(.bar) }
        .sheet(item: $showScore) { song in
            VStack(alignment: .leading) {
                Text("Score for song \(song.index)").font(.headline)
                ScrollView { Text(song.score).font(.system(.body, design: .monospaced)).textSelection(.enabled).frame(maxWidth: .infinity, alignment: .leading) }
                HStack { Spacer(); Button("Close") { showScore = nil }.keyboardShortcut(.cancelAction) }
            }.padding().frame(width: 640, height: 480)
        }
    }

    private func row(_ song: Song) -> some View {
        HStack {
            Button(action: { players.toggle(song) }) {
                switch song.status {
                case .ready: Image(systemName: players.playing == song.id ? "pause.circle.fill" : "play.circle.fill").font(.title)
                    .foregroundStyle(players.current?.id == song.id ? Color.accentColor : Color.primary)
                case .stalled: Image(systemName: "doc.text").font(.title).foregroundStyle(.secondary).frame(width: 28)
                case .failed: Image(systemName: "exclamationmark.triangle.fill").font(.title).foregroundStyle(.red).frame(width: 28)
                case .queued: Image(systemName: "clock").font(.title).foregroundStyle(.secondary).frame(width: 28)
                default: ProgressView().controlSize(.small).frame(width: 28)
                }
            }.buttonStyle(.plain).disabled(song.status != .ready)
            VStack(alignment: .leading, spacing: 2) {
                switch song.status {
                case .ready:
                    Text("\(song.rowName) · seed \(song.seed) · \(String(format: "%.1f", song.seconds)) s" + (song.truncated ? " · truncated" : "")
                         + (song.quality == "draft" ? " · draft" : "")).bold()
                case .stalled:
                    Text("\(song.rowName) · seed \(song.seed) · about \(Int(song.seconds)) s · tokens only").bold().foregroundStyle(.secondary)
                case .failed:
                    Text("\(song.rowName) · seed \(song.seed) · failed").bold().foregroundStyle(.red)
                default:
                    Text((song.priority > 0 ? "#\(song.priority) · " : "") + "\(song.rowName) · seed \(song.seed) · \(song.runLabel)" + (song.quality == "draft" ? " · draft" : "")).bold().foregroundStyle(.secondary)
                }
                if song.inFlight {
                    StageTrack(progress: song.trackProgress)
                    Text("Status: " + song.statusLine).font(.caption).lineLimit(1)
                } else if song.status == .failed || song.status == .stalled {
                    Text(song.statusLine).font(.caption).foregroundStyle(song.status == .failed ? .red : .secondary).lineLimit(2)
                } else {
                    Text(song.path).font(.caption).foregroundStyle(.secondary).lineLimit(1).truncationMode(.middle)
                }
            }
            Spacer()
            if song.inFlight {
                Button("Cancel") { backend.cancel(song) }.disabled(!backend.connected)
            } else if song.status == .stalled || (song.status == .failed && FileManager.default.fileExists(atPath: song.directory.appendingPathComponent("semantic.npy").path)) {
                Button(quality == "draft" ? "Synthesize draft" : "Synthesize full") { backend.render(song, engine: "auto", quality: quality, engines: engines) }.disabled(!backend.connected)
            } else if song.quality == "draft" && song.status == .ready {
                Button("Render full quality") { players.forget(song); backend.render(song, engine: "auto", quality: "full", engines: engines) }.disabled(!backend.connected)
            }
            if !song.score.isEmpty { Button("Score") { showScore = song } }
            Button("Reveal") { NSWorkspace.shared.activateFileViewerSelecting([URL(fileURLWithPath: song.path)]) }.disabled(song.status != .ready)
        }.padding(.vertical, 2)
    }

    /// A grab bar under a text editor: drag down to make it taller; the height persists.
    private func resizeHandle(height: Binding<Double>, dragStart: Binding<Double?>, range: ClosedRange<Double>) -> some View {
        HStack { Spacer(); Capsule().fill(Color.secondary.opacity(0.35)).frame(width: 44, height: 4); Spacer() }
            .frame(height: 10).contentShape(Rectangle())
            .onHover { inside in if inside { NSCursor.resizeUpDown.push() } else { NSCursor.pop() } }
            .gesture(DragGesture(minimumDistance: 1, coordinateSpace: .global)
                .onChanged { v in
                    let start = dragStart.wrappedValue ?? height.wrappedValue
                    dragStart.wrappedValue = start
                    height.wrappedValue = min(range.upperBound, max(range.lowerBound, start + v.translation.height))
                }
                .onEnded { _ in dragStart.wrappedValue = nil })
    }

    /// The divider above the log: drag up or down to resize it; the height persists.
    private var logSplitter: some View {
        ZStack { Divider() }
            .frame(height: 9)
            .contentShape(Rectangle())
            .onHover { inside in
                if inside { NSCursor.resizeUpDown.push() } else { NSCursor.pop() }
            }
            .gesture(DragGesture(minimumDistance: 1, coordinateSpace: .global)
                .onChanged { v in
                    let start = logDragStart ?? logPanelHeight
                    logDragStart = start
                    logPanelHeight = min(600, max(64, start - v.translation.height))
                }
                .onEnded { _ in logDragStart = nil })
    }

    private var logView: some View {
        VStack(alignment: .leading, spacing: 0) {
            HStack {
                Text("Log").font(.caption).bold(); Spacer()
                Button("Copy all") {
                    let text = backend.log.map { "\($0.time)  \($0.message)" }.joined(separator: "\n")
                    NSPasteboard.general.clearContents(); NSPasteboard.general.setString(text, forType: .string)
                }.font(.caption)
                Button("Clear") { backend.log.removeAll() }.font(.caption)
            }.padding(.horizontal, 8).padding(.vertical, 4)
            LogTextView(lines: backend.log)
        }
    }
}
