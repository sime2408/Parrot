import AVFoundation
import Foundation

// Live harness — the transcription loop and the live agent, measured without
// Xcode. The app itself needs Xcode (SwiftData macros); these engine sources
// don't, so they're symlinked in and built with Command Line Tools:
//
//   cd scripts/dev/live-harness
//   swift build -c release --disable-keychain     # no keychain prompt for binary artifacts
//   ./make-test-call.sh /tmp/call                 # bilingual test call + a docs folder
//
//   .build/release/LiveHarness selftest                              # LiveLatencyChecks
//   .build/release/LiveHarness stt /tmp/call/meeting.wav [parakeet|local] [whisperModel]
//   DOCS=/tmp/call/docs PARROT_AGENT_TRACE=1 \
//     .build/release/LiveHarness agent /tmp/call/meeting.wav [ollamaModel]
//
// `stt` prints each preview and commit against the audio clock; `agent` runs
// Parakeet into the LiveAgent on a local Ollama and prints question → first
// token latencies. Whisper models load from ~/Documents/huggingface (set
// CFFIXED_USER_HOME to point elsewhere); Parakeet downloads on first run.

func loadSamples16k(path: String) throws -> [Float] {
    let file = try AVAudioFile(forReading: URL(fileURLWithPath: path))
    let target = AVAudioFormat(commonFormat: .pcmFormatFloat32, sampleRate: 16000, channels: 1, interleaved: false)!
    let inBuf = AVAudioPCMBuffer(pcmFormat: file.processingFormat, frameCapacity: AVAudioFrameCount(file.length))!
    try file.read(into: inBuf)
    let converter = AVAudioConverter(from: file.processingFormat, to: target)!
    let outCap = AVAudioFrameCount(Double(file.length) * 16000 / file.processingFormat.sampleRate) + 1024
    let outBuf = AVAudioPCMBuffer(pcmFormat: target, frameCapacity: outCap)!
    var fed = false
    var convError: NSError?
    converter.convert(to: outBuf, error: &convError) { _, status in
        if fed { status.pointee = .endOfStream; return nil }
        fed = true
        status.pointee = .haveData
        return inBuf
    }
    return Array(UnsafeBufferPointer(start: outBuf.floatChannelData![0], count: Int(outBuf.frameLength)))
}

func pcmBuffer(_ samples: [Float]) -> AVAudioPCMBuffer {
    let format = AVAudioFormat(commonFormat: .pcmFormatFloat32, sampleRate: 16000, channels: 1, interleaved: false)!
    let buffer = AVAudioPCMBuffer(pcmFormat: format, frameCapacity: AVAudioFrameCount(samples.count))!
    buffer.frameLength = AVAudioFrameCount(samples.count)
    samples.withUnsafeBufferPointer { src in
        buffer.floatChannelData![0].update(from: src.baseAddress!, count: samples.count)
    }
    return buffer
}

@MainActor
func runSTT(args: [String]) async {
    let path = args[0]
    let backend = args.count > 1 ? args[1] : "parakeet"
    let whisperModel = args.count > 2 ? args[2] : "large-v3-turbo"
    UserDefaults.standard.set(backend, forKey: TranscriptionBackend.defaultsKey)
    UserDefaults.standard.set(ProcessInfo.processInfo.environment["LANG_SETTING"] ?? "auto", forKey: "transcriptionLanguage")
    UserDefaults.standard.set(ProcessInfo.processInfo.environment["VOCAB"] ?? "", forKey: "customVocabulary")

    let engine = TranscriptionEngine()
    let loadStart = Date()
    await engine.loadLiveEngine(whisperModel: whisperModel)
    print(String(format: "load %@: %.2fs ready=%@", backend, Date().timeIntervalSince(loadStart), engine.isReady ? "yes" : "no"))
    guard engine.isReady else { print("state: \(engine.liveModelState)"); exit(1) }

    let samples: [Float]
    do { samples = try loadSamples16k(path: path) } catch { print("audio: \(error)"); exit(1) }

    var clockStart = Date()
    var commits: [(wall: Double, start: Double, end: Double, text: String)] = []
    engine.onSegment = { r in
        commits.append((Date().timeIntervalSince(clockStart), r.startTime, r.endTime, r.text))
        print(String(format: "  COMMIT at %6.2fs  audio [%5.2f–%5.2f]  lag %.2fs  %@",
                     Date().timeIntervalSince(clockStart), r.startTime, r.endTime,
                     Date().timeIntervalSince(clockStart) - r.endTime, r.text))
    }

    engine.startTranscribing(meetingStartTime: .now)
    clockStart = Date()
    // Watch preview text on a side task.
    var lastPreview = ""
    var previews: [(wall: Double, text: String)] = []
    let watcher = Task { @MainActor in
        while !Task.isCancelled {
            let text = engine.currentText
            if !text.isEmpty, text != lastPreview {
                lastPreview = text
                let wall = Date().timeIntervalSince(clockStart)
                previews.append((wall, text))
                print(String(format: "  preview %6.2fs  %@", wall, text))
            }
            try? await Task.sleep(for: .milliseconds(30))
        }
    }

    let slice = 1600  // 100 ms, fed at recording pace
    var i = 0
    while i < samples.count {
        let end = min(i + slice, samples.count)
        let target = clockStart.addingTimeInterval(Double(end) / 16000)
        engine.appendAudio(pcmBuffer(Array(samples[i..<end])), source: .them)
        let wait = target.timeIntervalSinceNow
        if wait > 0 { try? await Task.sleep(for: .milliseconds(Int(wait * 1000))) }
        i = end
    }
    // Let the final pause register before draining.
    try? await Task.sleep(for: .seconds(1.0))
    await engine.stopTranscribing()
    try? await Task.sleep(for: .milliseconds(300))
    watcher.cancel()

    let lags = commits.map { $0.wall - $0.end }.sorted()
    func pct(_ p: Double) -> Double { lags.isEmpty ? .nan : lags[min(lags.count - 1, Int(Double(lags.count - 1) * p))] }
    print(String(format: "SUMMARY %@: %d commits, speech-end→commit lag median %.2fs p90 %.2fs max %.2fs; %d preview updates",
                 backend, commits.count, pct(0.5), pct(0.9), lags.last ?? .nan, previews.count))
}

// LiveHarness agent <audio> [model]
// Real-time end to end: audio → Parakeet → LiveAgent (Ollama). Prints when each
// line committed, when the agent's NOW line changed, and how long each answer
// took from the question's commit to its first token and to completion.
@MainActor
func runAgent(args: [String]) async {
    let path = args[0]
    let model = args.count > 1 ? args[1] : "qwen3.5:9b"
    UserDefaults.standard.set("parakeet", forKey: TranscriptionBackend.defaultsKey)
    UserDefaults.standard.set("auto", forKey: "transcriptionLanguage")
    UserDefaults.standard.set("OCTO,Syntio,Datatonic,ASDLC", forKey: "customVocabulary")

    let engine = TranscriptionEngine()
    await engine.loadLiveEngine(whisperModel: "base")
    guard engine.isReady else { print("parakeet not ready"); exit(1) }
    let samples: [Float]
    do { samples = try loadSamples16k(path: path) } catch { print("audio: \(error)"); exit(1) }

    let agent = LiveAgent()
    var clock = Date()
    func stamp() -> String { String(format: "%6.2fs", Date().timeIntervalSince(clock)) }
    // Real knowledge base over a docs folder (DOCS env), or none.
    let kb = KnowledgeBaseService(persistent: false)
    if let docsPath = ProcessInfo.processInfo.environment["DOCS"] {
        let started = Date()
        await kb.addFolder(at: URL(fileURLWithPath: docsPath))
        print(String(format: "kb: %d files indexed in %.2fs, error=%@", kb.folders.first.map { kb.documentCount(in: $0) } ?? 0,
                     Date().timeIntervalSince(started), kb.lastError ?? "-"))
    }
    agent.retrieve = { query in
        let started = Date()
        let refs = await kb.search(query: query, topK: 3)
        print(String(format: "  retrieve %.0f ms → %@", Date().timeIntervalSince(started) * 1000,
                     refs.map(\.documentName).joined(separator: ", ")))
        return refs.map { LiveAgent.Reference(id: $0.chunkID?.uuidString ?? $0.documentName, documentName: $0.documentName, text: $0.text) }
    }
    var answers: [(question: String, latency: Double?, total: Double)] = []
    var questionCommitted: [String: Date] = [:]
    agent.onOutput = { output in
        switch output {
        case .answer(let question, let text, let source, _, _):
            let total = questionCommitted[question].map { Date().timeIntervalSince($0) } ?? -1
            answers.append((question, agent.lastAnswerLatency, total))
            print("  \(stamp()) ANSWER (first token \(String(format: "%.2f", agent.lastAnswerLatency ?? -1))s, done \(String(format: "%.2f", total))s) [\(source ?? "-")] Q: \(question)\n           → \(text)")
        case .ask(let question, _, _):
            print("  \(stamp()) ASK: \(question)")
        case .note(let text, let source, _, _):
            print("  \(stamp()) NOTE [\(source ?? "-")]: \(text)")
        }
    }
    let startAgent = Date()
    agent.start(model: model, context: .init(counterpart: "the other person",
                                              glossary: ["OCTO", "Syntio", "Datatonic", "ASDLC"],
                                              documentNames: kb.folders.map { "\($0.name)/ (folder, \($0.fileCount) files)" }))
    while agent.status == .starting { try? await Task.sleep(for: .milliseconds(50)) }
    print(String(format: "agent warm in %.2fs, status=%@, now=%@", Date().timeIntervalSince(startAgent),
                 "\(agent.status)", agent.nowLine ?? "-"))

    engine.onSegment = { r in
        print("  \(stamp()) COMMIT \(r.source.label): \(r.text)")
        if r.source == .them, LiveAgent.looksLikeQuestion(r.text) { questionCommitted[r.text] = Date() }
        agent.ingest(text: r.text, at: r.endTime, source: r.source)
    }
    engine.startTranscribing(meetingStartTime: .now)
    clock = Date()
    var lastNow = agent.nowLine
    let watcher = Task { @MainActor in
        while !Task.isCancelled {
            if agent.nowLine != lastNow {
                lastNow = agent.nowLine
                print("  \(stamp()) NOW: \(lastNow ?? "-")")
            }
            try? await Task.sleep(for: .milliseconds(50))
        }
    }
    let slice = 1600
    var i = 0
    while i < samples.count {
        let end = min(i + slice, samples.count)
        let target = clock.addingTimeInterval(Double(end) / 16000)
        engine.appendAudio(pcmBuffer(Array(samples[i..<end])), source: .them)
        let wait = target.timeIntervalSinceNow
        if wait > 0 { try? await Task.sleep(for: .milliseconds(Int(wait * 1000))) }
        i = end
    }
    try? await Task.sleep(for: .seconds(1))
    await engine.stopTranscribing()
    // Let the last answer finish.
    for _ in 0..<200 where agent.isAnswering { try? await Task.sleep(for: .milliseconds(50)) }
    try? await Task.sleep(for: .seconds(4))
    watcher.cancel()
    let firsts = answers.compactMap(\.latency).sorted()
    print(String(format: "SUMMARY: %d answers; question commit → first token median %.2fs max %.2fs; last turn %@",
                 answers.count, firsts.isEmpty ? .nan : firsts[firsts.count / 2], firsts.last ?? .nan,
                 "\(String(describing: agent.lastStats))"))
    agent.stop()
}

let args = Array(CommandLine.arguments.dropFirst())
guard let command = args.first else {
    print("usage: LiveHarness stt <audio> [parakeet|local] [whisperModel] | agent <audio> [model]")
    exit(2)
}
Task { @MainActor in
    switch command {
    case "stt": await runSTT(args: Array(args.dropFirst()))
    case "agent": await runAgent(args: Array(args.dropFirst()))
    case "selftest":
        var failures = 0
        LiveLatencyChecks.run { name, ok in
            print("\(ok ? "PASS" : "FAIL") \(name)")
            if !ok { failures += 1 }
        }
        print(failures == 0 ? "ALL PASS" : "FAILURES: \(failures)")
        exit(failures == 0 ? 0 : 1)
    default: print("unknown command \(command)")
    }
    exit(0)
}
dispatchMain()
