import Foundation
import FluidAudio

/// On-device NVIDIA Parakeet TDT 0.6B v3 through FluidAudio — the instant live
/// engine. FluidAudio's FLEURS run on an M4 Pro measured ~205× real time for
/// Croatian and ~207× for English (25 European languages, punctuation and
/// casing built in). WhisperKit pads every decode to a 30 s window, so each
/// rolling preview of an open utterance cost a full large-model pass; here a
/// 12 s utterance decodes in tens of milliseconds, which is what lets words
/// show up while they're still being spoken.
///
/// Models (~460 MB) download to Application Support on first use, next to the
/// diarization models.
final class ParakeetTranscriber {
    enum TranscriberError: LocalizedError {
        case notLoaded

        var errorDescription: String? {
            switch self {
            case .notLoaded: "Parakeet model is not loaded yet."
            }
        }
    }

    private var manager: AsrManager?

    var isLoaded: Bool { manager != nil }

    static var modelsInstalled: Bool {
        AsrModels.modelsExist(at: AsrModels.defaultCacheDirectory(for: .v3), version: .v3)
    }

    /// Downloads (first run only) and loads the v3 models. `progress` gets the
    /// operation's monotonic fraction in [0, 1] and whether the bytes are in
    /// and CoreML is compiling (the slow part of a first launch).
    func load(progress: (@Sendable (_ fraction: Double, _ compiling: Bool) -> Void)? = nil) async throws {
        let models = try await AsrModels.downloadAndLoad(version: .v3) { update in
            if case .compiling = update.phase {
                progress?(update.fractionCompleted, true)
            } else {
                progress?(update.fractionCompleted, false)
            }
        }
        let manager = AsrManager(config: .default)
        try await manager.loadModels(models)
        self.manager = manager
    }

    /// One stateless decode of 16 kHz mono samples. Every call starts from a
    /// fresh decoder state, so re-decoding a growing utterance (the live
    /// preview) never inherits context from the previous pass.
    func transcribe(_ samples: [Float], language: Language?) async throws -> (text: String, confidence: Float) {
        guard let manager else { throw TranscriberError.notLoaded }
        // The model refuses clips under 0.3 s; a clipped "yes" is still speech,
        // so pad with silence instead of dropping it.
        let minimum = ASRConstants.minimumRequiredSamples(forSampleRate: 16_000)
        var input = samples
        if input.count < minimum {
            input.append(contentsOf: [Float](repeating: 0, count: minimum - input.count))
        }
        var state = try TdtDecoderState(decoderLayers: await manager.decoderLayerCount)
        let result = try await manager.transcribe(input, decoderState: &state, language: language)
        return (result.text, result.confidence)
    }

    /// Script hint for the decoder: keeps a Slavic speaker's words out of
    /// Cyrillic tokens (FluidAudio issue #512). Latin-script hints keep the
    /// full Latin Extended range, so English and Croatian can share one call.
    /// "auto" falls back to the Mac's own language when Parakeet knows it.
    static func languageHint(setting: String?, preferredLanguages: [String] = Locale.preferredLanguages) -> Language? {
        if let setting, setting != "auto" {
            return Language(rawValue: setting)
        }
        for tag in preferredLanguages {
            let code = String(tag.prefix { $0 != "-" && $0 != "_" }).lowercased()
            if let language = Language(rawValue: code) { return language }
        }
        return nil
    }
}
