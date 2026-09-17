import Foundation
import FluidAudio

/// Logic checks for instant transcription and the live agent — pure: no
/// network, no models, no SwiftData. `--profile-test` runs them, and so can a
/// Command-Line-Tools-only harness (this file compiles without SwiftUI).
enum LiveLatencyChecks {
    @MainActor
    static func run(_ check: (String, Bool) -> Void) {
        backendAndLanguage(check)
        glossaryRespelling(check)
        replyParsing(check)
        questionDetection(check)
        prompts(check)
        keywordIndex(check)
        chunkingAndExtraction(check)
        folderScan(check)
        ollamaModelMatch(check)
    }

    static func backendAndLanguage(_ check: (String, Bool) -> Void) {
        check("parakeet backend raw value", TranscriptionBackend(rawValue: "parakeet") == .parakeet)
        check("parakeet is on-device", TranscriptionBackend.parakeet.isOnDevice)
        check("whisper is on-device", TranscriptionBackend.local.isOnDevice)
        check("deepgram is not on-device", !TranscriptionBackend.deepgram.isOnDevice)
        check("parakeet needs no key", TranscriptionBackend.parakeet.keychainAccount == nil)
        check("language hint from setting", ParakeetTranscriber.languageHint(setting: "hr") == .croatian)
        check("auto hint follows Mac language",
              ParakeetTranscriber.languageHint(setting: "auto", preferredLanguages: ["hr-HR", "en-US"]) == .croatian)
        check("auto hint skips languages Parakeet lacks",
              ParakeetTranscriber.languageHint(setting: nil, preferredLanguages: ["ja-JP", "en-GB"]) == .english)
        check("auto hint nil when nothing maps",
              ParakeetTranscriber.languageHint(setting: "auto", preferredLanguages: ["ja-JP"]) == nil)
        check("preview waits for 800 ms of audio", TranscriptionEngine.Segmenter.minPreviewSamples == 12_800)
    }

    static func glossaryRespelling(_ check: (String, Bool) -> Void) {
        let terms = TranscriptionEngine.glossaryTerms(from: "OCTO, Syntio,\nDatatonic,ASDLC,")
        check("glossary terms parse", terms == ["OCTO", "Syntio", "Datatonic", "ASDLC"])
        func spell(_ text: String) -> String { TranscriptionEngine.respellingGlossary(text, terms: terms) }
        check("split word rejoins", spell("we work at data tonic now") == "we work at Datatonic now")
        check("spelled acronym rejoins", spell("the a s d l c review") == "the ASDLC review")
        check("dotted acronym rejoins", spell("the A.S.D.L.C. review") == "the ASDLC. review")
        check("case fixed", spell("octo is live") == "OCTO is live")
        check("possessive kept", spell("syntio's team") == "Syntio's team")
        check("longer word untouched", spell("October planning") == "October planning")
        check("misheard word untouched", spell("sintio called") == "sintio called")
        check("no terms no change", TranscriptionEngine.respellingGlossary("data tonic", terms: []) == "data tonic")
    }

    static func replyParsing(_ check: (String, Bool) -> Void) {
        let full = LiveAgentReply.parse("""
        NOW: BigQuery reservation costs
        ASK: Who signs off on the reservation?
        """)
        check("parse NOW", full.now == "BigQuery reservation costs")
        check("parse ASK", full.asks == ["Who signs off on the reservation?"])
        check("parse no answer", full.answer == nil)

        let dressed = LiveAgentReply.parse("""
        - **ANSWER:** About 4,100 USD a month
        on a 500-slot reservation.
        * Source: bigquery-costs.md
        """)
        check("parse bold bullet tag", dressed.answer == "About 4,100 USD a month on a 500-slot reservation.")
        check("parse lowercase source", dressed.sources == ["bigquery-costs.md"])

        let croatian = LiveAgentReply.parse("ODGOVOR: Oko 4000 dolara mjesečno.\nIZVOR: troskovi.md")
        check("parse croatian tags", croatian.answer == "Oko 4000 dolara mjesečno." && croatian.sources == ["troskovi.md"])

        let streaming = LiveAgentReply.parse("NOW: Migration timeline\nNOTE")
        check("partial next tag not glued", streaming.now == "Migration timeline" && streaming.notes.isEmpty)

        let untagged = LiveAgentReply.parse("The call covered costs and approvals.")
        check("untagged falls back", untagged.plainFallback == "The call covered costs and approvals." && untagged.now == nil)

        let noneSource = LiveAgentReply.parse("ANSWER: Yes.\nSOURCE: none")
        check("source none dropped", noneSource.sources.isEmpty)

        let clock = LiveAgentReply.parse("ANSWER: The call is at 10:30 tomorrow.")
        check("colon inside value kept", clock.answer == "The call is at 10:30 tomorrow.")
    }

    static func questionDetection(_ check: (String, Bool) -> Void) {
        check("question mark", LiveAgent.looksLikeQuestion("And the budget?"))
        check("english opener", LiveAgent.looksLikeQuestion("how much will the reservation cost"))
        check("croatian opener", LiveAgent.looksLikeQuestion("Koliko će koštati rezervacija"))
        check("croatian je li", LiveAgent.looksLikeQuestion("je li to odobreno"))
        check("statement is not a question", !LiveAgent.looksLikeQuestion("We shipped the dbt models last week."))
        check("croatian statement", !LiveAgent.looksLikeQuestion("Platformski tim predlaže, financije odobravaju."))
        check("opener needs word boundary", !LiveAgent.looksLikeQuestion("Showcase the whyte report"))
    }

    static func prompts(_ check: (String, Bool) -> Void) {
        let context = LiveAgent.CallContext(counterpart: "the prospect", brief: "Renewal call",
                                            glossary: ["Datatonic"], documentNames: ["pricing.md"])
        let system = LiveAgent.systemPrompt(context)
        check("system names counterpart", system.contains("\"Them\" are the prospect"))
        check("system carries glossary", system.contains("Datatonic"))
        check("system lists documents", system.contains("pricing.md"))
        check("system carries brief", system.contains("Renewal call"))
        check("system is stable", system == LiveAgent.systemPrompt(context))

        let lines: [LiveAgent.Line] = [(1, .them, "What does it cost?"), (2, .me, "Let me check.")]
        let reference = LiveAgent.Reference(id: "a", documentName: "pricing.md",
                                            text: String(repeating: "x", count: 800))
        let message = LiveAgent.userMessage(lines: lines, references: [reference],
                                            referenceCharacters: 100, directive: "[question] Them asked: \"What does it cost?\"")
        check("user message transcript", message.hasPrefix("<transcript>\nThem: What does it cost?\nMe: Let me check.\n</transcript>\n"))
        check("user message excerpt trimmed", message.contains(String(repeating: "x", count: 100) + "…")
              && !message.contains(String(repeating: "x", count: 101)))
        check("user message ends with directive", message.hasSuffix("[question] Them asked: \"What does it cost?\""))
    }

    static func keywordIndex(_ check: (String, Bool) -> Void) {
        let texts = [
            "BigQuery reservation costs about 4,100 USD per month after the cutover.",
            "The ASDLC review checks security controls, lineage and the rollback plan.",
            "Ako pregled prođe sljedeći tjedan, marketing domena ide u produkciju.",
            "Team lunch is on Friday at the usual place.",
        ]
        let index = LexicalIndex(texts: texts)
        let cost = index.search("what will the reservation cost after cutover", limit: 3)
        check("bm25 finds cost chunk first", cost.first?.index == 0)
        check("bm25 skips unrelated chunk", !cost.contains { $0.index == 3 })
        let accent = index.search("kada prode pregled", limit: 3)
        check("bm25 folds diacritics", accent.first?.index == 2)
        let rare = index.search("asdlc", limit: 3)
        check("bm25 single rare term matches", rare.first?.index == 1)
        check("bm25 stopwords only finds nothing", index.search("the and what is", limit: 3).isEmpty)
        check("tokens fold case and accents", LexicalIndex.tokens("Prođe ČAK") == ["prode", "cak"])

        let fused = KnowledgeBaseService.reciprocalRankFusion([[3, 1, 2], [1, 4]])
        check("rrf rewards agreement", fused.first == 1)
        check("rrf keeps every item", Set(fused) == [1, 2, 3, 4])
    }

    static func chunkingAndExtraction(_ check: (String, Bool) -> Void) {
        let long = (0..<60).map { "line \($0) with some words in it" }.joined(separator: "\n")
        let pieces = KnowledgeBaseService.chunkText(long)
        check("long paragraph splits", pieces.count > 1)
        check("chunks stay bounded", pieces.allSatisfy { $0.count <= 900 })
        let stripped = KnowledgeBaseService.stripHTML("<html><style>p{}</style><p>Hello&nbsp;<b>team</b></p><script>x()</script></html>")
        check("html stripped", stripped.contains("Hello") && stripped.contains("team")
              && !stripped.contains("<") && !stripped.contains("x()") && !stripped.contains("p{}"))
        check("secrets skipped", KnowledgeBaseService.isSecretLooking(".env.local")
              && KnowledgeBaseService.isSecretLooking("server.pem")
              && !KnowledgeBaseService.isSecretLooking("meeting-notes.md"))
    }

    static func folderScan(_ check: (String, Bool) -> Void) {
        let fm = FileManager.default
        let root = fm.temporaryDirectory.appendingPathComponent("parrot-kb-\(UUID().uuidString)", isDirectory: true)
        defer { try? fm.removeItem(at: root) }
        do {
            try fm.createDirectory(at: root.appendingPathComponent("node_modules/pkg"), withIntermediateDirectories: true)
            try fm.createDirectory(at: root.appendingPathComponent("specs"), withIntermediateDirectories: true)
            let body = String(repeating: "The migration plan covers reservations and rollback. ", count: 4)
            try body.write(to: root.appendingPathComponent("notes.md"), atomically: true, encoding: .utf8)
            try body.write(to: root.appendingPathComponent("specs/plan.txt"), atomically: true, encoding: .utf8)
            try "SECRET=1".write(to: root.appendingPathComponent(".env"), atomically: true, encoding: .utf8)
            try body.write(to: root.appendingPathComponent("node_modules/pkg/readme.md"), atomically: true, encoding: .utf8)
            try Data([0, 1, 2]).write(to: root.appendingPathComponent("image.png"))
        } catch {
            check("folder scan fixture", false)
            return
        }
        let first = KnowledgeBaseService.scanFolder(at: root, known: [:])
        check("scan finds documents", first.present == ["notes.md", "specs/plan.txt"])
        check("scan reads new files", first.changed.count == 2 && first.changed.allSatisfy { !$0.pieces.isEmpty })
        let known = Dictionary(uniqueKeysWithValues: first.changed.map { ($0.relativePath, $0.modifiedAt) })
        let second = KnowledgeBaseService.scanFolder(at: root, known: known)
        check("rescan skips unchanged", second.changed.isEmpty && second.present.count == 2)
    }

    static func ollamaModelMatch(_ check: (String, Bool) -> Void) {
        check("exact tag installed", OllamaChatClient.isInstalled("qwen3.5:9b", in: ["qwen3.5:9b", "llama3.2:latest"]))
        check("bare name means latest", OllamaChatClient.isInstalled("llama3.2", in: ["llama3.2:latest"]))
        check("similar name is not installed", !OllamaChatClient.isInstalled("qwen3.5:9b", in: ["qwen3.5-mem:9b"]))
    }
}
