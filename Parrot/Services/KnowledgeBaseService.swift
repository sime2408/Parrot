import Foundation
import NaturalLanguage
import Observation
import PDFKit

/// On-device knowledge base: documents are chunked and indexed locally, then
/// matched against the live conversation. Nothing here ever touches the
/// network — only the few best-matching chunks are later included in copilot
/// requests (to a local model, unless the user chose a cloud one).
///
/// Two ways in: documents added one by one (embedded with the NaturalLanguage
/// framework, as before), and **folders** — indexed whole, kept readable through
/// a security-scoped bookmark, and re-checked for new, changed and deleted
/// files before each call. Retrieval ranks chunks by keywords (BM25, accent-
/// and case-insensitive, so it works for Croatian as well as English and for
/// names and jargon embeddings miss) and, where a chunk has one, by embedding
/// similarity; the two rankings are fused.
@MainActor
@Observable
final class KnowledgeBaseService {
    private(set) var documents: [KBDocument] = []
    private(set) var folders: [KBFolder] = []
    private(set) var isIndexing = false
    /// Progress line while a folder indexes ("Indexing Notes — 120 files").
    private(set) var indexingStatus: String?
    private(set) var lastError: String?

    private var chunks: [KBChunk] = []
    /// Bumped on every chunk change; the keyword index rebuilds lazily.
    @ObservationIgnored private var chunksVersion = 0
    @ObservationIgnored private var lexical: (version: Int, index: LexicalIndex)?
    @ObservationIgnored private var saveTask: Task<Void, Never>?

    var isEmpty: Bool { documents.isEmpty }

    private let persistent: Bool

    init(persistent: Bool = true) {
        self.persistent = persistent
        if persistent { load() }
    }

    // MARK: - Document Management

    func addDocuments(at urls: [URL]) async {
        isIndexing = true
        lastError = nil
        for url in urls {
            await addDocument(at: url)
        }
        isIndexing = false
    }

    private func addDocument(at url: URL) async {
        let accessing = url.startAccessingSecurityScopedResource()
        defer {
            if accessing { url.stopAccessingSecurityScopedResource() }
        }

        let name = url.lastPathComponent
        guard let text = Self.extractText(from: url), !text.isEmpty else {
            lastError = "Couldn't read \(name)"
            return
        }

        let pieces = Self.chunkText(text)
        let language = NLLanguageRecognizer.dominantLanguage(for: text) ?? .english

        let embedded: [KBChunk] = await Task.detached(priority: .userInitiated) {
            pieces.compactMap { piece in
                guard let vector = Self.embed(piece, language: language) else { return nil }
                return KBChunk(
                    documentName: name,
                    languageRaw: language.rawValue,
                    text: piece,
                    embedding: vector
                )
            }
        }.value

        guard !embedded.isEmpty else {
            lastError = "No embeddable text in \(name) — the document language may not be supported on this Mac"
            return
        }

        // Re-adding a document replaces its previous version, keeping its note.
        let existingNote = documents.first { $0.name == name }?.note ?? ""
        chunks.removeAll { $0.documentName == name }
        documents.removeAll { $0.name == name }
        chunks.append(contentsOf: embedded)
        documents.append(KBDocument(name: name, note: existingNote, chunkCount: embedded.count, addedAt: .now))
        chunksVersion += 1
        save()
    }

    func removeDocument(_ document: KBDocument) {
        documents.removeAll { $0.id == document.id }
        chunks.removeAll { $0.documentName == document.name }
        chunksVersion += 1
        save()
    }

    func updateNote(_ note: String, for document: KBDocument) {
        guard let index = documents.firstIndex(where: { $0.id == document.id }) else { return }
        documents[index].note = note
        save()
    }

    // MARK: - Folders

    /// Hand-added documents only — folder files are listed per folder.
    var standaloneDocuments: [KBDocument] { documents.filter { $0.folderID == nil } }

    func documentCount(in folder: KBFolder) -> Int {
        documents.filter { $0.folderID == folder.id && $0.chunkCount > 0 }.count
    }

    /// Points the knowledge base at a folder the user picked: keeps access to
    /// it with a security-scoped bookmark and indexes its files. Picking the
    /// same folder again re-indexes it.
    func addFolder(at url: URL) async {
        let accessing = url.startAccessingSecurityScopedResource()
        defer { if accessing { url.stopAccessingSecurityScopedResource() } }
        lastError = nil
        let bookmark: Data
        do {
            bookmark = try url.bookmarkData(options: .withSecurityScope,
                                            includingResourceValuesForKeys: nil, relativeTo: nil)
        } catch {
            lastError = "Couldn't keep access to \(url.lastPathComponent): \(error.localizedDescription)"
            return
        }
        var existing: KBFolder?
        for candidate in folders where resolve(candidate)?.standardizedFileURL == url.standardizedFileURL {
            existing = candidate
            break
        }
        let folder: KBFolder
        if let existing {
            folder = existing
            updateFolder(existing.id) { $0.bookmark = bookmark }
        } else {
            folder = KBFolder(name: Self.uniqueFolderName(url.lastPathComponent, taken: folders.map(\.name)),
                              bookmark: bookmark)
            folders.append(folder)
        }
        await index(folderID: folder.id, at: url)
    }

    func removeFolder(_ folder: KBFolder) {
        folders.removeAll { $0.id == folder.id }
        let names = Set(documents.filter { $0.folderID == folder.id }.map(\.name))
        documents.removeAll { $0.folderID == folder.id }
        chunks.removeAll { names.contains($0.documentName) }
        chunksVersion += 1
        save()
    }

    /// Re-checks every folder for new, changed and deleted files. Cheap when
    /// nothing changed (a directory walk and modification dates), so it runs
    /// at launch and again when a recording starts.
    func rescanFolders() async {
        guard !isIndexing else { return }
        for folder in folders {
            guard let url = resolve(folder) else {
                lastError = "Lost access to the folder \(folder.name) — remove it and add it again."
                continue
            }
            let accessing = url.startAccessingSecurityScopedResource()
            await index(folderID: folder.id, at: url)
            if accessing { url.stopAccessingSecurityScopedResource() }
        }
    }

    private func resolve(_ folder: KBFolder) -> URL? {
        var stale = false
        guard let url = try? URL(resolvingBookmarkData: folder.bookmark, options: .withSecurityScope,
                                 relativeTo: nil, bookmarkDataIsStale: &stale) else { return nil }
        if stale {
            let accessing = url.startAccessingSecurityScopedResource()
            if let fresh = try? url.bookmarkData(options: .withSecurityScope,
                                                 includingResourceValuesForKeys: nil, relativeTo: nil) {
                updateFolder(folder.id) { $0.bookmark = fresh }
            }
            if accessing { url.stopAccessingSecurityScopedResource() }
        }
        return url
    }

    private func updateFolder(_ id: UUID, _ change: (inout KBFolder) -> Void) {
        guard let index = folders.firstIndex(where: { $0.id == id }) else { return }
        change(&folders[index])
    }

    /// Walks a folder off the main thread and applies the difference: files
    /// whose modification date is unchanged are skipped, changed files are
    /// re-chunked, deleted ones disappear.
    private func index(folderID: UUID, at url: URL) async {
        guard let folder = folders.first(where: { $0.id == folderID }) else { return }
        isIndexing = true
        indexingStatus = "Indexing \(folder.name)…"
        defer {
            isIndexing = false
            indexingStatus = nil
        }

        let known: [String: Date] = Dictionary(
            documents.filter { $0.folderID == folderID }
                .compactMap { doc in doc.relativePath.map { ($0, doc.modifiedAt ?? .distantPast) } },
            uniquingKeysWith: { first, _ in first })
        let folderName = folder.name

        let scan = await Task.detached(priority: .utility) {
            Self.scanFolder(at: url, known: known)
        }.value

        // Apply: drop deleted and changed files, add the (re)read ones.
        let removedPaths = Set(known.keys).subtracting(scan.present).union(scan.changed.map(\.relativePath))
        if !removedPaths.isEmpty {
            let removedNames = Set(documents
                .filter { $0.folderID == folderID && removedPaths.contains($0.relativePath ?? "") }
                .map(\.name))
            documents.removeAll { removedNames.contains($0.name) }
            chunks.removeAll { removedNames.contains($0.documentName) }
        }
        for file in scan.changed {
            let name = "\(folderName)/\(file.relativePath)"
            chunks.append(contentsOf: file.pieces.map {
                KBChunk(documentName: name, languageRaw: file.languageRaw, text: $0, embedding: [])
            })
            // Unreadable files are kept with zero chunks so their modification
            // date is remembered and they aren't re-read on every scan.
            documents.append(KBDocument(name: name, chunkCount: file.pieces.count, addedAt: .now,
                                        folderID: folderID, relativePath: file.relativePath,
                                        modifiedAt: file.modifiedAt))
        }
        let indexedFiles = documents.filter { $0.folderID == folderID && $0.chunkCount > 0 }.count
        updateFolder(folderID) {
            $0.lastIndexedAt = .now
            $0.fileCount = indexedFiles
        }
        if scan.truncated {
            lastError = "\(folderName) has more files than the knowledge base indexes (\(Self.maxFilesPerFolder)); the rest are skipped."
        }
        if !removedPaths.isEmpty || !scan.changed.isEmpty {
            chunksVersion += 1
        }
        save()
    }

    // MARK: Folder scanning (off the main actor)

    struct ScannedFile: Sendable {
        let relativePath: String
        let modifiedAt: Date
        let languageRaw: String
        let pieces: [String]
    }

    struct FolderScan: Sendable {
        /// Every indexable file currently in the folder.
        var present: Set<String> = []
        /// New or modified files, read and chunked.
        var changed: [ScannedFile] = []
        var truncated = false
    }

    nonisolated static let maxFilesPerFolder = 2_000
    nonisolated static let maxChunksPerFile = 80

    /// Directories that never hold anything a meeting needs.
    nonisolated static let skippedDirectories: Set<String> = [
        "node_modules", ".git", ".build", "build", "dist", "DerivedData", "Pods", ".venv", "venv",
        "__pycache__", "target", ".next", ".cache", "vendor",
    ]

    nonisolated static let plainTextExtensions: Set<String> = [
        "txt", "md", "markdown", "mdx", "rst", "org", "tex", "csv", "tsv", "json", "yaml", "yml",
        "toml", "xml", "sql", "py", "swift", "ts", "tsx", "js", "jsx", "java", "kt", "go", "rs",
        "rb", "php", "c", "h", "cpp", "hpp", "cs", "sh", "tf", "hcl", "vtt", "srt", "ini", "cfg",
    ]
    nonisolated static let richTextExtensions: Set<String> = ["pdf", "docx", "doc", "rtf", "odt", "html", "htm"]

    /// Files that tend to hold credentials — never indexed, even locally.
    nonisolated static func isSecretLooking(_ fileName: String) -> Bool {
        let lower = fileName.lowercased()
        return lower.hasPrefix(".env") || lower.hasSuffix(".pem") || lower.hasSuffix(".key")
            || lower.contains("secret") || lower.contains("credential") || lower.hasPrefix("id_rsa")
    }

    nonisolated static func isIndexable(_ url: URL) -> Bool {
        let ext = url.pathExtension.lowercased()
        return (plainTextExtensions.contains(ext) || richTextExtensions.contains(ext))
            && !isSecretLooking(url.lastPathComponent)
    }

    nonisolated static func scanFolder(at root: URL, known: [String: Date]) -> FolderScan {
        var scan = FolderScan()
        let keys: [URLResourceKey] = [.isDirectoryKey, .contentModificationDateKey, .fileSizeKey]
        guard let walker = FileManager.default.enumerator(
            at: root, includingPropertiesForKeys: keys,
            options: [.skipsHiddenFiles, .skipsPackageDescendants]) else { return scan }
        let rootPath = root.standardizedFileURL.path
        var count = 0
        for case let url as URL in walker {
            let values = try? url.resourceValues(forKeys: Set(keys))
            if values?.isDirectory == true {
                if skippedDirectories.contains(url.lastPathComponent) { walker.skipDescendants() }
                continue
            }
            guard isIndexable(url) else { continue }
            let ext = url.pathExtension.lowercased()
            let size = values?.fileSize ?? 0
            // Plain text over 2 MB is a data dump, not a document.
            if size > (plainTextExtensions.contains(ext) ? 2_000_000 : 40_000_000) { continue }
            count += 1
            if count > maxFilesPerFolder { scan.truncated = true; break }

            var relative = url.standardizedFileURL.path
            if relative.hasPrefix(rootPath) { relative.removeFirst(rootPath.count) }
            relative = relative.trimmingCharacters(in: CharacterSet(charactersIn: "/"))
            scan.present.insert(relative)

            let modified = values?.contentModificationDate ?? .distantPast
            if let indexed = known[relative], abs(indexed.timeIntervalSince(modified)) < 1 { continue }
            guard let text = extractText(from: url), !text.isEmpty else {
                // Unreadable (scanned PDF, binary): recorded with no chunks.
                scan.changed.append(ScannedFile(relativePath: relative, modifiedAt: modified,
                                                languageRaw: NLLanguage.undetermined.rawValue, pieces: []))
                continue
            }
            let language = NLLanguageRecognizer.dominantLanguage(for: String(text.prefix(4_000))) ?? .undetermined
            scan.changed.append(ScannedFile(relativePath: relative, modifiedAt: modified,
                                            languageRaw: language.rawValue,
                                            pieces: Array(chunkText(text).prefix(maxChunksPerFile))))
        }
        return scan
    }

    private nonisolated static func uniqueFolderName(_ name: String, taken: [String]) -> String {
        guard taken.contains(name) else { return name }
        var n = 2
        while taken.contains("\(name) \(n)") { n += 1 }
        return "\(name) \(n)"
    }

    // MARK: - Profile Scoping

    /// Tags every document in the KB into the given profile ID.
    func tagAllDocuments(into id: UUID) {
        for i in documents.indices { documents[i].profileIDs.insert(id) }
        save()
    }

    /// Replaces the full set of profile tags for a document.
    func setProfiles(_ ids: Set<UUID>, for document: KBDocument) {
        guard let i = documents.firstIndex(where: { $0.id == document.id }) else { return }
        documents[i].profileIDs = ids
        save()
    }

    /// Returns the names of documents tagged into the given profile ID, plus
    /// every folder document (folders serve all profiles).
    func documentNames(for profileID: UUID) -> [String] {
        documents.filter { $0.profileIDs.contains(profileID) || $0.folderID != nil }.map(\.name)
    }

    // MARK: - Retrieval

    /// Returns the best-matching chunks for the recent conversation, joined with
    /// each document's current note. Pass `profileID` to restrict hand-added
    /// documents to those tagged into that profile; `nil` searches all.
    func search(query: String, profileID: UUID? = nil, topK: Int = 4) async -> [KBReference] {
        guard !chunks.isEmpty, !query.isEmpty else { return [] }

        let allowedNames: Set<String>? = profileID.map { id in
            Set(documents.filter { $0.profileIDs.contains(id) || $0.folderID != nil }.map(\.name))
        }
        let snapshot = chunks
        let version = chunksVersion
        let cached = lexical?.version == version ? lexical?.index : nil
        let notesByDocument = Dictionary(
            documents.map { ($0.name, $0.note) },
            uniquingKeysWith: { first, _ in first }
        )

        let (best, built): ([KBChunk], LexicalIndex) = await Task.detached(priority: .userInitiated) {
            let index = cached ?? LexicalIndex(texts: snapshot.map { $0.documentName + " " + $0.text })
            let allowed: (Int) -> Bool = { i in allowedNames?.contains(snapshot[i].documentName) ?? true }

            // Keyword ranking.
            let keywordRanked = index.search(query, limit: 20).filter { allowed($0.index) }.map(\.index)

            // Embedding ranking — only chunks that have a vector (hand-added
            // documents). Query embedded once per language space.
            let languages = Set(snapshot.filter { !$0.embedding.isEmpty }.map(\.languageRaw))
            var queryVectors: [String: [Double]] = [:]
            for raw in languages {
                queryVectors[raw] = Self.embed(query, language: NLLanguage(rawValue: raw))
            }
            let semanticRanked: [Int] = snapshot.indices
                .compactMap { i -> (Int, Double)? in
                    let chunk = snapshot[i]
                    guard !chunk.embedding.isEmpty, allowed(i),
                          let vector = queryVectors[chunk.languageRaw] else { return nil }
                    let score = Self.cosineSimilarity(vector, chunk.embedding)
                    return score > 0.3 ? (i, score) : nil
                }
                .sorted { $0.1 > $1.1 }
                .prefix(20)
                .map(\.0)

            let fused = Self.reciprocalRankFusion([keywordRanked, semanticRanked])
            return (fused.prefix(topK).map { snapshot[$0] }, index)
        }.value

        if cached == nil, chunksVersion == version {
            lexical = (version, built)
        }

        return best.map { chunk in
            let note = notesByDocument[chunk.documentName]?.nilIfEmpty
            return KBReference(documentName: chunk.documentName, note: note, text: chunk.text, chunkID: chunk.id)
        }
    }

    /// Merges rankings by reciprocal rank: an item's score is Σ 1/(k + rank).
    /// Needs no score normalization between keyword and embedding scales.
    nonisolated static func reciprocalRankFusion(_ rankings: [[Int]], k: Double = 60) -> [Int] {
        var scores: [Int: Double] = [:]
        for ranking in rankings {
            for (rank, item) in ranking.enumerated() {
                scores[item, default: 0] += 1 / (k + Double(rank + 1))
            }
        }
        return scores.sorted { $0.value == $1.value ? $0.key < $1.key : $0.value > $1.value }.map(\.key)
    }

    // MARK: - Text Extraction & Chunking

    nonisolated static func extractText(from url: URL) -> String? {
        let ext = url.pathExtension.lowercased()
        switch ext {
        case "pdf":
            return PDFDocument(url: url)?.string
        case "docx", "doc", "rtf", "odt":
            let type: NSAttributedString.DocumentType = switch ext {
            case "docx": .officeOpenXML
            case "doc": .docFormat
            case "odt": .openDocument
            default: .rtf
            }
            return (try? NSAttributedString(url: url, options: [.documentType: type],
                                            documentAttributes: nil))?.string
        case "html", "htm":
            // Tag-stripped, not NSAttributedString: its HTML importer needs
            // the main thread, and folders index in the background.
            guard let html = try? String(contentsOf: url, encoding: .utf8) else { return nil }
            return stripHTML(html)
        default:
            if let utf8 = try? String(contentsOf: url, encoding: .utf8) {
                return utf8
            }
            return try? String(contentsOf: url, encoding: .isoLatin1)
        }
    }

    nonisolated static func stripHTML(_ html: String) -> String {
        html
            .replacingOccurrences(of: #"(?is)<(script|style)\b.*?</\1>"#, with: " ", options: .regularExpression)
            .replacingOccurrences(of: #"(?i)<br\s*/?>|</(p|div|li|h[1-6]|tr)>"#, with: "\n\n", options: .regularExpression)
            .replacingOccurrences(of: #"<[^>]+>"#, with: " ", options: .regularExpression)
            .replacingOccurrences(of: "&nbsp;", with: " ")
            .replacingOccurrences(of: "&amp;", with: "&")
            .replacingOccurrences(of: "&lt;", with: "<")
            .replacingOccurrences(of: "&gt;", with: ">")
            .replacingOccurrences(of: "&quot;", with: "\"")
            .replacingOccurrences(of: #"[ \t]+"#, with: " ", options: .regularExpression)
    }

    /// Splits text into ~900-character chunks along paragraph boundaries.
    /// A paragraph longer than a chunk (code, dense PDFs with no blank lines)
    /// is split by lines, then hard-wrapped, so no chunk balloons.
    nonisolated static func chunkText(_ text: String) -> [String] {
        let limit = 900
        let paragraphs = text
            .components(separatedBy: "\n\n")
            .map { $0.trimmingCharacters(in: .whitespacesAndNewlines) }
            .filter { !$0.isEmpty }
            .flatMap { paragraph -> [String] in
                guard paragraph.count > limit else { return [paragraph] }
                var pieces: [String] = []
                var current = ""
                for line in paragraph.components(separatedBy: "\n") {
                    var rest = Substring(line)
                    while rest.count > limit {
                        pieces.append(String(rest.prefix(limit)))
                        rest = rest.dropFirst(limit)
                    }
                    if current.count + 1 + rest.count > limit, !current.isEmpty {
                        pieces.append(current)
                        current = ""
                    }
                    current += current.isEmpty ? String(rest) : "\n" + rest
                }
                if !current.isEmpty { pieces.append(current) }
                return pieces
            }

        var result: [String] = []
        var current = ""
        for paragraph in paragraphs {
            if current.count + 2 + paragraph.count > limit, !current.isEmpty {
                result.append(current)
                current = ""
            }
            current += current.isEmpty ? paragraph : "\n\n" + paragraph
        }
        if !current.isEmpty {
            result.append(current)
        }
        return result.filter { $0.count >= 40 }
    }

    // MARK: - Embedding

    private nonisolated static func embed(_ text: String, language: NLLanguage) -> [Double]? {
        let embedding = NLEmbedding.sentenceEmbedding(for: language)
            ?? NLEmbedding.sentenceEmbedding(for: .english)
        return embedding?.vector(for: text)
    }

    private nonisolated static func cosineSimilarity(_ a: [Double], _ b: [Double]) -> Double {
        guard a.count == b.count, !a.isEmpty else { return 0 }
        var dot = 0.0, normA = 0.0, normB = 0.0
        for i in a.indices {
            dot += a[i] * b[i]
            normA += a[i] * a[i]
            normB += b[i] * b[i]
        }
        guard normA > 0, normB > 0 else { return 0 }
        return dot / ((normA * normB).squareRoot())
    }

    // MARK: - Persistence

    private struct Store: Codable {
        var documents: [KBDocument]
        var chunks: [KBChunk]
        var folders: [KBFolder]

        init(documents: [KBDocument], chunks: [KBChunk], folders: [KBFolder]) {
            self.documents = documents
            self.chunks = chunks
            self.folders = folders
        }

        init(from decoder: Decoder) throws {
            let c = try decoder.container(keyedBy: CodingKeys.self)
            documents = try c.decode([KBDocument].self, forKey: .documents)
            chunks = try c.decode([KBChunk].self, forKey: .chunks)
            // Stores written before folders existed have no key.
            folders = try c.decodeIfPresent([KBFolder].self, forKey: .folders) ?? []
        }
    }

    private nonisolated static var storeURL: URL {
        let dir = FileManager.default
            .urls(for: .applicationSupportDirectory, in: .userDomainMask)[0]
            .appendingPathComponent("Parrot/KnowledgeBase", isDirectory: true)
        try? FileManager.default.createDirectory(at: dir, withIntermediateDirectories: true)
        return dir.appendingPathComponent("index.json")
    }

    private func load() {
        guard let data = try? Data(contentsOf: Self.storeURL) else { return }
        do {
            let store = try JSONDecoder().decode(Store.self, from: data)
            documents = store.documents
            chunks = store.chunks
            folders = store.folders
            chunksVersion += 1
        } catch {
            // The index exists but doesn't decode. Starting with an empty KB is
            // fine; silently OVERWRITING the broken index on the next save is
            // not — move it aside so the data stays recoverable.
            let stamp = ISO8601DateFormatter().string(from: .now)
                .replacingOccurrences(of: ":", with: "-")
            let backup = Self.storeURL.deletingLastPathComponent()
                .appendingPathComponent("index-corrupt-\(stamp).json")
            try? FileManager.default.moveItem(at: Self.storeURL, to: backup)
            NSLog("Parrot: knowledge base index failed to decode (\(error.localizedDescription)) — moved aside to \(backup.lastPathComponent)")
        }
    }

    /// Encodes and writes off the main thread (a folder index is megabytes of
    /// JSON), one write at a time, newest state last.
    private func save() {
        guard persistent else { return }
        let store = Store(documents: documents, chunks: chunks, folders: folders)
        let previous = saveTask
        saveTask = Task.detached(priority: .utility) {
            await previous?.value
            guard let data = try? JSONEncoder().encode(store) else { return }
            try? data.write(to: Self.storeURL, options: .atomic)
        }
    }
}

// MARK: - Keyword index

/// BM25 over chunk texts. Tokens are case- and accent-folded ("prođe" matches
/// "prode"), so it's language-agnostic, and rare words — names, product terms,
/// numbers — carry the most weight, which is what a live call's questions hinge on.
struct LexicalIndex: Sendable {
    private let postings: [String: [(doc: Int, tf: Int)]]
    private let lengths: [Int]
    private let averageLength: Double

    init(texts: [String]) {
        var postings: [String: [(doc: Int, tf: Int)]] = [:]
        var lengths: [Int] = []
        lengths.reserveCapacity(texts.count)
        for (doc, text) in texts.enumerated() {
            let tokens = Self.tokens(text)
            lengths.append(tokens.count)
            var counts: [String: Int] = [:]
            for token in tokens { counts[token, default: 0] += 1 }
            for (token, tf) in counts { postings[token, default: []].append((doc, tf)) }
        }
        self.postings = postings
        self.lengths = lengths
        self.averageLength = lengths.isEmpty ? 1 : max(1, Double(lengths.reduce(0, +)) / Double(lengths.count))
    }

    /// Ranked chunk indices. A chunk qualifies when it matches two distinct
    /// query terms, or one rare one (a name, a product term) — a live
    /// transcript shares common words with everything, and those are noise.
    func search(_ query: String, limit: Int) -> [(index: Int, score: Double)] {
        let n = Double(lengths.count)
        guard n > 0 else { return [] }
        let k1 = 1.2, b = 0.75
        // "Rare" scales with the corpus: in 2% of chunks at most, and always
        // allowed in two, so a small knowledge base still answers "ASDLC?".
        let rareDF = max(2, n * 0.02)
        var scores: [Int: Double] = [:]
        var matched: [Int: Int] = [:]
        var strongMatch: Set<Int> = []
        for term in Set(Self.tokens(query)) {
            guard let list = postings[term] else { continue }
            let df = Double(list.count)
            // Present in most chunks of a real corpus: carries no signal.
            guard n < 4 || df / n < 0.5 else { continue }
            let idf = log(1 + (n - df + 0.5) / (df + 0.5))
            for (doc, tf) in list {
                let norm = Double(tf) * (k1 + 1) / (Double(tf) + k1 * (1 - b + b * Double(lengths[doc]) / averageLength))
                scores[doc, default: 0] += idf * norm
                matched[doc, default: 0] += 1
                if df <= rareDF { strongMatch.insert(doc) }
            }
        }
        guard let top = scores.values.max() else { return [] }
        return scores
            .filter { (matched[$0.key] ?? 0) >= 2 || strongMatch.contains($0.key) }
            .filter { $0.value >= top * 0.35 }
            .sorted { $0.value == $1.value ? $0.key < $1.key : $0.value > $1.value }
            .prefix(limit)
            .map { (index: $0.key, score: $0.value) }
    }

    static func tokens(_ text: String) -> [String] {
        // đ has no decomposition, so diacritic folding leaves it alone.
        text.replacingOccurrences(of: "đ", with: "d").replacingOccurrences(of: "Đ", with: "D")
            .folding(options: [.caseInsensitive, .diacriticInsensitive], locale: nil)
            .components(separatedBy: CharacterSet.alphanumerics.inverted)
            .filter { $0.count >= 2 && !stopWords.contains($0) }
    }

    /// Function words in English and Croatian (accent-folded).
    static let stopWords: Set<String> = [
        "the", "and", "for", "you", "your", "are", "was", "were", "with", "that", "this", "these",
        "those", "from", "have", "has", "had", "not", "but", "can", "will", "would", "could",
        "should", "what", "when", "where", "who", "why", "how", "which", "about", "into", "our",
        "their", "they", "them", "there", "then", "than", "its", "it's", "is", "be", "been", "to",
        "of", "in", "on", "at", "by", "or", "an", "as", "if", "so", "we", "us", "do", "does",
        "did", "my", "me", "he", "she", "his", "her", "all", "any", "some", "just", "also", "very",
        "je", "su", "sam", "smo", "ste", "si", "biti", "bio", "bila", "bilo", "bi", "ce", "cu",
        "na", "za", "od", "do", "iz", "po", "sa", "se", "da", "ne", "ni", "li", "ili", "ali",
        "pa", "te", "jer", "kao", "sto", "sta", "koji", "koja", "koje", "kako", "kada", "kad",
        "gdje", "tko", "ovo", "ono", "taj", "ta", "to", "ja", "ti", "on", "ona", "mi", "vi", "oni",
        "vec", "jos", "samo", "ima", "nema", "nije", "nisu", "moze", "mogu", "treba", "sve", "svi",
    ]
}
