import Foundation

/// One chunk of a knowledge base document. Hand-added documents carry a
/// sentence embedding; documents indexed from a folder leave it empty and are
/// found by keyword ranking only (fast to index, works in any language).
struct KBChunk: Codable, Identifiable {
    var id = UUID()
    var documentName: String
    var languageRaw: String
    var text: String
    var embedding: [Double]
}

/// A document the user added to the knowledge base.
struct KBDocument: Codable, Identifiable {
    var id = UUID()
    var name: String
    /// User guidance for when to use this document, e.g. "use for pricing questions".
    var note: String = ""
    var chunkCount: Int
    var addedAt: Date
    /// Profiles this document is tagged into. Empty = unscoped (all-profiles).
    var profileIDs: Set<UUID> = []
    /// Set for documents that come from a knowledge folder. Folder documents
    /// serve every profile — the folder is the user's scoping.
    var folderID: UUID?
    /// Path inside the folder, and the file's modification date when indexed —
    /// how a rescan tells new, changed and deleted files apart.
    var relativePath: String?
    var modifiedAt: Date?

    enum CodingKeys: String, CodingKey {
        case id, name, note, chunkCount, addedAt, profileIDs, folderID, relativePath, modifiedAt
    }

    init(id: UUID = UUID(), name: String, note: String = "", chunkCount: Int, addedAt: Date,
         profileIDs: Set<UUID> = [], folderID: UUID? = nil, relativePath: String? = nil, modifiedAt: Date? = nil) {
        self.id = id
        self.name = name
        self.note = note
        self.chunkCount = chunkCount
        self.addedAt = addedAt
        self.profileIDs = profileIDs
        self.folderID = folderID
        self.relativePath = relativePath
        self.modifiedAt = modifiedAt
    }

    init(from decoder: Decoder) throws {
        let c = try decoder.container(keyedBy: CodingKeys.self)
        id = try c.decode(UUID.self, forKey: .id)
        name = try c.decode(String.self, forKey: .name)
        // Fields added after 1.0 decode leniently (decodeIfPresent + default):
        // a strict decode of a missing key fails the WHOLE store load, and the
        // next save would then overwrite the store with empty — total KB loss.
        note = try c.decodeIfPresent(String.self, forKey: .note) ?? ""
        chunkCount = try c.decode(Int.self, forKey: .chunkCount)
        addedAt = try c.decode(Date.self, forKey: .addedAt)
        profileIDs = try c.decodeIfPresent(Set<UUID>.self, forKey: .profileIDs) ?? []
        folderID = try c.decodeIfPresent(UUID.self, forKey: .folderID)
        relativePath = try c.decodeIfPresent(String.self, forKey: .relativePath)
        modifiedAt = try c.decodeIfPresent(Date.self, forKey: .modifiedAt)
    }
}

/// A folder the user pointed the knowledge base at. Its files are indexed and
/// re-checked before each call; the bookmark keeps the sandboxed app's access
/// to it across launches.
struct KBFolder: Codable, Identifiable, Equatable {
    var id = UUID()
    var name: String
    var bookmark: Data
    var lastIndexedAt: Date?
    var fileCount = 0
}

/// A retrieved chunk handed to the analysis provider, joined with its
/// document's current note.
struct KBReference {
    let documentName: String
    let note: String?
    let text: String
    /// The chunk's identity, so a live session can send each excerpt once.
    var chunkID: UUID? = nil
}
