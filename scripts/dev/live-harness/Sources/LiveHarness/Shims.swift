import Foundation

// Stand-ins for app types that live in SwiftUI/SwiftData files the harness
// can't compile without Xcode.

enum APIKeyStore {
    static func load(account: String = "claude-api-key") -> String? { nil }
}

enum RecordingError: LocalizedError {
    case modelNotReady
    var errorDescription: String? { "model not ready" }
}

extension String {
    var nilIfEmpty: String? { isEmpty ? nil : self }
}
