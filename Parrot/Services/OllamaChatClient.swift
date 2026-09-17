import Foundation

/// Streaming client for Ollama's native /api/chat — the live agent's transport.
///
/// Native rather than the OpenAI-compatible endpoint because only /api/chat
/// takes `think`, `keep_alive` and `options` (num_ctx, num_batch) per request,
/// and streams the prompt-eval counts the agent uses to size its session.
/// Measured on qwen3.5:9b, M4 Pro, 2026-09-17: the compat endpoint with no
/// think flag spent its whole 400-token budget reasoning and returned no text
/// after 29 s; native `think: false` answered with its first token in 0.18 s.
struct OllamaChatClient: Sendable {
    var baseURL = URL(string: "http://localhost:11434")!

    struct Message: Codable, Equatable, Sendable {
        let role: String
        let content: String

        static func system(_ content: String) -> Message { Message(role: "system", content: content) }
        static func user(_ content: String) -> Message { Message(role: "user", content: content) }
        static func assistant(_ content: String) -> Message { Message(role: "assistant", content: content) }
    }

    /// Request options that must stay IDENTICAL across a session: Ollama
    /// reloads the model when num_ctx or num_batch change between requests
    /// (a 5–8 s stall), and a reload also drops the cached session.
    struct Options: Equatable, Sendable {
        var numCtx = 32_768
        /// 1024 prefilled a fresh 1.2k-token prompt in 4.0 s vs 5.5 s at the
        /// default 512 (qwen3.5:9b, M4 Pro, 2026-09-17).
        var numBatch = 1_024
        var temperature = 0.3
        var keepAlive = "30m"
    }

    struct Metrics: Equatable, Sendable {
        /// Tokens in the prompt as Ollama counts them (cached ones included).
        var promptTokens = 0
        var promptEvalSeconds = 0.0
        var outputTokens = 0
        var evalSeconds = 0.0
        var loadSeconds = 0.0
        /// Wall clock from sending the request to the first content token.
        var firstTokenSeconds: Double?
        var totalSeconds = 0.0
        var doneReason: String?
    }

    enum ClientError: LocalizedError, Equatable {
        case notRunning
        case modelMissing(String)
        case server(String)

        var errorDescription: String? {
            switch self {
            case .notRunning: "Ollama isn't running on this Mac — start the Ollama app or run `ollama serve`."
            case .modelMissing(let model): "Model \(model) isn't installed — run `ollama pull \(model)`."
            case .server(let message): "Ollama: \(message)"
            }
        }
    }

    /// Streams one chat turn. `onText` receives each content delta on the main
    /// actor as it arrives. Cancelling the calling task closes the connection,
    /// which stops generation in Ollama; tokens already produced stay in its
    /// cache, so a caller that keeps them in the session loses nothing.
    func stream(model: String, messages: [Message], options: Options, maxTokens: Int,
                onText: @escaping @MainActor (String) -> Void) async throws -> (text: String, metrics: Metrics) {
        var sendThink = true
        while true {
            do {
                return try await streamOnce(model: model, messages: messages, options: options,
                                            maxTokens: maxTokens, think: sendThink, onText: onText)
            } catch ClientError.server(let message)
                where sendThink && message.localizedCaseInsensitiveContains("think") {
                // Older Ollama builds and some models reject the key — one retry without it.
                sendThink = false
            }
        }
    }

    private func streamOnce(model: String, messages: [Message], options: Options, maxTokens: Int,
                            think: Bool, onText: @escaping @MainActor (String) -> Void) async throws -> (text: String, metrics: Metrics) {
        var body: [String: Any] = [
            "model": model,
            "stream": true,
            "keep_alive": options.keepAlive,
            "messages": messages.map { ["role": $0.role, "content": $0.content] },
            "options": [
                "num_ctx": options.numCtx,
                "num_batch": options.numBatch,
                "num_predict": maxTokens,
                "temperature": options.temperature,
            ],
        ]
        // The live path must never reason before speaking: thinking models spend
        // the token budget on hidden chain-of-thought and the user sees nothing.
        if think { body["think"] = false }

        var request = URLRequest(url: baseURL.appendingPathComponent("api/chat"))
        request.httpMethod = "POST"
        // Generous: a first request after a cold start loads ~7 GB before the
        // first token. Liveness is the agent's per-turn deadline, not this.
        request.timeoutInterval = 300
        request.setValue("application/json", forHTTPHeaderField: "Content-Type")
        request.httpBody = try JSONSerialization.data(withJSONObject: body)

        let started = Date()
        let bytes: URLSession.AsyncBytes
        let response: URLResponse
        do {
            (bytes, response) = try await URLSession.shared.bytes(for: request)
        } catch let error as URLError where Self.isConnectionRefusal(error) {
            throw ClientError.notRunning
        }
        guard let http = response as? HTTPURLResponse else {
            throw ClientError.server("no HTTP response")
        }
        if http.statusCode != 200 {
            var data = Data()
            for try await byte in bytes {
                data.append(byte)
                if data.count > 8_192 { break }
            }
            let message = Self.errorMessage(from: data) ?? "HTTP \(http.statusCode)"
            if http.statusCode == 404, message.localizedCaseInsensitiveContains("not found") {
                throw ClientError.modelMissing(model)
            }
            throw ClientError.server(message)
        }

        var text = ""
        var metrics = Metrics()
        for try await line in bytes.lines {
            guard let data = line.data(using: .utf8),
                  let chunk = try? JSONDecoder().decode(StreamChunk.self, from: data) else { continue }
            if let error = chunk.error { throw ClientError.server(error) }
            if let delta = chunk.message?.content, !delta.isEmpty {
                if metrics.firstTokenSeconds == nil {
                    metrics.firstTokenSeconds = Date().timeIntervalSince(started)
                }
                text += delta
                await onText(delta)
            }
            if chunk.done == true {
                metrics.promptTokens = chunk.promptEvalCount ?? 0
                metrics.promptEvalSeconds = Double(chunk.promptEvalDuration ?? 0) / 1e9
                metrics.outputTokens = chunk.evalCount ?? 0
                metrics.evalSeconds = Double(chunk.evalDuration ?? 0) / 1e9
                metrics.loadSeconds = Double(chunk.loadDuration ?? 0) / 1e9
                metrics.doneReason = chunk.doneReason
            }
        }
        try Task.checkCancellation()
        metrics.totalSeconds = Date().timeIntervalSince(started)
        return (text, metrics)
    }

    /// Installed model names ("qwen3.5:9b"). Throws `.notRunning` when the
    /// server isn't up; a short timeout keeps the check off the call's path.
    func installedModels(timeout: TimeInterval = 2) async throws -> [String] {
        var request = URLRequest(url: baseURL.appendingPathComponent("api/tags"))
        request.timeoutInterval = timeout
        let data: Data
        do {
            (data, _) = try await URLSession.shared.data(for: request)
        } catch let error as URLError where Self.isConnectionRefusal(error) || error.code == .timedOut {
            throw ClientError.notRunning
        }
        struct Tags: Decodable {
            struct Model: Decodable { let name: String }
            let models: [Model]
        }
        return (try? JSONDecoder().decode(Tags.self, from: data))?.models.map(\.name) ?? []
    }

    /// "qwen3.5:9b" is installed under that exact name; a bare "llama3.2"
    /// means ":latest".
    static func isInstalled(_ model: String, in names: [String]) -> Bool {
        let wanted = model.contains(":") ? model : model + ":latest"
        return names.contains(wanted) || names.contains(model)
    }

    private struct StreamChunk: Decodable {
        struct Message: Decodable { let content: String? }
        let message: Message?
        let done: Bool?
        let doneReason: String?
        let promptEvalCount: Int?
        let promptEvalDuration: Int?
        let evalCount: Int?
        let evalDuration: Int?
        let loadDuration: Int?
        let error: String?

        enum CodingKeys: String, CodingKey {
            case message, done, error
            case doneReason = "done_reason"
            case promptEvalCount = "prompt_eval_count"
            case promptEvalDuration = "prompt_eval_duration"
            case evalCount = "eval_count"
            case evalDuration = "eval_duration"
            case loadDuration = "load_duration"
        }
    }

    private static func isConnectionRefusal(_ error: URLError) -> Bool {
        [.cannotConnectToHost, .cannotFindHost, .networkConnectionLost, .notConnectedToInternet]
            .contains(error.code)
    }

    private static func errorMessage(from data: Data) -> String? {
        guard let object = try? JSONSerialization.jsonObject(with: data) as? [String: Any] else {
            return String(data: data, encoding: .utf8)?.nilIfEmpty
        }
        return object["error"] as? String
    }
}
