import Foundation

public protocol VLLMAppleRuntimeClient: Sendable {
    func hardware() async throws -> HardwareInfo
    func runtimeProfile() async throws -> RuntimeProfile
    func health() async throws -> HealthStatus
    func chat(_ request: ChatRequest) async throws -> ChatResponse
    func streamChat(_ request: ChatRequest) -> AsyncThrowingStream<ChatEvent, Error>
    func runtimeEvents(afterEventID: String?) -> AsyncThrowingStream<RuntimeEvent, Error>
}

public enum RuntimeClientError: Error, Sendable, Equatable {
    case invalidResponse
    case incompatibleSchema(received: Int, supported: Int)
    case server(status: Int, code: String?, message: String)

    public var messageKey: String {
        switch self {
        case .invalidResponse: "runtime.error.invalid_response"
        case .incompatibleSchema: "runtime.error.incompatible_schema"
        case .server: "runtime.error.server"
        }
    }
}

private struct HardwareEnvelope: Decodable {
    let schemaVersion: Int
    let hardware: HardwareInfo
}

private struct RuntimeEnvelope: Decodable {
    let schemaVersion: Int
    let profile: RuntimeProfile
}

private struct ErrorEnvelope: Decodable {
    struct Body: Decodable {
        let message: String
        let code: String?
    }
    let error: Body
}

public final class HTTPRuntimeClient: VLLMAppleRuntimeClient, @unchecked Sendable {
    public static let supportedSchemaVersion = 1

    private let baseURL: URL
    private let session: URLSession
    private let decoder: JSONDecoder
    private let encoder: JSONEncoder
    private let sessionToken: String?

    public init(baseURL: URL, session: URLSession = .shared, sessionToken: String? = nil) {
        self.baseURL = baseURL
        self.session = session
        self.decoder = JSONDecoder()
        self.encoder = JSONEncoder()
        self.sessionToken = sessionToken
        self.decoder.keyDecodingStrategy = .convertFromSnakeCase
        self.encoder.keyEncodingStrategy = .convertToSnakeCase
    }

    public func health() async throws -> HealthStatus {
        try await get("health", as: HealthStatus.self)
    }

    public func hardware() async throws -> HardwareInfo {
        let envelope = try await get("v1/hardware", as: HardwareEnvelope.self)
        try validate(schemaVersion: envelope.schemaVersion)
        return envelope.hardware
    }

    public func runtimeProfile() async throws -> RuntimeProfile {
        let envelope = try await get("v1/runtime", as: RuntimeEnvelope.self)
        try validate(schemaVersion: envelope.schemaVersion)
        return envelope.profile
    }

    public func chat(_ request: ChatRequest) async throws -> ChatResponse {
        var request = request
        request.stream = false
        let body = try encoder.encode(request)
        return try await send("v1/chat/completions", method: "POST", body: body, as: ChatResponse.self)
    }

    public func streamChat(_ request: ChatRequest) -> AsyncThrowingStream<ChatEvent, Error> {
        AsyncThrowingStream { continuation in
            let task = Task {
                do {
                    var streamingRequest = request
                    streamingRequest.stream = true
                    var urlRequest = authorizedRequest(url: url(for: "v1/chat/completions"))
                    urlRequest.httpMethod = "POST"
                    urlRequest.httpBody = try encoder.encode(streamingRequest)
                    urlRequest.setValue("application/json", forHTTPHeaderField: "Content-Type")
                    let (bytes, response) = try await session.bytes(for: urlRequest)
                    try validate(response: response)
                    for try await line in bytes.lines {
                        try Task.checkCancellation()
                        guard line.hasPrefix("data:") else { continue }
                        let value = line.dropFirst(5).trimmingCharacters(in: .whitespaces)
                        if value == "[DONE]" {
                            continuation.yield(.completed)
                            continuation.finish()
                            return
                        }
                        continuation.yield(.data(value))
                    }
                    continuation.yield(.completed)
                    continuation.finish()
                } catch {
                    continuation.finish(throwing: error)
                }
            }
            continuation.onTermination = { _ in task.cancel() }
        }
    }

    public func runtimeEvents(
        afterEventID: String? = nil
    ) -> AsyncThrowingStream<RuntimeEvent, Error> {
        AsyncThrowingStream { continuation in
            let task = Task {
                do {
                    var request = authorizedRequest(url: url(for: "v1/events"))
                    request.timeoutInterval = 24 * 60 * 60
                    if let afterEventID {
                        request.setValue(afterEventID, forHTTPHeaderField: "Last-Event-ID")
                    }
                    let (bytes, response) = try await session.bytes(for: request)
                    try validate(response: response)
                    for try await line in bytes.lines {
                        try Task.checkCancellation()
                        guard line.hasPrefix("data:") else { continue }
                        let value = line.dropFirst(5).trimmingCharacters(in: .whitespaces)
                        guard let data = value.data(using: .utf8) else { continue }
                        let event = try decoder.decode(RuntimeEvent.self, from: data)
                        try validate(schemaVersion: event.schemaVersion)
                        continuation.yield(event)
                    }
                    continuation.finish()
                } catch {
                    continuation.finish(throwing: error)
                }
            }
            continuation.onTermination = { _ in task.cancel() }
        }
    }

    private func get<T: Decodable>(_ path: String, as type: T.Type) async throws -> T {
        try await send(path, method: "GET", body: nil, as: type)
    }

    private func send<T: Decodable>(
        _ path: String,
        method: String,
        body: Data?,
        as type: T.Type
    ) async throws -> T {
        var request = authorizedRequest(url: url(for: path))
        request.httpMethod = method
        request.httpBody = body
        request.timeoutInterval = 30
        if body != nil {
            request.setValue("application/json", forHTTPHeaderField: "Content-Type")
        }
        let (data, response) = try await session.data(for: request)
        try validate(response: response, data: data)
        do {
            return try decoder.decode(type, from: data)
        } catch {
            throw RuntimeClientError.invalidResponse
        }
    }

    private func url(for path: String) -> URL {
        baseURL.appending(path: path)
    }

    private func authorizedRequest(url: URL) -> URLRequest {
        var request = URLRequest(url: url)
        if let sessionToken {
            request.setValue("Bearer \(sessionToken)", forHTTPHeaderField: "Authorization")
        }
        return request
    }

    private func validate(schemaVersion: Int) throws {
        guard schemaVersion <= Self.supportedSchemaVersion else {
            throw RuntimeClientError.incompatibleSchema(
                received: schemaVersion,
                supported: Self.supportedSchemaVersion
            )
        }
    }

    private func validate(response: URLResponse, data: Data = Data()) throws {
        guard let http = response as? HTTPURLResponse else {
            throw RuntimeClientError.invalidResponse
        }
        guard (200..<300).contains(http.statusCode) else {
            if let envelope = try? decoder.decode(ErrorEnvelope.self, from: data) {
                throw RuntimeClientError.server(
                    status: http.statusCode,
                    code: envelope.error.code,
                    message: envelope.error.message
                )
            }
            throw RuntimeClientError.server(
                status: http.statusCode,
                code: nil,
                message: HTTPURLResponse.localizedString(forStatusCode: http.statusCode)
            )
        }
    }
}
