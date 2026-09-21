import Foundation

enum OptimizerDaemonOperation: String, Codable, Sendable {
    case plan
    case execute
    case resume
    case cancel
    case pause
    case `continue`
    case status
}

struct OptimizerDaemonRequest: Codable, Sendable {
    let schemaVersion: Int
    let requestID: UUID
    let operation: OptimizerDaemonOperation
    let payload: Data?

    init(operation: OptimizerDaemonOperation, payload: Data? = nil, requestID: UUID = UUID()) {
        schemaVersion = 1
        self.requestID = requestID
        self.operation = operation
        self.payload = payload
    }

    enum CodingKeys: String, CodingKey {
        case schemaVersion = "schema_version"
        case requestID = "request_id"
        case operation, payload
    }
}

struct OptimizerDaemonResponse: Codable, Sendable {
    let schemaVersion: Int
    let requestID: UUID
    let succeeded: Bool
    let payload: Data?
    let errorCode: String?

    enum CodingKeys: String, CodingKey {
        case schemaVersion = "schema_version"
        case requestID = "request_id"
        case succeeded, payload
        case errorCode = "error_code"
    }
}

enum OptimizerDaemonTransportError: LocalizedError {
    case nonLoopbackEndpoint
    case requestTooLarge
    case responseTooLarge
    case invalidHTTPStatus
    case invalidResponse

    var errorDescription: String? {
        switch self {
        case .nonLoopbackEndpoint: "Optimizer daemon endpoint must be an HTTP loopback address."
        case .requestTooLarge: "Optimizer daemon request exceeded the 1 MiB safety limit."
        case .responseTooLarge: "Optimizer daemon response exceeded the 1 MiB safety limit."
        case .invalidHTTPStatus: "Optimizer daemon returned an invalid HTTP status."
        case .invalidResponse: "Optimizer daemon returned an invalid response contract."
        }
    }
}

/// A sandbox-safe transport. This type never discovers, launches, or signals a process.
/// The daemon lifecycle belongs to an independently managed, non-sandboxed component.
struct OptimizerDaemonTransport: Sendable {
    static let maximumMessageBytes = 1_048_576

    let endpoint: URL
    private let session: URLSession

    init(endpoint: URL, session: URLSession? = nil) throws {
        guard endpoint.scheme == "http",
              endpoint.user == nil,
              endpoint.password == nil,
              endpoint.query == nil,
              endpoint.fragment == nil,
              let host = endpoint.host?.lowercased(),
              ["127.0.0.1", "::1", "localhost"].contains(host) else {
            throw OptimizerDaemonTransportError.nonLoopbackEndpoint
        }
        self.endpoint = endpoint
        if let session {
            self.session = session
        } else {
            let configuration = URLSessionConfiguration.ephemeral
            configuration.timeoutIntervalForRequest = 120
            configuration.timeoutIntervalForResource = 120
            configuration.urlCache = nil
            configuration.requestCachePolicy = .reloadIgnoringLocalCacheData
            configuration.httpCookieStorage = nil
            configuration.httpShouldSetCookies = false
            configuration.urlCredentialStorage = nil
            configuration.httpMaximumConnectionsPerHost = 1
            self.session = URLSession(configuration: configuration)
        }
    }

    func send(_ request: OptimizerDaemonRequest) async throws -> OptimizerDaemonResponse {
        let body = try JSONEncoder().encode(request)
        guard body.count <= Self.maximumMessageBytes else {
            throw OptimizerDaemonTransportError.requestTooLarge
        }
        var urlRequest = URLRequest(url: endpoint)
        urlRequest.httpMethod = "POST"
        urlRequest.httpBody = body
        urlRequest.setValue("application/json", forHTTPHeaderField: "Content-Type")
        urlRequest.setValue("no-store", forHTTPHeaderField: "Cache-Control")
        let (data, response) = try await session.data(for: urlRequest)
        guard data.count <= Self.maximumMessageBytes else {
            throw OptimizerDaemonTransportError.responseTooLarge
        }
        guard let http = response as? HTTPURLResponse, http.statusCode == 200 else {
            throw OptimizerDaemonTransportError.invalidHTTPStatus
        }
        let decoded = try JSONDecoder().decode(OptimizerDaemonResponse.self, from: data)
        guard decoded.schemaVersion == 1,
              decoded.requestID == request.requestID,
              decoded.errorCode?.utf8.count ?? 0 <= 128,
              decoded.payload?.count ?? 0 <= Self.maximumMessageBytes else {
            throw OptimizerDaemonTransportError.invalidResponse
        }
        return decoded
    }
}
