import Darwin
import Foundation

private let maximumHTTPResponseBytes = 16 * 1024 * 1024
private let maximumHTTPHeaderBytes = 64 * 1024

private final class LockedSocket: @unchecked Sendable {
    private let lock = NSLock()
    private var descriptor: Int32 = -1
    private var closed = false

    func store(_ value: Int32) {
        lock.lock()
        if closed {
            lock.unlock()
            Darwin.close(value)
            return
        }
        descriptor = value
        lock.unlock()
    }

    func close() {
        lock.lock()
        let value = descriptor
        descriptor = -1
        closed = true
        lock.unlock()
        if value >= 0 {
            Darwin.shutdown(value, SHUT_RDWR)
            Darwin.close(value)
        }
    }
}

private struct UnixHTTPResponse: Sendable {
    let status: Int
    let headers: [String: String]
    let body: Data
}

private struct UnixHardwareEnvelope: Decodable {
    let schemaVersion: Int
    let hardware: HardwareInfo
}

private struct UnixRuntimeEnvelope: Decodable {
    let schemaVersion: Int
    let profile: RuntimeProfile
}

public final class UnixSocketRuntimeClient: VLLMAppleRuntimeClient, @unchecked Sendable {
    public let socketPath: String
    private let sessionToken: String?

    public init(socketPath: String, sessionToken: String? = nil) {
        self.socketPath = socketPath
        self.sessionToken = sessionToken
    }

    public func health() async throws -> HealthStatus {
        try await request("/health", as: HealthStatus.self)
    }

    public func hardware() async throws -> HardwareInfo {
        let envelope = try await request("/v1/hardware", as: UnixHardwareEnvelope.self)
        try validate(schemaVersion: envelope.schemaVersion)
        return envelope.hardware
    }

    public func runtimeProfile() async throws -> RuntimeProfile {
        let envelope = try await request("/v1/runtime", as: UnixRuntimeEnvelope.self)
        try validate(schemaVersion: envelope.schemaVersion)
        return envelope.profile
    }

    public func chat(_ request: ChatRequest) async throws -> ChatResponse {
        var request = request
        request.stream = false
        let body = try makeEncoder().encode(request)
        return try await self.request(
            "/v1/chat/completions", method: "POST", body: body, as: ChatResponse.self
        )
    }

    public func streamChat(_ request: ChatRequest) -> AsyncThrowingStream<ChatEvent, Error> {
        var request = request
        request.stream = true
        let body: Data
        do {
            body = try makeEncoder().encode(request)
        } catch {
            return AsyncThrowingStream { $0.finish(throwing: error) }
        }
        return lineStream(path: "/v1/chat/completions", method: "POST", body: body) { line in
            guard line.hasPrefix("data:") else { return nil }
            let value = line.dropFirst(5).trimmingCharacters(in: .whitespaces)
            return value == "[DONE]" ? .completed : .data(value)
        }
    }

    public func runtimeEvents(
        afterEventID: String? = nil
    ) -> AsyncThrowingStream<RuntimeEvent, Error> {
        var headers: [String: String] = [:]
        if let afterEventID { headers["Last-Event-ID"] = afterEventID }
        return lineStream(path: "/v1/events", headers: headers) { line in
            guard line.hasPrefix("data:") else { return nil }
            let value = line.dropFirst(5).trimmingCharacters(in: .whitespaces)
            guard let data = value.data(using: .utf8) else { return nil }
            let event = try makeDecoder().decode(RuntimeEvent.self, from: data)
            try validate(schemaVersion: event.schemaVersion)
            return event
        }
    }

    private func request<T: Decodable>(
        _ path: String,
        method: String = "GET",
        body: Data? = nil,
        as type: T.Type
    ) async throws -> T {
        let response = try await Task.detached {
            try performUnixRequest(
                socketPath: self.socketPath,
                sessionToken: self.sessionToken,
                path: path,
                method: method,
                body: body
            )
        }.value
        try validate(response: response)
        do {
            return try makeDecoder().decode(type, from: response.body)
        } catch let error as RuntimeClientError {
            throw error
        } catch {
            throw RuntimeClientError.invalidResponse
        }
    }

    private func lineStream<T: Sendable>(
        path: String,
        method: String = "GET",
        body: Data? = nil,
        headers: [String: String] = [:],
        transform: @escaping @Sendable (String) throws -> T?
    ) -> AsyncThrowingStream<T, Error> {
        AsyncThrowingStream { continuation in
            let lockedSocket = LockedSocket()
            let task = Task.detached {
                do {
                    try streamUnixLines(
                        socketPath: self.socketPath,
                        sessionToken: self.sessionToken,
                        path: path,
                        method: method,
                        body: body,
                        headers: headers,
                        lockedSocket: lockedSocket
                    ) { line in
                        try Task.checkCancellation()
                        if let item = try transform(line) {
                            continuation.yield(item)
                        }
                    }
                    continuation.finish()
                } catch {
                    continuation.finish(throwing: error)
                }
                lockedSocket.close()
            }
            continuation.onTermination = { _ in
                task.cancel()
                lockedSocket.close()
            }
        }
    }
}

private func makeDecoder() -> JSONDecoder {
    let decoder = JSONDecoder()
    decoder.keyDecodingStrategy = .convertFromSnakeCase
    return decoder
}

private func makeEncoder() -> JSONEncoder {
    let encoder = JSONEncoder()
    encoder.keyEncodingStrategy = .convertToSnakeCase
    return encoder
}

private func validate(schemaVersion: Int) throws {
    guard schemaVersion <= HTTPRuntimeClient.supportedSchemaVersion else {
        throw RuntimeClientError.incompatibleSchema(
            received: schemaVersion,
            supported: HTTPRuntimeClient.supportedSchemaVersion
        )
    }
}

private func validate(response: UnixHTTPResponse) throws {
    guard (200..<300).contains(response.status) else {
        let message: String
        let code: String?
        if
            let object = try? JSONSerialization.jsonObject(with: response.body) as? [String: Any],
            let error = object["error"] as? [String: Any]
        {
            message = error["message"] as? String ?? "Server error"
            code = error["code"] as? String
        } else {
            message = "HTTP status \(response.status)"
            code = nil
        }
        throw RuntimeClientError.server(status: response.status, code: code, message: message)
    }
}

private func connectUnixSocket(path: String) throws -> Int32 {
    let pathBytes = Array(path.utf8)
    var address = sockaddr_un()
    let capacity = MemoryLayout.size(ofValue: address.sun_path)
    guard !pathBytes.isEmpty, pathBytes.count < capacity else {
        throw RuntimeClientError.invalidResponse
    }
    address.sun_family = sa_family_t(AF_UNIX)
    address.sun_len = UInt8(MemoryLayout<sockaddr_un>.size)
    withUnsafeMutableBytes(of: &address.sun_path) { destination in
        destination.copyBytes(from: pathBytes)
        destination[pathBytes.count] = 0
    }

    let descriptor = Darwin.socket(AF_UNIX, SOCK_STREAM, 0)
    guard descriptor >= 0 else { throw POSIXError(.ENOTSOCK) }
    var noSignal: Int32 = 1
    Darwin.setsockopt(
        descriptor,
        SOL_SOCKET,
        SO_NOSIGPIPE,
        &noSignal,
        socklen_t(MemoryLayout<Int32>.size)
    )
    let result = withUnsafePointer(to: &address) { pointer in
        pointer.withMemoryRebound(to: sockaddr.self, capacity: 1) {
            Darwin.connect(descriptor, $0, socklen_t(MemoryLayout<sockaddr_un>.size))
        }
    }
    guard result == 0 else {
        let error = POSIXErrorCode(rawValue: errno) ?? .ECONNREFUSED
        Darwin.close(descriptor)
        throw POSIXError(error)
    }
    return descriptor
}

private func sendAll(_ data: Data, descriptor: Int32) throws {
    try data.withUnsafeBytes { rawBuffer in
        guard let baseAddress = rawBuffer.baseAddress else { return }
        var sent = 0
        while sent < data.count {
            let count = Darwin.send(descriptor, baseAddress.advanced(by: sent), data.count - sent, 0)
            if count < 0 {
                if errno == EINTR { continue }
                throw POSIXError(POSIXErrorCode(rawValue: errno) ?? .EIO)
            }
            sent += count
        }
    }
}

private func requestData(
    path: String,
    method: String,
    body: Data?,
    sessionToken: String?,
    headers: [String: String] = [:]
) -> Data {
    var lines = [
        "\(method) \(path) HTTP/1.1",
        "Host: localhost",
        "Connection: close",
        "Accept: application/json, text/event-stream",
    ]
    if let sessionToken { lines.append("Authorization: Bearer \(sessionToken)") }
    for (name, value) in headers { lines.append("\(name): \(value)") }
    if let body {
        lines.append("Content-Type: application/json")
        lines.append("Content-Length: \(body.count)")
    }
    var data = Data((lines.joined(separator: "\r\n") + "\r\n\r\n").utf8)
    if let body { data.append(body) }
    return data
}

private func receiveChunk(descriptor: Int32) throws -> Data? {
    var bytes = [UInt8](repeating: 0, count: 4096)
    while true {
        let count = Darwin.recv(descriptor, &bytes, bytes.count, 0)
        if count > 0 { return Data(bytes[0..<count]) }
        if count == 0 { return nil }
        if errno == EINTR { continue }
        throw POSIXError(POSIXErrorCode(rawValue: errno) ?? .EIO)
    }
}

private func headerBoundary(in data: Data) -> Range<Data.Index>? {
    data.range(of: Data("\r\n\r\n".utf8))
}

private func parseHeader(_ data: Data) throws -> (Int, [String: String]) {
    guard let text = String(data: data, encoding: .utf8) else {
        throw RuntimeClientError.invalidResponse
    }
    let lines = text.components(separatedBy: "\r\n")
    let statusParts = lines.first?.split(separator: " ", maxSplits: 2) ?? []
    guard statusParts.count >= 2, let status = Int(statusParts[1]) else {
        throw RuntimeClientError.invalidResponse
    }
    var headers: [String: String] = [:]
    for line in lines.dropFirst() {
        guard let separator = line.firstIndex(of: ":") else { continue }
        let name = line[..<separator].lowercased()
        let value = line[line.index(after: separator)...].trimmingCharacters(in: .whitespaces)
        headers[name] = value
    }
    return (status, headers)
}

private func performUnixRequest(
    socketPath: String,
    sessionToken: String?,
    path: String,
    method: String,
    body: Data?
) throws -> UnixHTTPResponse {
    let descriptor = try connectUnixSocket(path: socketPath)
    defer { Darwin.close(descriptor) }
    try sendAll(
        requestData(path: path, method: method, body: body, sessionToken: sessionToken),
        descriptor: descriptor
    )
    var received = Data()
    while let chunk = try receiveChunk(descriptor: descriptor) {
        received.append(chunk)
        if received.count > maximumHTTPResponseBytes + maximumHTTPHeaderBytes {
            throw RuntimeClientError.invalidResponse
        }
    }
    guard let boundary = headerBoundary(in: received) else {
        throw RuntimeClientError.invalidResponse
    }
    guard boundary.lowerBound <= maximumHTTPHeaderBytes else {
        throw RuntimeClientError.invalidResponse
    }
    let (status, headers) = try parseHeader(received[..<boundary.lowerBound])
    let bodyStart = boundary.upperBound
    return UnixHTTPResponse(status: status, headers: headers, body: received[bodyStart...])
}

private func streamUnixLines(
    socketPath: String,
    sessionToken: String?,
    path: String,
    method: String,
    body: Data?,
    headers: [String: String],
    lockedSocket: LockedSocket,
    consume: (String) throws -> Void
) throws {
    let descriptor = try connectUnixSocket(path: socketPath)
    lockedSocket.store(descriptor)
    defer { lockedSocket.close() }
    try sendAll(
        requestData(
            path: path,
            method: method,
            body: body,
            sessionToken: sessionToken,
            headers: headers
        ),
        descriptor: descriptor
    )
    var buffer = Data()
    while headerBoundary(in: buffer) == nil {
        guard let chunk = try receiveChunk(descriptor: descriptor) else {
            throw RuntimeClientError.invalidResponse
        }
        buffer.append(chunk)
        if buffer.count > maximumHTTPHeaderBytes { throw RuntimeClientError.invalidResponse }
    }
    guard let boundary = headerBoundary(in: buffer) else { throw RuntimeClientError.invalidResponse }
    let (status, responseHeaders) = try parseHeader(buffer[..<boundary.lowerBound])
    var response = UnixHTTPResponse(
        status: status,
        headers: responseHeaders,
        body: buffer[boundary.upperBound...]
    )
    try validate(response: response)
    buffer = response.body
    response = UnixHTTPResponse(status: status, headers: responseHeaders, body: Data())

    while true {
        while let newline = buffer.firstIndex(of: 0x0A) {
            var lineData = buffer[..<newline]
            if lineData.last == 0x0D { lineData = lineData.dropLast() }
            guard let line = String(data: lineData, encoding: .utf8) else {
                throw RuntimeClientError.invalidResponse
            }
            buffer.removeSubrange(...newline)
            try consume(line)
        }
        guard let chunk = try receiveChunk(descriptor: descriptor) else { break }
        buffer.append(chunk)
        if buffer.count > maximumHTTPResponseBytes { throw RuntimeClientError.invalidResponse }
    }
}
