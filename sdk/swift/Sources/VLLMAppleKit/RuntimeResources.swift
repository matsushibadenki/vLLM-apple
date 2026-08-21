import Darwin
import Foundation

public struct RuntimeResources: Sendable, Equatable {
    public let daemonExecutableURL: URL?
    public let applicationSupportURL: URL
    public let profileDirectoryURL: URL
    public let logDirectoryURL: URL
    public let socketURL: URL
    public let sessionTokenURL: URL

    public init(
        daemonExecutableURL: URL?,
        applicationSupportURL: URL,
        profileDirectoryURL: URL,
        logDirectoryURL: URL,
        socketURL: URL,
        sessionTokenURL: URL
    ) {
        self.daemonExecutableURL = daemonExecutableURL
        self.applicationSupportURL = applicationSupportURL
        self.profileDirectoryURL = profileDirectoryURL
        self.logDirectoryURL = logDirectoryURL
        self.socketURL = socketURL
        self.sessionTokenURL = sessionTokenURL
    }
}

public enum RuntimeResourceError: Error, Sendable, Equatable {
    case invalidApplicationIdentifier
    case applicationSupportUnavailable
    case daemonNotExecutable
    case insecureExistingPath(String)
    case socketPathTooLong(Int)
    case filesystem(String)

    public var messageKey: String {
        switch self {
        case .invalidApplicationIdentifier: "runtime.error.invalid_application_identifier"
        case .applicationSupportUnavailable: "runtime.error.application_support_unavailable"
        case .daemonNotExecutable: "runtime.error.daemon_not_executable"
        case .insecureExistingPath: "runtime.error.insecure_existing_path"
        case .socketPathTooLong: "runtime.error.socket_path_too_long"
        case .filesystem: "runtime.error.filesystem"
        }
    }
}

/// Resolves all mutable runtime files outside the app bundle and creates them with
/// private permissions. The Unix socket intentionally lives below the temporary
/// directory because Darwin limits `sun_path` to 104 bytes.
public struct RuntimeResourceResolver: Sendable {
    public static let maximumUnixSocketPathBytes = 103

    private let applicationIdentifier: String
    private let explicitDaemonURL: URL?
    private let bundledDaemonURL: URL?
    private let applicationSupportRoot: URL?
    private let temporaryRoot: URL

    public init(
        applicationIdentifier: String,
        daemonExecutableURL: URL? = nil,
        bundle: Bundle = .main,
        applicationSupportRoot: URL? = nil,
        temporaryRoot: URL? = nil
    ) {
        self.applicationIdentifier = applicationIdentifier
        self.explicitDaemonURL = daemonExecutableURL
        let resourceCandidate = bundle.resourceURL?.appending(path: "vllm-appled")
        if let auxiliary = bundle.url(forAuxiliaryExecutable: "vllm-appled") {
            self.bundledDaemonURL = auxiliary
        } else if let resourceCandidate,
                  FileManager.default.fileExists(atPath: resourceCandidate.path) {
            self.bundledDaemonURL = resourceCandidate
        } else {
            self.bundledDaemonURL = nil
        }
        self.applicationSupportRoot = applicationSupportRoot
        // Keep the textual path short: Darwin stores the original path in a
        // 104-byte `sockaddr_un.sun_path` buffer.
        self.temporaryRoot = temporaryRoot
            ?? URL(fileURLWithPath: "/tmp", isDirectory: true)
    }

    public func resolve(createSessionToken: Bool = true) throws -> RuntimeResources {
        let component = try safeComponent(applicationIdentifier)
        let supportRoot = try resolvedApplicationSupportRoot()
        let support = supportRoot.appending(path: component, directoryHint: .isDirectory)
        let profiles = support.appending(path: "Profiles", directoryHint: .isDirectory)
        let logs = support.appending(path: "Logs", directoryHint: .isDirectory)
        let socketDirectory = temporaryRoot.appending(
            path: "vllm-apple-\(getuid())-\(component)",
            directoryHint: .isDirectory
        )
        let socket = socketDirectory.appending(path: "runtime.sock")
        let token = support.appending(path: "session.token")

        try createPrivateDirectory(support)
        try createPrivateDirectory(profiles)
        try createPrivateDirectory(logs)
        try createPrivateDirectory(socketDirectory)
        try validateSocketPath(socket.path)
        if createSessionToken {
            try ensurePrivateSessionToken(at: token)
        }

        return RuntimeResources(
            daemonExecutableURL: try resolveDaemonExecutable(),
            applicationSupportURL: support,
            profileDirectoryURL: profiles,
            logDirectoryURL: logs,
            socketURL: socket,
            sessionTokenURL: token
        )
    }

    private func resolvedApplicationSupportRoot() throws -> URL {
        if let applicationSupportRoot { return applicationSupportRoot }
        guard let value = FileManager.default.urls(
            for: .applicationSupportDirectory,
            in: .userDomainMask
        ).first else {
            throw RuntimeResourceError.applicationSupportUnavailable
        }
        return value
    }

    private func resolveDaemonExecutable() throws -> URL? {
        if let explicitDaemonURL {
            guard FileManager.default.isExecutableFile(atPath: explicitDaemonURL.path) else {
                throw RuntimeResourceError.daemonNotExecutable
            }
            return explicitDaemonURL
        }
        guard let bundledDaemonURL else { return nil }
        guard FileManager.default.isExecutableFile(atPath: bundledDaemonURL.path) else {
            throw RuntimeResourceError.daemonNotExecutable
        }
        return bundledDaemonURL
    }

    private func safeComponent(_ value: String) throws -> String {
        let allowed = CharacterSet.alphanumerics.union(CharacterSet(charactersIn: ".-_"))
        guard !value.isEmpty,
              value.count <= 48,
              value.unicodeScalars.allSatisfy(allowed.contains),
              value != ".", value != ".." else {
            throw RuntimeResourceError.invalidApplicationIdentifier
        }
        return value
    }

    private func createPrivateDirectory(_ url: URL) throws {
        let manager = FileManager.default
        do {
            if manager.fileExists(atPath: url.path) {
                let attributes = try manager.attributesOfItem(atPath: url.path)
                guard attributes[.type] as? FileAttributeType == .typeDirectory,
                      attributes[.ownerAccountID] as? NSNumber == NSNumber(value: getuid()) else {
                    throw RuntimeResourceError.insecureExistingPath(url.path)
                }
            } else {
                try manager.createDirectory(
                    at: url,
                    withIntermediateDirectories: true,
                    attributes: [.posixPermissions: 0o700]
                )
            }
            try manager.setAttributes([.posixPermissions: 0o700], ofItemAtPath: url.path)
        } catch let error as RuntimeResourceError {
            throw error
        } catch {
            throw RuntimeResourceError.filesystem(error.localizedDescription)
        }
    }

    private func validateSocketPath(_ path: String) throws {
        let length = path.utf8.count
        guard length <= Self.maximumUnixSocketPathBytes else {
            throw RuntimeResourceError.socketPathTooLong(length)
        }
    }

    private func ensurePrivateSessionToken(at url: URL) throws {
        let manager = FileManager.default
        if manager.fileExists(atPath: url.path) {
            do {
                let attributes = try manager.attributesOfItem(atPath: url.path)
                guard attributes[.type] as? FileAttributeType == .typeRegular,
                      attributes[.ownerAccountID] as? NSNumber == NSNumber(value: getuid()) else {
                    throw RuntimeResourceError.insecureExistingPath(url.path)
                }
                try manager.setAttributes([.posixPermissions: 0o600], ofItemAtPath: url.path)
                let token = try String(contentsOf: url, encoding: .utf8)
                    .trimmingCharacters(in: .whitespacesAndNewlines)
                guard token.count >= 32 else {
                    throw RuntimeResourceError.insecureExistingPath(url.path)
                }
                return
            } catch let error as RuntimeResourceError {
                throw error
            } catch {
                throw RuntimeResourceError.filesystem(error.localizedDescription)
            }
        }

        var generator = SystemRandomNumberGenerator()
        let value = (0..<32).map { _ in UInt8.random(in: .min ... .max, using: &generator) }
            .map { String(format: "%02x", $0) }
            .joined()
        let temporary = url.deletingLastPathComponent()
            .appending(path: ".session-token-\(UUID().uuidString)")
        do {
            guard manager.createFile(
                atPath: temporary.path,
                contents: Data((value + "\n").utf8),
                attributes: [.posixPermissions: 0o600]
            ) else {
                throw RuntimeResourceError.filesystem("Unable to create session token")
            }
            let handle = try FileHandle(forWritingTo: temporary)
            try handle.synchronize()
            try handle.close()
            do {
                try manager.moveItem(at: temporary, to: url)
            } catch where manager.fileExists(atPath: url.path) {
                try? manager.removeItem(at: temporary)
                try ensurePrivateSessionToken(at: url)
                return
            }
            try manager.setAttributes([.posixPermissions: 0o600], ofItemAtPath: url.path)
        } catch let error as RuntimeResourceError {
            try? manager.removeItem(at: temporary)
            throw error
        } catch {
            try? manager.removeItem(at: temporary)
            throw RuntimeResourceError.filesystem(error.localizedDescription)
        }
    }
}
