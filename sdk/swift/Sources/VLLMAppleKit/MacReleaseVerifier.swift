import CryptoKit
import Foundation

public struct MacReleaseManifest: Codable, Sendable, Equatable {
    public struct Source: Codable, Sendable, Equatable { public let commit: String }
    public struct Build: Codable, Sendable, Equatable {
        public let runID: String
        public let runnerImage: String

        private enum CodingKeys: String, CodingKey {
            case runID = "runId"
            case runnerImage
        }
    }
    public struct Notarization: Codable, Sendable, Equatable {
        public let id: String
        public let status: String
        public let reportSha256: String
        public let reportSizeBytes: Int
    }
    public struct Component: Codable, Sendable, Equatable {
        public let path: String
        public let sha256: String
        public let sizeBytes: Int
    }
    public struct Artifact: Codable, Sendable, Equatable {
        public let filename: String
        public let sha256: String
        public let sizeBytes: Int
        public let bundleIdentifier: String
        public let bundleVersion: String
        public let shortVersion: String
        public let minimumSystemVersion: String
        public let components: [Component]
    }

    public let schemaVersion: Int
    public let createdAt: String
    public let source: Source
    public let build: Build
    public let notarization: Notarization
    public let artifact: Artifact
}

public enum MacReleaseVerificationError: Error, Sendable, Equatable {
    case invalidEvidence
    case evidenceTooLarge
    case checksumMismatch
}

/// Verifies the bounded, portable evidence shipped beside a notarized Mac archive.
/// Gatekeeper remains the authority for the notarized app after extraction.
public struct MacReleaseVerifier: Sendable {
    public static let maximumManifestBytes = 1_048_576
    public static let maximumArchiveBytes = 512 * 1_048_576
    private static let archiveName = "VLLMAppleChat-notarized-arm64.zip"
    private static let expectedComponents: Set<String> = [
        "VLLMAppleChat.app/Contents/MacOS/VLLMAppleChat",
        "VLLMAppleChat.app/Contents/MacOS/vllm-appled"
    ]

    public init() {}

    public func verify(directoryURL: URL) throws -> MacReleaseManifest {
        let manifestURL = directoryURL.appending(path: "release-manifest-v1.json")
        let reportURL = directoryURL.appending(path: "notary-result.json")
        let archiveURL = directoryURL.appending(path: Self.archiveName)
        let manifestData = try boundedRegularFile(
            manifestURL, maximumBytes: Self.maximumManifestBytes
        )
        let reportData = try boundedRegularFile(
            reportURL, maximumBytes: Self.maximumManifestBytes
        )
        let archiveSize = try regularFileSize(
            archiveURL, maximumBytes: Self.maximumArchiveBytes
        )
        let decoder = JSONDecoder()
        decoder.keyDecodingStrategy = .convertFromSnakeCase
        guard let manifest = try? decoder.decode(MacReleaseManifest.self, from: manifestData),
              let report = try? decoder.decode(NotaryResult.self, from: reportData),
              manifest.schemaVersion == 1,
              manifest.artifact.filename == Self.archiveName,
              manifest.artifact.sizeBytes == archiveSize,
              manifest.notarization.status == "Accepted",
              report.status == "Accepted",
              report.id == manifest.notarization.id,
              manifest.notarization.reportSizeBytes == reportData.count,
              isLowercaseHex(manifest.source.commit, count: 40),
              isLowercaseHex(manifest.artifact.sha256, count: 64),
              isLowercaseHex(manifest.notarization.reportSha256, count: 64),
              !manifest.artifact.bundleIdentifier.isEmpty,
              !manifest.artifact.bundleVersion.isEmpty,
              !manifest.artifact.shortVersion.isEmpty,
              !manifest.artifact.minimumSystemVersion.isEmpty,
              Set(manifest.artifact.components.map(\.path)) == Self.expectedComponents,
              manifest.artifact.components.count == Self.expectedComponents.count,
              manifest.artifact.components.allSatisfy({
                  $0.sizeBytes > 0 && isLowercaseHex($0.sha256, count: 64)
              }) else { throw MacReleaseVerificationError.invalidEvidence }

        guard sha256(reportData) == manifest.notarization.reportSha256,
              try sha256(archiveURL) == manifest.artifact.sha256 else {
            throw MacReleaseVerificationError.checksumMismatch
        }
        return manifest
    }

    private struct NotaryResult: Codable { let id: String; let status: String }

    private func boundedRegularFile(_ url: URL, maximumBytes: Int) throws -> Data {
        let size = try regularFileSize(url, maximumBytes: maximumBytes)
        guard let data = try? Data(contentsOf: url, options: .mappedIfSafe),
              data.count == size else { throw MacReleaseVerificationError.invalidEvidence }
        return data
    }

    private func regularFileSize(_ url: URL, maximumBytes: Int) throws -> Int {
        guard let values = try? url.resourceValues(
            forKeys: [.isRegularFileKey, .isSymbolicLinkKey, .fileSizeKey]
        ), values.isRegularFile == true, values.isSymbolicLink != true,
              let size = values.fileSize, size > 0 else {
            throw MacReleaseVerificationError.invalidEvidence
        }
        guard size <= maximumBytes else { throw MacReleaseVerificationError.evidenceTooLarge }
        return size
    }

    private func sha256(_ data: Data) -> String {
        SHA256.hash(data: data).map { String(format: "%02x", $0) }.joined()
    }

    private func sha256(_ url: URL) throws -> String {
        guard let handle = try? FileHandle(forReadingFrom: url) else {
            throw MacReleaseVerificationError.invalidEvidence
        }
        defer { try? handle.close() }
        var digest = SHA256()
        do {
            while let data = try handle.read(upToCount: 8 * 1_048_576), !data.isEmpty {
                digest.update(data: data)
            }
        } catch {
            throw MacReleaseVerificationError.invalidEvidence
        }
        return digest.finalize().map { String(format: "%02x", $0) }.joined()
    }

    private func isLowercaseHex(_ value: String, count: Int) -> Bool {
        value.count == count && value.allSatisfy { ("0"..."9").contains($0) || ("a"..."f").contains($0) }
    }
}
