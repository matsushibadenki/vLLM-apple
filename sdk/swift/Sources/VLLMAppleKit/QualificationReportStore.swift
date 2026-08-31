import Foundation
import CryptoKit
import Darwin

public struct QualificationReportRecord: Identifiable, Sendable, Equatable {
    public var id: URL { fileURL }
    public let fileURL: URL
    public let modifiedAt: Date
    public let report: QualificationReport
}

/// Bounded, fail-soft reader for reports produced by `vllm-apple qualify-model`.
public struct QualificationReportStore: Sendable {
    public static let defaultMaximumReports = 32
    public static let defaultMaximumFileBytes = 1_048_576
    private static let maximumCandidateFiles = 128
    private static let maximumScannedEntries = 4_096

    public let directoryURL: URL
    public let maximumReports: Int
    public let maximumFileBytes: Int

    public init(
        directoryURL: URL = Self.defaultDirectoryURL(),
        maximumReports: Int = Self.defaultMaximumReports,
        maximumFileBytes: Int = Self.defaultMaximumFileBytes
    ) {
        self.directoryURL = directoryURL
        self.maximumReports = max(0, maximumReports)
        self.maximumFileBytes = max(0, maximumFileBytes)
    }

    public static func defaultDirectoryURL(fileManager: FileManager = .default) -> URL {
        let support = fileManager.urls(for: .applicationSupportDirectory, in: .userDomainMask).first
            ?? URL(fileURLWithPath: NSHomeDirectory())
                .appending(path: "Library/Application Support", directoryHint: .isDirectory)
        return support
            .appending(path: "vllm-apple", directoryHint: .isDirectory)
            .appending(path: "qualification-reports", directoryHint: .isDirectory)
    }

    public func load() -> [QualificationReportRecord] {
        guard maximumReports > 0, maximumFileBytes > 0 else { return [] }
        let keys: Set<URLResourceKey> = [
            .isRegularFileKey, .isSymbolicLinkKey, .fileSizeKey, .contentModificationDateKey
        ]
        guard let enumerator = FileManager.default.enumerator(
            at: directoryURL,
            includingPropertiesForKeys: Array(keys),
            options: [.skipsHiddenFiles]
        ) else { return [] }

        var eligible: [(URL, Date)] = []
        var scanned = 0
        while scanned < Self.maximumScannedEntries,
              let url = enumerator.nextObject() as? URL {
            scanned += 1
            guard url.pathExtension.lowercased() == "json",
                  let values = try? url.resourceValues(forKeys: keys),
                  values.isRegularFile == true,
                  values.isSymbolicLink != true,
                  let size = values.fileSize,
                  size > 0, size <= maximumFileBytes else { continue }
            eligible.append((url, values.contentModificationDate ?? .distantPast))
        }
        eligible.sort { $0.1 > $1.1 }

        let decoder = JSONDecoder()
        decoder.keyDecodingStrategy = .convertFromSnakeCase
        let candidateLimit = min(Self.maximumCandidateFiles, max(maximumReports, maximumReports * 4))
        let decoded: [QualificationReportRecord] = eligible.prefix(candidateLimit).compactMap {
            url, modifiedAt -> QualificationReportRecord? in
            guard let data = try? Data(contentsOf: url, options: .mappedIfSafe),
                  data.count <= maximumFileBytes,
                  let report = try? decoder.decode(QualificationReport.self, from: data),
                  report.schemaVersion == 1 else { return nil }
            return QualificationReportRecord(fileURL: url, modifiedAt: modifiedAt, report: report)
        }
        return Array(decoded.prefix(maximumReports))
    }
}

public enum QualificationPromotionImportError: Error, Sendable, Equatable {
    case sourceInvalid
    case destinationExists
    case filesystem
    case copiedEvidenceInvalid
}

public struct QualificationPromotionSignatureRequirement: Sendable, Equatable {
    public let trustedCAURL: URL
    public let expectedSignerSHA256: String

    public init(trustedCAURL: URL, expectedSignerSHA256: String) {
        self.trustedCAURL = trustedCAURL
        self.expectedSignerSHA256 = expectedSignerSHA256
    }
}

public struct QualificationPromotionBundleImporter: Sendable {
    public let destinationRootURL: URL

    public init(
        destinationRootURL: URL = QualificationReportStore.defaultDirectoryURL()
    ) {
        self.destinationRootURL = destinationRootURL
    }

    public func importDirectory(
        _ sourceURL: URL,
        signatureRequirement: QualificationPromotionSignatureRequirement? = nil
    ) throws -> URL {
        guard let qualification = QualificationReportStore(
            directoryURL: sourceURL, maximumReports: 4
        ).load().first(where: { $0.fileURL.lastPathComponent == "qualification.json" }) else {
            throw QualificationPromotionImportError.sourceInvalid
        }
        let sourceBundleURL = sourceURL.appending(
            path: "promotion-bundle.json", directoryHint: .notDirectory
        )
        let sourceStore = QualificationPromotionBundleStore(fileURL: sourceBundleURL)
        guard let bundle = sourceStore.loadValidated(
            reportsDirectoryURL: sourceURL, qualification: qualification.report
        ) else { throw QualificationPromotionImportError.sourceInvalid }
        if let signatureRequirement {
            do {
                try QualificationCMSVerifier().verify(
                    bundleURL: sourceBundleURL,
                    signatureURL: sourceURL.appending(path: "promotion-bundle.cms"),
                    trustedCAURL: signatureRequirement.trustedCAURL,
                    expectedSignerSHA256: signatureRequirement.expectedSignerSHA256
                )
            } catch {
                throw QualificationPromotionImportError.sourceInvalid
            }
        }
        let manager = FileManager.default
        do {
            try manager.createDirectory(
                at: destinationRootURL,
                withIntermediateDirectories: true,
                attributes: [.posixPermissions: 0o700]
            )
        } catch {
            throw QualificationPromotionImportError.filesystem
        }
        var rootAttributes = stat()
        guard lstat(destinationRootURL.path, &rootAttributes) == 0,
              rootAttributes.st_mode & S_IFMT == S_IFDIR,
              rootAttributes.st_uid == getuid(),
              chmod(destinationRootURL.path, 0o700) == 0 else {
            throw QualificationPromotionImportError.filesystem
        }
        let destination = destinationRootURL.appending(
            path: bundle.bundleID, directoryHint: .isDirectory
        )
        guard !manager.fileExists(atPath: destination.path) else {
            throw QualificationPromotionImportError.destinationExists
        }
        let staging = destinationRootURL.appending(
            path: ".import-\(UUID().uuidString)", directoryHint: .isDirectory
        )
        do {
            try manager.createDirectory(
                at: staging,
                withIntermediateDirectories: false,
                attributes: [.posixPermissions: 0o700]
            )
            var names = Array(bundle.evidenceSha256.keys) + ["promotion-bundle.json"]
            if signatureRequirement != nil { names.append("promotion-bundle.cms") }
            for name in names {
                let source = sourceURL.appending(path: name, directoryHint: .notDirectory)
                let target = staging.appending(path: name, directoryHint: .notDirectory)
                try manager.copyItem(at: source, to: target)
                try manager.setAttributes([.posixPermissions: 0o600], ofItemAtPath: target.path)
            }
            guard let copiedQualification = QualificationReportStore(
                directoryURL: staging, maximumReports: 4
            ).load().first(where: { $0.fileURL.lastPathComponent == "qualification.json" }),
                  QualificationPromotionBundleStore(
                    fileURL: staging.appending(path: "promotion-bundle.json")
                  ).loadValidated(
                    reportsDirectoryURL: staging,
                    qualification: copiedQualification.report
                  ) != nil else {
                throw QualificationPromotionImportError.copiedEvidenceInvalid
            }
            try manager.moveItem(at: staging, to: destination)
            return destination
        } catch let error as QualificationPromotionImportError {
            try? manager.removeItem(at: staging)
            throw error
        } catch {
            try? manager.removeItem(at: staging)
            throw QualificationPromotionImportError.filesystem
        }
    }
}

public struct ArtifactAdmissionReportStore: Sendable {
    public static let defaultMaximumFileBytes = 65_536
    public let fileURL: URL
    public let maximumFileBytes: Int

    public init(fileURL: URL, maximumFileBytes: Int = Self.defaultMaximumFileBytes) {
        self.fileURL = fileURL
        self.maximumFileBytes = max(0, maximumFileBytes)
    }

    public func load() -> ArtifactAdmissionReport? {
        guard maximumFileBytes > 0,
              let values = try? fileURL.resourceValues(
                forKeys: [.isRegularFileKey, .isSymbolicLinkKey, .fileSizeKey]
              ),
              values.isRegularFile == true,
              values.isSymbolicLink != true,
              let size = values.fileSize,
              size > 0, size <= maximumFileBytes,
              let data = try? Data(contentsOf: fileURL, options: .mappedIfSafe),
              data.count <= maximumFileBytes else { return nil }
        let decoder = JSONDecoder()
        decoder.keyDecodingStrategy = .convertFromSnakeCase
        guard let report = try? decoder.decode(ArtifactAdmissionReport.self, from: data),
              report.schemaVersion == 1 else { return nil }
        return report
    }
}

public struct QualificationPromotionBundleStore: Sendable {
    public static let defaultMaximumFileBytes = 65_536
    public let fileURL: URL
    public let maximumFileBytes: Int

    public init(fileURL: URL, maximumFileBytes: Int = Self.defaultMaximumFileBytes) {
        self.fileURL = fileURL
        self.maximumFileBytes = max(0, maximumFileBytes)
    }

    public func loadValidated(
        reportsDirectoryURL: URL,
        qualification: QualificationReport
    ) -> QualificationPromotionBundle? {
        guard let data = boundedRegularFile(fileURL, maximumBytes: maximumFileBytes) else {
            return nil
        }
        let decoder = JSONDecoder()
        decoder.keyDecodingStrategy = .convertFromSnakeCase
        guard let bundle = try? decoder.decode(QualificationPromotionBundle.self, from: data),
              bundle.schemaVersion == 1,
              bundle.passed,
              bundle.requestedModes == ["text"],
              bundle.backend == qualification.backend,
              bundle.backendVersions == qualification.backendVersions,
              bundle.modelSha256 == sha256(Data(qualification.model.utf8)),
              isSHA256(bundle.bundleId) else { return nil }

        let expectedNames: Set<String> = FileManager.default.fileExists(
            atPath: reportsDirectoryURL.appending(path: "artifact-admission.json").path
        ) ? ["preflight.json", "qualification.json", "artifact-admission.json"]
          : ["preflight.json", "qualification.json"]
        guard Set(bundle.evidenceSha256.keys) == expectedNames else { return nil }
        for name in expectedNames {
            let evidenceURL = reportsDirectoryURL.appending(path: name, directoryHint: .notDirectory)
            guard let evidence = boundedRegularFile(evidenceURL, maximumBytes: 1_048_576),
                  bundle.evidenceSha256[name] == sha256(evidence) else { return nil }
        }
        guard recomputedBundleID(bundle) == bundle.bundleId else { return nil }
        return bundle
    }

    private func recomputedBundleID(_ bundle: QualificationPromotionBundle) -> String? {
        var body: [String: Any] = [
            "schema_version": bundle.schemaVersion,
            "model_sha256": bundle.modelSha256,
            "backend": bundle.backend,
            "requested_modes": bundle.requestedModes,
            "evidence_sha256": bundle.evidenceSha256,
            "passed": bundle.passed
        ]
        if let versions = bundle.backendVersions {
            var encoded: [String: Any] = [:]
            if let value = versions.vllm { encoded["vllm"] = value }
            if let value = versions.vllmMetal { encoded["vllm_metal"] = value }
            if let value = versions.transformers { encoded["transformers"] = value }
            if let value = versions.mlxLm { encoded["mlx_lm"] = value }
            body["backend_versions"] = encoded
        } else {
            body["backend_versions"] = NSNull()
        }
        guard let canonical = try? JSONSerialization.data(
            withJSONObject: body, options: [.sortedKeys, .withoutEscapingSlashes]
        ) else { return nil }
        return sha256(canonical)
    }

    private func boundedRegularFile(_ url: URL, maximumBytes: Int) -> Data? {
        guard maximumBytes > 0 else { return nil }
        let descriptor = Darwin.open(url.path, O_RDONLY | O_NOFOLLOW)
        guard descriptor >= 0 else { return nil }
        defer { Darwin.close(descriptor) }
        var before = stat()
        guard fstat(descriptor, &before) == 0,
              before.st_mode & S_IFMT == S_IFREG,
              before.st_size > 0,
              before.st_size <= off_t(maximumBytes) else { return nil }
        var data = Data()
        data.reserveCapacity(Int(before.st_size))
        var buffer = [UInt8](repeating: 0, count: min(65_536, maximumBytes + 1))
        while true {
            let count = Darwin.read(descriptor, &buffer, buffer.count)
            guard count >= 0 else { return nil }
            if count == 0 { break }
            guard data.count + count <= maximumBytes else { return nil }
            data.append(buffer, count: count)
        }
        var after = stat()
        guard fstat(descriptor, &after) == 0,
              before.st_dev == after.st_dev,
              before.st_ino == after.st_ino,
              before.st_size == after.st_size,
              before.st_mtimespec.tv_sec == after.st_mtimespec.tv_sec,
              before.st_mtimespec.tv_nsec == after.st_mtimespec.tv_nsec,
              data.count == Int(before.st_size) else { return nil }
        return data
    }

    private func sha256(_ data: Data) -> String {
        SHA256.hash(data: data).map { String(format: "%02x", $0) }.joined()
    }

    private func isSHA256(_ value: String) -> Bool {
        value.utf8.count == 64 && value.allSatisfy { $0.isHexDigit && !$0.isUppercase }
    }
}
