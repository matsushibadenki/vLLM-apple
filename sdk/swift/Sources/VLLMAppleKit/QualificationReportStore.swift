import Foundation

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
            options: [.skipsHiddenFiles, .skipsSubdirectoryDescendants]
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
