import Foundation

enum OptimizationObjective: String, CaseIterable, Codable, Identifiable, Sendable {
    case balanced, memory, speed, quality
    var id: String { rawValue }
}

struct SourceModel: Codable, Sendable {
    let modelID: String
    let path: String
    let weightsBytes: Int64
    let metadataFingerprint: String
    let license: String?

    enum CodingKeys: String, CodingKey {
        case modelID = "model_id"
        case path
        case weightsBytes = "weights_bytes"
        case metadataFingerprint = "metadata_fingerprint"
        case license
    }
}

struct ResourceBudget: Codable, Sendable {
    let maximumMemoryBytes: Int64
    let maximumDiskBytes: Int64
    let maximumDurationSeconds: Int?

    enum CodingKeys: String, CodingKey {
        case maximumMemoryBytes = "maximum_memory_bytes"
        case maximumDiskBytes = "maximum_disk_bytes"
        case maximumDurationSeconds = "maximum_duration_seconds"
    }
}

struct OptimizationCandidate: Codable, Identifiable, Sendable {
    let strategy: String
    let targetWeightBits: Int
    let estimatedOutputBytes: Int64
    let requiredDiskBytes: Int64
    let estimatedPeakMemoryBytes: Int64
    let estimatedDurationSeconds: Int?
    let withinBudget: Bool
    let executable: Bool
    let blockingReasons: [String]

    var id: String { "\(strategy)-\(targetWeightBits)" }

    enum CodingKeys: String, CodingKey {
        case strategy
        case targetWeightBits = "target_weight_bits"
        case estimatedOutputBytes = "estimated_output_bytes"
        case requiredDiskBytes = "required_disk_bytes"
        case estimatedPeakMemoryBytes = "estimated_peak_memory_bytes"
        case estimatedDurationSeconds = "estimated_duration_seconds"
        case withinBudget = "within_budget"
        case executable
        case blockingReasons = "blocking_reasons"
    }
}

struct OptimizationPlan: Codable, Sendable {
    let schemaVersion: Int
    let planID: String
    let createdAt: String
    let objective: OptimizationObjective
    let source: SourceModel
    let outputPath: String
    let hardwareFingerprint: String
    let resourceBudget: ResourceBudget
    let candidates: [OptimizationCandidate]
    let warnings: [String]
    let dryRun: Bool

    enum CodingKeys: String, CodingKey {
        case schemaVersion = "schema_version"
        case planID = "plan_id"
        case createdAt = "created_at"
        case objective, source
        case outputPath = "output_path"
        case hardwareFingerprint = "hardware_fingerprint"
        case resourceBudget = "resource_budget"
        case candidates, warnings
        case dryRun = "dry_run"
    }

    static func decodeValidated(_ data: Data) throws -> OptimizationPlan {
        guard data.count <= 1_048_576 else { throw PlanValidationError.outputTooLarge }
        let decoder = JSONDecoder()
        decoder.keyDecodingStrategy = .useDefaultKeys
        let plan = try decoder.decode(Self.self, from: data)
        guard plan.schemaVersion == 1, plan.dryRun else {
            throw PlanValidationError.unsupportedContract
        }
        guard !plan.planID.isEmpty,
              plan.planID.utf8.count <= 256,
              plan.hardwareFingerprint.utf8.count >= 16,
              !plan.candidates.isEmpty,
              plan.candidates.count <= 64 else {
            throw PlanValidationError.invalidIdentity
        }
        guard plan.resourceBudget.maximumMemoryBytes > 0,
              plan.resourceBudget.maximumDiskBytes > 0,
              plan.candidates.allSatisfy({ candidate in
                  [candidate.estimatedOutputBytes,
                   candidate.requiredDiskBytes,
                   candidate.estimatedPeakMemoryBytes].allSatisfy { $0 > 0 }
                      && [4, 8, 16].contains(candidate.targetWeightBits)
                      && !candidate.strategy.isEmpty
              }) else {
            throw PlanValidationError.invalidEstimate
        }
        return plan
    }
}

enum PlanValidationError: LocalizedError {
    case outputTooLarge, unsupportedContract, invalidIdentity, invalidEstimate

    var errorDescription: String? {
        switch self {
        case .outputTooLarge: "Optimizer output exceeded the 1 MiB safety limit."
        case .unsupportedContract: "The optimizer returned an unsupported plan contract."
        case .invalidIdentity: "The optimizer plan identity is invalid."
        case .invalidEstimate: "The optimizer plan contains an invalid estimate."
        }
    }
}

struct WorkerResultSummary: Codable, Sendable {
    let state: String
    let outputPath: String?
    let outputBytes: Int64
    let fileCount: Int
    let outputHash: String?
    let elapsedMilliseconds: Int64
    let peakChildRSSBytes: Int64
    let errorCode: String?

    enum CodingKeys: String, CodingKey {
        case state
        case outputPath = "output_path"
        case outputBytes = "output_bytes"
        case fileCount = "file_count"
        case outputHash = "output_hash"
        case elapsedMilliseconds = "elapsed_milliseconds"
        case peakChildRSSBytes = "peak_child_rss_bytes"
        case errorCode = "error_code"
    }
}

struct ArtifactTransform: Codable, Sendable {
    let type: String
    let backend: String
    let weightBits: Int
    let groupSize: Int

    enum CodingKeys: String, CodingKey {
        case type, backend
        case weightBits = "weight_bits"
        case groupSize = "group_size"
    }
}

struct ArtifactManifestSummary: Codable, Sendable {
    let artifactID: String
    let sourceHash: String
    let outputHash: String
    let outputBytes: Int64
    let transforms: [ArtifactTransform]
    let toolVersions: [String: String]
    let evaluation: [String: Double]
    let license: String?

    enum CodingKeys: String, CodingKey {
        case artifactID = "artifact_id"
        case sourceHash = "source_hash"
        case outputHash = "output_hash"
        case outputBytes = "output_bytes"
        case transforms
        case toolVersions = "tool_versions"
        case evaluation, license
    }
}

struct ExportReport: Codable, Sendable {
    let schemaVersion: Int
    let workerResult: WorkerResultSummary
    let artifactManifest: ArtifactManifestSummary
    let manifestPath: String

    enum CodingKeys: String, CodingKey {
        case schemaVersion = "schema_version"
        case workerResult = "worker_result"
        case artifactManifest = "artifact_manifest"
        case manifestPath = "manifest_path"
    }

    static func decodeValidated(_ data: Data) throws -> Self {
        guard data.count <= 1_048_576 else { throw PlanValidationError.outputTooLarge }
        let report = try JSONDecoder().decode(Self.self, from: data)
        guard report.schemaVersion == 1,
              report.workerResult.state == "completed",
              report.workerResult.outputBytes > 0,
              report.workerResult.fileCount > 0,
              report.workerResult.outputHash == report.artifactManifest.outputHash,
              report.artifactManifest.outputHash.count == 64,
              report.manifestPath.hasPrefix("/") else {
            throw PlanValidationError.unsupportedContract
        }
        return report
    }
}

struct PerplexitySliceSummary: Codable, Identifiable, Sendable {
    let domain: String
    let language: String
    let sampleCount: Int
    let tokenCount: Int
    let perplexity: Double
    var id: String { "\(domain)/\(language)" }

    enum CodingKeys: String, CodingKey {
        case domain, language, perplexity
        case sampleCount = "sample_count"
        case tokenCount = "token_count"
    }
}

struct PerplexityReportSummary: Codable, Sendable {
    let schemaVersion: Int
    let modelHash: String
    let datasetFingerprint: String
    let sampleCount: Int
    let tokenCount: Int
    let perplexity: Double
    let elapsedMilliseconds: Int64
    let peakRSSBytes: Int64
    let slices: [PerplexitySliceSummary]

    enum CodingKeys: String, CodingKey {
        case schemaVersion = "schema_version"
        case modelHash = "model_hash"
        case datasetFingerprint = "dataset_fingerprint"
        case sampleCount = "sample_count"
        case tokenCount = "token_count"
        case perplexity
        case elapsedMilliseconds = "elapsed_milliseconds"
        case peakRSSBytes = "peak_rss_bytes"
        case slices
    }
}

struct QualityGateSliceSummary: Codable, Identifiable, Sendable {
    let domain: String
    let language: String
    let baselinePerplexity: Double
    let candidatePerplexity: Double
    let relativeRegression: Double
    let maximumRegression: Double
    let passed: Bool
    var id: String { "\(domain)/\(language)" }

    enum CodingKeys: String, CodingKey {
        case domain, language, passed
        case baselinePerplexity = "baseline_perplexity"
        case candidatePerplexity = "candidate_perplexity"
        case relativeRegression = "relative_regression"
        case maximumRegression = "maximum_regression"
    }
}

struct QualityComparisonReport: Codable, Sendable {
    let schemaVersion: Int
    let datasetFingerprint: String
    let approved: Bool
    let slices: [QualityGateSliceSummary]
    let untestedCapabilities: [String]

    enum CodingKeys: String, CodingKey {
        case schemaVersion = "schema_version"
        case datasetFingerprint = "dataset_fingerprint"
        case approved, slices
        case untestedCapabilities = "untested_capabilities"
    }

    static func decodeValidated(_ data: Data) throws -> Self {
        guard data.count <= 1_048_576 else { throw PlanValidationError.outputTooLarge }
        let report = try JSONDecoder().decode(Self.self, from: data)
        guard report.schemaVersion == 1,
              report.datasetFingerprint.count == 64,
              !report.slices.isEmpty,
              report.slices.count <= 64,
              report.approved == report.slices.allSatisfy(\.passed),
              !report.untestedCapabilities.isEmpty else {
            throw PlanValidationError.unsupportedContract
        }
        return report
    }
}

struct OptimizerEventSummary: Codable, Sendable {
    let schemaVersion: Int
    let planID: String
    let stage: String
    let state: String
    let progress: Double
    let messageKey: String

    enum CodingKeys: String, CodingKey {
        case schemaVersion = "schema_version"
        case planID = "plan_id"
        case stage, state, progress
        case messageKey = "message_key"
    }
}

struct GenerationEvaluationSummary: Codable, Sendable {
    let schemaVersion: Int
    let datasetFingerprint: String
    let elapsedMilliseconds: Int64
    let peakRSSBytes: Int64

    enum CodingKeys: String, CodingKey {
        case schemaVersion = "schema_version"
        case datasetFingerprint = "dataset_fingerprint"
        case elapsedMilliseconds = "elapsed_milliseconds"
        case peakRSSBytes = "peak_rss_bytes"
    }
}

struct GenerationGateSampleSummary: Codable, Identifiable, Sendable {
    let sampleID: String
    let domain: String
    let language: String
    let tokenAgreement: Double
    let baselineExpectationScore: Double
    let candidateExpectationScore: Double
    let passed: Bool
    var id: String { sampleID }

    enum CodingKeys: String, CodingKey {
        case sampleID = "sample_id"
        case domain, language, passed
        case tokenAgreement = "token_agreement"
        case baselineExpectationScore = "baseline_expectation_score"
        case candidateExpectationScore = "candidate_expectation_score"
    }
}

struct GenerationComparisonReport: Codable, Sendable {
    let schemaVersion: Int
    let approved: Bool
    let samples: [GenerationGateSampleSummary]
    let untestedCapabilities: [String]

    enum CodingKeys: String, CodingKey {
        case schemaVersion = "schema_version"
        case approved, samples
        case untestedCapabilities = "untested_capabilities"
    }

    static func decodeValidated(_ data: Data) throws -> Self {
        guard data.count <= 1_048_576 else { throw PlanValidationError.outputTooLarge }
        let report = try JSONDecoder().decode(Self.self, from: data)
        guard report.schemaVersion == 1,
              !report.samples.isEmpty,
              report.samples.count <= 64,
              report.approved == report.samples.allSatisfy(\.passed),
              report.samples.allSatisfy({ $0.tokenAgreement.isFinite && 0...1 ~= $0.tokenAgreement })
        else { throw PlanValidationError.unsupportedContract }
        return report
    }
}

struct GenerationComparison: Sendable {
    let baseline: GenerationEvaluationSummary
    let candidate: GenerationEvaluationSummary
    let gate: GenerationComparisonReport
}
