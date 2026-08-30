import Foundation

public enum RuntimeState: String, Codable, Sendable {
    case stopped
    case starting
    case profiling
    case loadingModel = "loading_model"
    case ready
    case degraded
    case failed
    case stopping
}

public enum MemoryPressure: String, Codable, Sendable {
    case normal
    case warning
    case critical
    case unknown
}

public enum RuntimeRecoverability: String, Codable, Sendable {
    case retryable
    case userActionRequired = "user_action_required"
    case fatal
}

public struct RuntimeFailure: Codable, Sendable, Equatable {
    public let schemaVersion: Int
    public let code: String
    public let messageKey: String
    public let recoverability: RuntimeRecoverability
    public let detailFingerprint: String

    enum CodingKeys: String, CodingKey {
        case code, recoverability
        case schemaVersion = "schema_version"
        case messageKey = "message_key"
        case detailFingerprint = "detail_fingerprint"
    }
}

public struct MemoryInfo: Codable, Sendable, Equatable {
    public let totalBytes: Int64
    public let availableBytes: Int64
    public let processResidentBytes: Int64
    public let pressure: MemoryPressure
    public let source: String
}

public struct HardwareInfo: Codable, Sendable, Equatable {
    public let platform: String
    public let architecture: String
    public let soc: String
    public let physicalCPUCount: Int
    public let logicalCPUCount: Int
    public let gpuCoreCount: Int?
    public let memory: MemoryInfo
    public let isAppleSilicon: Bool
    public let osVersion: String

    enum CodingKeys: String, CodingKey {
        case platform, architecture, soc, memory
        case physicalCPUCount = "physical_cpu_count"
        case logicalCPUCount = "logical_cpu_count"
        case gpuCoreCount = "gpu_core_count"
        case isAppleSilicon = "is_apple_silicon"
        case osVersion = "os_version"
    }
}

public struct ContextTier: Codable, Sendable, Equatable {
    public let name: String
    public let maxTokens: Int
    public let kvBudgetBytes: Int64
}

public struct ContextRecommendation: Codable, Sendable, Equatable {
    public let modelID: String
    public let allocatableBytes: Int64
    public let osReserveBytes: Int64
    public let safetyHeadroomBytes: Int64
    public let workspaceBytes: Int64
    public let tiers: [ContextTier]
    public let limitingFactor: String?

    enum CodingKeys: String, CodingKey {
        case tiers
        case modelID = "model_id"
        case allocatableBytes = "allocatable_bytes"
        case osReserveBytes = "os_reserve_bytes"
        case safetyHeadroomBytes = "safety_headroom_bytes"
        case workspaceBytes = "workspace_bytes"
        case limitingFactor = "limiting_factor"
    }
}

public struct RuntimeProfile: Codable, Sendable, Equatable {
    public let profileVersion: Int
    public let runtimeVersion: String
    public let createdAt: String
    public let hardware: HardwareInfo
    public let context: ContextRecommendation?
    public let capabilities: [String]
}

public enum MemoryBudgetAccounting: String, Codable, Sendable {
    case additive
    case overlapEnvelope = "overlap_envelope"
}

public struct MemoryBudgetComponent: Codable, Sendable, Equatable {
    public let currentBytes: Int64?
    public let peakBytes: Int64?
    public let source: String?
    public let accounting: MemoryBudgetAccounting
}

public struct MemoryBudgetComponents: Codable, Sendable, Equatable {
    public let weights: MemoryBudgetComponent
    public let kv: MemoryBudgetComponent
    public let prefix: MemoryBudgetComponent
    public let scratch: MemoryBudgetComponent
    public let metalHeap: MemoryBudgetComponent
    public let coreml: MemoryBudgetComponent
}

public struct MemoryBudget: Codable, Sendable, Equatable {
    public let capacityBytes: Int64
    public let knownComponentBytes: Int64
    public let knownRemainingBytes: Int64
    public let overcommittedBytes: Int64
    public let unknownComponents: [String]
    public let overlapEnvelopeBytes: Int64?
    public let components: MemoryBudgetComponents
}

public enum KVCalibrationStatus: String, Codable, Sendable {
    case notConfigured = "not_configured"
    case disabled
    case notFound = "not_found"
    case invalid
    case applied
}

public struct KVCalibrationProvenance: Codable, Sendable, Equatable {
    public let enabled: Bool
    public let status: KVCalibrationStatus
    public let backend: String?
    public let evaluationID: String?
    public let calibratedBytesPerToken: Int64?
    public let maximumObservedContext: Int?
    public let sampleCount: Int?
    public let safetyMarginRatio: Double?

    enum CodingKeys: String, CodingKey {
        case enabled, status, backend
        case evaluationID = "evaluation_id"
        case calibratedBytesPerToken = "calibrated_bytes_per_token"
        case maximumObservedContext = "maximum_observed_context"
        case sampleCount = "sample_count"
        case safetyMarginRatio = "safety_margin_ratio"
    }
}

public enum NativeV2TuningStatus: String, Codable, Sendable {
    case disabled
    case idle
    case waitingForIdle = "waiting_for_idle"
    case running
    case applied
    case failed
}

public struct NativeV2TuningState: Codable, Sendable, Equatable {
    public let enabled: Bool
    public let status: NativeV2TuningStatus
    public let runID: Int
    public let profileID: String?
    public let errorCode: String?
    public let quarantinedProfiles: Int
    public let latestQuarantinedProfileID: String?

    public static let idle = NativeV2TuningState(
        enabled: true,
        status: .idle,
        runID: 0,
        profileID: nil,
        errorCode: nil,
        quarantinedProfiles: 0,
        latestQuarantinedProfileID: nil
    )

    public init(
        enabled: Bool,
        status: NativeV2TuningStatus,
        runID: Int,
        profileID: String?,
        errorCode: String?,
        quarantinedProfiles: Int = 0,
        latestQuarantinedProfileID: String? = nil
    ) {
        self.enabled = enabled
        self.status = status
        self.runID = runID
        self.profileID = profileID
        self.errorCode = errorCode
        self.quarantinedProfiles = quarantinedProfiles
        self.latestQuarantinedProfileID = latestQuarantinedProfileID
    }

    enum CodingKeys: String, CodingKey {
        case enabled, status
        case runID = "run_id"
        case profileID = "profile_id"
        case errorCode = "error_code"
        case quarantinedProfiles = "quarantined_profiles"
        case latestQuarantinedProfileID = "latest_quarantined_profile_id"
    }

    public init(from decoder: Decoder) throws {
        let container = try decoder.container(keyedBy: CodingKeys.self)
        enabled = try container.decodeIfPresent(Bool.self, forKey: .enabled) ?? true
        status = try container.decode(NativeV2TuningStatus.self, forKey: .status)
        runID = try container.decodeIfPresent(Int.self, forKey: .runID) ?? 0
        profileID = try container.decodeIfPresent(String.self, forKey: .profileID)
        errorCode = try container.decodeIfPresent(String.self, forKey: .errorCode)
        quarantinedProfiles = try container.decodeIfPresent(
            Int.self, forKey: .quarantinedProfiles
        ) ?? 0
        latestQuarantinedProfileID = try container.decodeIfPresent(
            String.self, forKey: .latestQuarantinedProfileID
        )
    }
}

public enum ContextReevaluationStatus: String, Codable, Sendable {
    case disabled
    case pending
    case sufficient
    case reduced
}

public struct RuntimeContextReevaluation: Sendable, Equatable {
    public let status: ContextReevaluationStatus
    public let configuredContextTokens: Int
    public let effectiveContextTokens: Int
    public let capacityContextTokens: Int?
}

public enum QualificationContextStatus: String, Codable, Sendable {
    case unavailable
    case sufficient
    case reduced
}

public struct QualificationContextReevaluation: Codable, Sendable, Equatable {
    public let enabled: Bool
    public let status: QualificationContextStatus
    public let configuredContextTokens: Int?
    public let effectiveContextTokens: Int?
    public let capacityContextTokens: Int?
    public let kvCapacityBytes: Int64?
    public let kvBytesPerToken: Int64?
    public let weightsBytes: Int64?
    public let source: String?
    public let reevaluations: Int
    public let passed: Bool
}

public struct QualificationLatencySummary: Codable, Sendable, Equatable {
    public let meanMs: Double
    public let maxMs: Double
}

public struct QualificationPrefillProfile: Codable, Sendable, Equatable {
    public let promptTokens: Int
    public let ttft: QualificationLatencySummary
}

public struct QualificationDecodeProfile: Codable, Sendable, Equatable {
    public let outputTokens: Int
    public let tokenIntervals: Int
    public let durationMs: Double
    public let tpot: QualificationLatencySummary
    public let tokensPerSecond: Double
}

public struct QualificationPhaseProfile: Codable, Sendable, Equatable {
    public let schemaVersion: Int
    public let profileId: String
    public let hardwareFingerprint: String
    public let modelId: String
    public let backend: String
    public let sampleCount: Int
    public let prefill: QualificationPrefillProfile
    public let decode: QualificationDecodeProfile
    public let peakMemoryBytes: Int64

    public var profileID: String { profileId }
    public var modelID: String { modelId }
}

public struct QualificationModelMemoryFit: Codable, Sendable, Equatable {
    public let artifactBytes: Int64
    public let estimatedResidentBytes: Int64
    public let hardCeilingBytes: Int64
    public let contextTokens: Int
    public let fits: Bool
}

public struct QualificationReport: Codable, Sendable, Equatable {
    public let schemaVersion: Int
    public let model: String
    public let backend: String
    public let loadSeconds: Double
    public let shutdownClean: Bool
    public let promotionProbe: [String: JSONValue]?
    public let phaseProfile: QualificationPhaseProfile?
    public let modelMemoryFit: QualificationModelMemoryFit?
    public let soak: [String: JSONValue]?
    public let contextReevaluation: QualificationContextReevaluation
    public let passed: Bool
}

public struct HealthStatus: Codable, Sendable, Equatable {
    public let status: RuntimeState
    public let controlReady: Bool
    public let inferenceReady: Bool
    public let apiVersion: String
    public let schemaVersion: Int
    public let runtimeVersion: String
    public let minimumClientVersion: String
}

public struct ChatMessage: Codable, Sendable, Equatable {
    public let role: String
    public let content: String

    public init(role: String, content: String) {
        self.role = role
        self.content = content
    }
}

public struct ChatRequest: Codable, Sendable, Equatable {
    public let model: String
    public let messages: [ChatMessage]
    public var temperature: Double?
    public var maxTokens: Int?
    public var stream: Bool

    public init(
        model: String,
        messages: [ChatMessage],
        temperature: Double? = nil,
        maxTokens: Int? = nil,
        stream: Bool = false
    ) {
        self.model = model
        self.messages = messages
        self.temperature = temperature
        self.maxTokens = maxTokens
        self.stream = stream
    }
}

public struct ChatChoice: Codable, Sendable, Equatable {
    public let index: Int
    public let message: ChatMessage
    public let finishReason: String?
}

public struct ChatResponse: Codable, Sendable, Equatable {
    public let id: String
    public let object: String
    public let created: Int
    public let model: String?
    public let choices: [ChatChoice]
}

public enum ChatEvent: Sendable, Equatable {
    case data(String)
    case completed
}

public indirect enum JSONValue: Codable, Sendable, Equatable {
    case string(String)
    case number(Double)
    case bool(Bool)
    case object([String: JSONValue])
    case array([JSONValue])
    case null

    public init(from decoder: Decoder) throws {
        let container = try decoder.singleValueContainer()
        if container.decodeNil() { self = .null }
        else if let value = try? container.decode(Bool.self) { self = .bool(value) }
        else if let value = try? container.decode(Double.self) { self = .number(value) }
        else if let value = try? container.decode(String.self) { self = .string(value) }
        else if let value = try? container.decode([String: JSONValue].self) { self = .object(value) }
        else if let value = try? container.decode([JSONValue].self) { self = .array(value) }
        else { throw DecodingError.dataCorruptedError(in: container, debugDescription: "Invalid JSON") }
    }

    public func encode(to encoder: Encoder) throws {
        var container = encoder.singleValueContainer()
        switch self {
        case .string(let value): try container.encode(value)
        case .number(let value): try container.encode(value)
        case .bool(let value): try container.encode(value)
        case .object(let value): try container.encode(value)
        case .array(let value): try container.encode(value)
        case .null: try container.encodeNil()
        }
    }
}

public struct RuntimeEvent: Codable, Sendable, Equatable {
    public let schemaVersion: Int
    public let eventID: String
    public let type: String
    public let timestamp: String
    public let payload: [String: JSONValue]

    enum CodingKeys: String, CodingKey {
        case schemaVersion = "schema_version"
        case eventID = "event_id"
        case type, timestamp, payload
    }
}

public extension RuntimeEvent {
    var nativeV2Tuning: NativeV2TuningState? {
        guard type == "runtime.native_v2_tuning",
              case .string(let statusValue)? = payload["status"],
              let status = NativeV2TuningStatus(rawValue: statusValue)
        else { return nil }
        let runID: Int
        if case .number(let value)? = payload["run_id"] {
            runID = Int(value)
        } else {
            runID = 0
        }
        let profileID: String?
        if case .string(let value)? = payload["profile_id"] { profileID = value }
        else { profileID = nil }
        let errorCode: String?
        if case .string(let value)? = payload["error_code"] { errorCode = value }
        else { errorCode = nil }
        let quarantinedProfiles: Int
        if case .number(let value)? = payload["quarantined_profiles"] {
            quarantinedProfiles = Int(value)
        } else {
            quarantinedProfiles = 0
        }
        let latestQuarantinedProfileID: String?
        if case .string(let value)? = payload["latest_quarantined_profile_id"] {
            latestQuarantinedProfileID = value
        } else {
            latestQuarantinedProfileID = nil
        }
        return NativeV2TuningState(
            enabled: status != .disabled,
            status: status,
            runID: runID,
            profileID: profileID,
            errorCode: errorCode,
            quarantinedProfiles: quarantinedProfiles,
            latestQuarantinedProfileID: latestQuarantinedProfileID
        )
    }

    var runtimeFailure: RuntimeFailure? {
        guard type == "runtime.failure",
              case .object(let failure)? = payload["failure"],
              case .number(let schemaVersion)? = failure["schema_version"],
              case .string(let code)? = failure["code"],
              case .string(let messageKey)? = failure["message_key"],
              case .string(let recoverabilityValue)? = failure["recoverability"],
              case .string(let detailFingerprint)? = failure["detail_fingerprint"],
              let recoverability = RuntimeRecoverability(rawValue: recoverabilityValue)
        else { return nil }
        return RuntimeFailure(
            schemaVersion: Int(schemaVersion),
            code: code,
            messageKey: messageKey,
            recoverability: recoverability,
            detailFingerprint: detailFingerprint
        )
    }

    var contextReevaluation: RuntimeContextReevaluation? {
        guard type == "runtime.context_reevaluation",
              case .string(let statusValue)? = payload["status"],
              let status = ContextReevaluationStatus(rawValue: statusValue),
              case .number(let configured)? = payload["configured_context_tokens"],
              case .number(let effective)? = payload["effective_context_tokens"]
        else { return nil }
        let capacity: Int?
        if case .number(let value)? = payload["capacity_context_tokens"] {
            capacity = Int(value)
        } else {
            capacity = nil
        }
        return RuntimeContextReevaluation(
            status: status,
            configuredContextTokens: Int(configured),
            effectiveContextTokens: Int(effective),
            capacityContextTokens: capacity
        )
    }
}

public enum NativeV2TuningControlAction: String, Codable, Sendable {
    case enable
    case disable
    case retry
}

public struct NativeV2TuningControlResult: Codable, Sendable, Equatable {
    public let accepted: Bool
    public let nativeV2Tuning: NativeV2TuningState
}
