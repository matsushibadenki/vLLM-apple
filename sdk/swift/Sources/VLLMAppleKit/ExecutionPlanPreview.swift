import Foundation

public struct ExecutionPhasePreview: Decodable, Sendable, Equatable {
    public let phase: String
    public let backend: String
    public let batchSize: Int
    public let statePrecision: String
}

public struct ExecutionPlanPreview: Decodable, Sendable, Equatable {
    public let schemaVersion: Int
    public let planId: String
    public let modelId: String
    public let hardwareFingerprint: String
    public let contextTokens: Int
    public let memoryCeilingBytes: Int64
    public let estimatedPeakBytes: Int64
    public let prefill: ExecutionPhasePreview
    public let decode: ExecutionPhasePreview
    public let fallbackChain: [String]
    public let decisionReasons: [String]
    public let dryRun: Bool
}

public struct ExecutionPlanPreviewResult: Decodable, Sendable, Equatable {
    public let schemaVersion: Int
    public let available: Bool
    public let reason: String?
    public let plan: ExecutionPlanPreview?

    func validated() throws -> Self {
        guard schemaVersion == 1 else {
            throw RuntimeClientError.incompatibleSchema(received: schemaVersion, supported: 1)
        }
        if !available {
            guard plan == nil, let reason, !reason.isEmpty else {
                throw RuntimeClientError.invalidResponse
            }
            return self
        }
        guard reason == nil, let plan else { throw RuntimeClientError.invalidResponse }
        guard plan.schemaVersion == 1 else {
            throw RuntimeClientError.incompatibleSchema(received: plan.schemaVersion, supported: 1)
        }
        let backends: Set<String> = ["vllm_metal", "native_mlx", "native_metal", "coreml_draft", "cpu"]
        guard plan.dryRun, plan.planId.count == 24, !plan.modelId.isEmpty,
              !plan.hardwareFingerprint.isEmpty,
              plan.contextTokens >= 0, plan.memoryCeilingBytes >= 0,
              plan.estimatedPeakBytes >= 0, plan.estimatedPeakBytes <= plan.memoryCeilingBytes,
              plan.prefill.phase == "prefill", plan.decode.phase == "decode",
              plan.prefill.batchSize > 0, plan.decode.batchSize > 0,
              backends.contains(plan.prefill.backend), backends.contains(plan.decode.backend),
              plan.fallbackChain.allSatisfy({ backends.contains($0) }),
              !plan.prefill.statePrecision.isEmpty, !plan.decode.statePrecision.isEmpty,
              !plan.decisionReasons.isEmpty, plan.decisionReasons.allSatisfy({ !$0.isEmpty })
        else { throw RuntimeClientError.invalidResponse }
        return self
    }
}
