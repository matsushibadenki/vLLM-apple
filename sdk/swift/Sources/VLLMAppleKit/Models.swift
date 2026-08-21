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

