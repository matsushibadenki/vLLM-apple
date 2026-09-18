import Foundation

private let maximumSchedulingCount = 2_147_483_647

public enum SchedulingPreference: String, Codable, Sendable, CaseIterable {
    case automatic
    case lowPower = "low_power"
    case highPerformance = "high_performance"
}

public struct SchedulingBackendAssignments: Decodable, Sendable, Equatable {
    public let cpu: Int
    public let vllmMetal: Int
    public let nativeMLX: Int
    public let nativeMetal: Int
    public let coreMLDraft: Int

    static let zero = SchedulingBackendAssignments(
        cpu: 0, vllmMetal: 0, nativeMLX: 0, nativeMetal: 0, coreMLDraft: 0
    )

    private init(cpu: Int, vllmMetal: Int, nativeMLX: Int, nativeMetal: Int, coreMLDraft: Int) {
        self.cpu = cpu
        self.vllmMetal = vllmMetal
        self.nativeMLX = nativeMLX
        self.nativeMetal = nativeMetal
        self.coreMLDraft = coreMLDraft
    }

    public init(from decoder: Decoder) throws {
        let values = try decoder.singleValueContainer().decode([String: Int].self)
        guard Set(values.keys) == Set([
            "cpu", "vllm_metal", "native_mlx", "native_metal", "coreml_draft"
        ]) else { throw DecodingError.dataCorruptedError(
            in: try decoder.singleValueContainer(), debugDescription: "Invalid backend keys"
        ) }
        cpu = values["cpu"]!
        vllmMetal = values["vllm_metal"]!
        nativeMLX = values["native_mlx"]!
        nativeMetal = values["native_metal"]!
        coreMLDraft = values["coreml_draft"]!
    }

    public var total: Int { cpu + vllmMetal + nativeMLX + nativeMetal + coreMLDraft }
    var valid: Bool {
        [cpu, vllmMetal, nativeMLX, nativeMetal, coreMLDraft].allSatisfy {
            (0...maximumSchedulingCount).contains($0)
        }
    }
}

public struct SchedulingQueueWaitBuckets: Decodable, Sendable, Equatable {
    public let under1ms: Int
    public let from1To10ms: Int
    public let from10To100ms: Int
    public let atLeast100ms: Int

    static let zero = SchedulingQueueWaitBuckets(
        under1ms: 0, from1To10ms: 0, from10To100ms: 0, atLeast100ms: 0
    )

    private init(under1ms: Int, from1To10ms: Int, from10To100ms: Int, atLeast100ms: Int) {
        self.under1ms = under1ms
        self.from1To10ms = from1To10ms
        self.from10To100ms = from10To100ms
        self.atLeast100ms = atLeast100ms
    }

    public init(from decoder: Decoder) throws {
        let values = try decoder.singleValueContainer().decode([String: Int].self)
        guard Set(values.keys) == Set([
            "under_1ms", "1_to_10ms", "10_to_100ms", "100ms_or_more"
        ]) else { throw DecodingError.dataCorruptedError(
            in: try decoder.singleValueContainer(), debugDescription: "Invalid wait bucket keys"
        ) }
        under1ms = values["under_1ms"]!
        from1To10ms = values["1_to_10ms"]!
        from10To100ms = values["10_to_100ms"]!
        atLeast100ms = values["100ms_or_more"]!
    }

    var valid: Bool {
        [under1ms, from1To10ms, from10To100ms, atLeast100ms].allSatisfy {
            (0...maximumSchedulingCount).contains($0)
        }
    }
}

public struct SchedulingAdaptiveTransitions: Decodable, Sendable, Equatable {
    public let applied: Int
    public let deferred: Int
    public let ignored: Int

    static let zero = SchedulingAdaptiveTransitions(applied: 0, deferred: 0, ignored: 0)

    private init(applied: Int, deferred: Int, ignored: Int) {
        self.applied = applied
        self.deferred = deferred
        self.ignored = ignored
    }

    public init(from decoder: Decoder) throws {
        let values = try decoder.singleValueContainer().decode([String: Int].self)
        guard Set(values.keys) == Set(["applied", "deferred", "ignored"]) else {
            throw DecodingError.dataCorruptedError(
                in: try decoder.singleValueContainer(), debugDescription: "Invalid transition keys"
            )
        }
        applied = values["applied"]!
        deferred = values["deferred"]!
        ignored = values["ignored"]!
    }

    var valid: Bool {
        [applied, deferred, ignored].allSatisfy { (0...maximumSchedulingCount).contains($0) }
    }
}

public struct SchedulingAdaptivePolicy: Decodable, Sendable, Equatable {
    public let level: Int
    public let maximumActiveRequests: Int
    public let maximumBatchSize: Int
    public let pressure: MemoryPressure
    public let thermal: ThermalState
    public let power: PowerMode
    public let preference: SchedulingPreference
    public let preferenceAvailable: Bool
    public let pendingLevel: Int?

    static let unknown = SchedulingAdaptivePolicy(
        level: 0, maximumActiveRequests: 1024, maximumBatchSize: 2_147_483_647,
        pressure: .unknown, thermal: .unknown, power: .unknown,
        preference: .automatic, preferenceAvailable: false, pendingLevel: nil
    )

    private init(
        level: Int, maximumActiveRequests: Int, maximumBatchSize: Int,
        pressure: MemoryPressure, thermal: ThermalState, power: PowerMode,
        preference: SchedulingPreference, preferenceAvailable: Bool, pendingLevel: Int?
    ) {
        self.level = level
        self.maximumActiveRequests = maximumActiveRequests
        self.maximumBatchSize = maximumBatchSize
        self.pressure = pressure
        self.thermal = thermal
        self.power = power
        self.preference = preference
        self.preferenceAvailable = preferenceAvailable
        self.pendingLevel = pendingLevel
    }

    private enum CodingKeys: String, CodingKey {
        case level, maximumActiveRequests, maximumBatchSize, pressure, thermal, power
        case preference, pendingLevel
    }

    public init(from decoder: Decoder) throws {
        let values = try decoder.container(keyedBy: CodingKeys.self)
        level = try values.decode(Int.self, forKey: .level)
        maximumActiveRequests = try values.decode(Int.self, forKey: .maximumActiveRequests)
        maximumBatchSize = try values.decode(Int.self, forKey: .maximumBatchSize)
        pressure = try values.decode(MemoryPressure.self, forKey: .pressure)
        thermal = try values.decode(ThermalState.self, forKey: .thermal)
        power = try values.decode(PowerMode.self, forKey: .power)
        preferenceAvailable = values.contains(.preference)
        preference = preferenceAvailable
            ? try values.decode(SchedulingPreference.self, forKey: .preference)
            : .automatic
        pendingLevel = try values.decodeIfPresent(Int.self, forKey: .pendingLevel)
    }

    var valid: Bool {
        (0...2).contains(level)
            && maximumActiveRequests == [1024, 2, 1][level]
            && maximumBatchSize == [2_147_483_647, 4, 1][level]
            && (pendingLevel == nil || (0..<level).contains(pendingLevel!))
    }
}

public struct SchedulingPreferenceControlResult: Decodable, Sendable {
    public let accepted: Bool
    public let transition: String
    public let schedulingObservability: SchedulingObservabilityState

    public var hasValidEvidence: Bool {
        accepted && ["applied", "deferred", "ignored"].contains(transition)
            && schedulingObservability.hasValidEvidence
            && schedulingObservability.adaptivePolicy.preferenceAvailable
    }
}

public struct SchedulingObservabilityState: Decodable, Sendable, Equatable {
    public let available: Bool
    public let assignments: SchedulingBackendAssignments
    public let queueWaitBuckets: SchedulingQueueWaitBuckets
    public let fallbackAttempts: Int
    public let fallbackExhausted: Int
    public let contentionRejections: Int
    public let steals: Int
    public let adaptiveTransitions: SchedulingAdaptiveTransitions
    public let adaptivePolicy: SchedulingAdaptivePolicy

    public static let unavailable = SchedulingObservabilityState(
        available: false,
        assignments: .zero, queueWaitBuckets: .zero,
        fallbackAttempts: 0, fallbackExhausted: 0,
        contentionRejections: 0, steals: 0,
        adaptiveTransitions: .zero, adaptivePolicy: .unknown
    )

    private enum CodingKeys: String, CodingKey {
        case assignments, queueWaitBuckets, fallbackAttempts, fallbackExhausted
        case contentionRejections, steals, adaptiveTransitions, adaptivePolicy
    }

    public init(from decoder: Decoder) throws {
        let container = try decoder.container(keyedBy: CodingKeys.self)
        available = true
        assignments = try container.decode(SchedulingBackendAssignments.self, forKey: .assignments)
        queueWaitBuckets = try container.decode(SchedulingQueueWaitBuckets.self, forKey: .queueWaitBuckets)
        fallbackAttempts = try container.decode(Int.self, forKey: .fallbackAttempts)
        fallbackExhausted = try container.decode(Int.self, forKey: .fallbackExhausted)
        contentionRejections = try container.decode(Int.self, forKey: .contentionRejections)
        steals = try container.decode(Int.self, forKey: .steals)
        adaptiveTransitions = try container.decode(SchedulingAdaptiveTransitions.self, forKey: .adaptiveTransitions)
        adaptivePolicy = try container.decode(SchedulingAdaptivePolicy.self, forKey: .adaptivePolicy)
    }

    private init(
        available: Bool,
        assignments: SchedulingBackendAssignments,
        queueWaitBuckets: SchedulingQueueWaitBuckets,
        fallbackAttempts: Int,
        fallbackExhausted: Int,
        contentionRejections: Int,
        steals: Int,
        adaptiveTransitions: SchedulingAdaptiveTransitions,
        adaptivePolicy: SchedulingAdaptivePolicy
    ) {
        self.available = available
        self.assignments = assignments
        self.queueWaitBuckets = queueWaitBuckets
        self.fallbackAttempts = fallbackAttempts
        self.fallbackExhausted = fallbackExhausted
        self.contentionRejections = contentionRejections
        self.steals = steals
        self.adaptiveTransitions = adaptiveTransitions
        self.adaptivePolicy = adaptivePolicy
    }

    public var hasValidEvidence: Bool {
        available && assignments.valid && queueWaitBuckets.valid
            && [fallbackAttempts, fallbackExhausted, contentionRejections, steals].allSatisfy {
                (0...maximumSchedulingCount).contains($0)
            }
            && adaptiveTransitions.valid && adaptivePolicy.valid
    }
}
