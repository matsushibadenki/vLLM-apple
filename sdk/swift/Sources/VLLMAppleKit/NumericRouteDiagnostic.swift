import Darwin
import Foundation

public enum NumericFormat: String, Codable, Sendable, CaseIterable {
    case nvfp4E2M1 = "nvfp4_e2m1"
    case mxfp4E2M1 = "mxfp4_e2m1"
    case mxfp6E2M3 = "mxfp6_e2m3"
    case mxfp8E4M3 = "mxfp8_e4m3"
    case fp8E4M3FN = "fp8_e4m3fn"
    case fp8E5M2 = "fp8_e5m2"
    case fp32, bf16, fp16, int8, uint8, int4, uint4, int2, uint2, nf4
}

public enum NumericTensorRole: String, Codable, Sendable, CaseIterable {
    case weight, activation, expert, vision, audio, diffusion
    case kvState = "kv_state"
    case recurrentState = "recurrent_state"
}

public enum NumericRouteStrategy: String, Codable, Sendable, CaseIterable {
    case loadConvert = "load_convert"
    case firstUseConvert = "first_use_convert"
    case cachedConvert = "cached_convert"
    case fusedEveryUse = "fused_every_use"
}

public enum NumericDiagnosticLanguage: String, Codable, Sendable {
    case english = "en"
    case japanese = "ja"
    case simplifiedChinese = "zh-Hans"
}

public struct NumericErrorBudget: Codable, Sendable, Equatable {
    public let maximumAbsoluteError: Double
    public let maximumRMSE: Double

    enum CodingKeys: String, CodingKey {
        case maximumAbsoluteError = "maximum_absolute_error"
        case maximumRMSE = "maximum_rmse"
    }

    fileprivate func validated() throws -> Self {
        guard maximumAbsoluteError.isFinite, maximumAbsoluteError >= 0,
              maximumRMSE.isFinite, maximumRMSE >= 0 else {
            throw NumericRouteDiagnosticError.invalidEvidence
        }
        return self
    }
}

public struct NumericRouteDiagnostic: Codable, Sendable, Equatable {
    public let schemaVersion: Int
    public let valid: Bool
    public let sourceFormat: NumericFormat
    public let runtimeFormat: NumericFormat
    public let computeFormat: NumericFormat
    public let tensorRole: NumericTensorRole
    public let route: NumericRouteStrategy
    public let errorBudget: NumericErrorBudget
    public let fallbackReason: String?
    public let messageKey: String
    public let message: String
    public let language: NumericDiagnosticLanguage

    enum CodingKeys: String, CodingKey {
        case valid, route, message, language
        case schemaVersion = "schema_version"
        case sourceFormat = "source_format"
        case runtimeFormat = "runtime_format"
        case computeFormat = "compute_format"
        case tensorRole = "tensor_role"
        case errorBudget = "error_budget"
        case fallbackReason = "fallback_reason"
        case messageKey = "message_key"
    }

    public func validated() throws -> Self {
        guard schemaVersion == 1, valid,
              messageKey == "numeric_route_ready_for_inspection",
              !message.isEmpty, message.count <= 256,
              fallbackReason.map({
                  !$0.isEmpty && $0.count <= 128 && !$0.unicodeScalars.contains(where: {
                      $0.value < 0x20
                  })
              }) ?? true else {
            throw NumericRouteDiagnosticError.invalidEvidence
        }
        _ = try errorBudget.validated()
        return self
    }

    public static func decodeValidated(_ data: Data) throws -> Self {
        guard !data.isEmpty, data.count <= 64 * 1024 else {
            throw NumericRouteDiagnosticError.invalidEvidence
        }
        return try JSONDecoder().decode(Self.self, from: data).validated()
    }
}

public enum NumericRouteDiagnosticError: Error, Sendable, Equatable {
    case invalidEvidence
    case unsafeFile

    public var messageKey: String {
        switch self {
        case .invalidEvidence: "numeric_route.error.invalid_evidence"
        case .unsafeFile: "numeric_route.error.unsafe_file"
        }
    }
}

public struct NumericRouteDiagnosticLoader: Sendable {
    public init() {}

    public func load(fileURL: URL) throws -> NumericRouteDiagnostic {
        let path = fileURL.path
        var fileInfo = stat()
        guard lstat(path, &fileInfo) == 0,
              (fileInfo.st_mode & S_IFMT) == S_IFREG,
              fileInfo.st_uid == getuid(),
              fileInfo.st_size > 0, fileInfo.st_size <= 64 * 1024 else {
            throw NumericRouteDiagnosticError.unsafeFile
        }
        let descriptor = open(path, O_RDONLY | O_NOFOLLOW)
        guard descriptor >= 0 else { throw NumericRouteDiagnosticError.unsafeFile }
        defer { close(descriptor) }
        var openedInfo = stat()
        guard fstat(descriptor, &openedInfo) == 0,
              openedInfo.st_ino == fileInfo.st_ino,
              openedInfo.st_size == fileInfo.st_size else {
            throw NumericRouteDiagnosticError.unsafeFile
        }
        var data = Data(count: Int(openedInfo.st_size))
        let complete = data.withUnsafeMutableBytes { buffer in
            var offset = 0
            while offset < buffer.count {
                guard let baseAddress = buffer.baseAddress else { return false }
                let count = read(descriptor, baseAddress.advanced(by: offset), buffer.count - offset)
                if count <= 0 { return false }
                offset += count
            }
            return true
        }
        guard complete else { throw NumericRouteDiagnosticError.unsafeFile }
        return try NumericRouteDiagnostic.decodeValidated(data)
    }
}
