import Foundation
import VLLMAppleKit

private let maximumObjectiveCPromptCharacters = 32 * 1024
private let maximumObjectiveCModelCharacters = 256

@objc(VLLMAppleHealthStatus)
public final class ObjectiveCHealthStatus: NSObject {
    @objc public let status: String
    @objc public let controlReady: Bool
    @objc public let inferenceReady: Bool
    @objc public let runtimeVersion: String

    init(_ value: HealthStatus) {
        status = value.status.rawValue
        controlReady = value.controlReady
        inferenceReady = value.inferenceReady
        runtimeVersion = value.runtimeVersion
    }
}

@objc(VLLMAppleChatResult)
public final class ObjectiveCChatResult: NSObject {
    @objc public let identifier: String
    @objc public let model: String?
    @objc public let content: String
    @objc public let finishReason: String?

    init(_ value: ChatResponse) throws {
        guard let choice = value.choices.first else {
            throw RuntimeClientError.invalidResponse
        }
        identifier = value.id
        model = value.model
        content = choice.message.content
        finishReason = choice.finishReason
    }
}

@objc(VLLMAppleObjectiveCClient)
public final class ObjectiveCRuntimeClient: NSObject, @unchecked Sendable {
    private let client: any VLLMAppleRuntimeClient

    @objc public init(baseURL: URL, sessionToken: String? = nil) {
        client = HTTPRuntimeClient(baseURL: baseURL, sessionToken: sessionToken)
        super.init()
    }

    @objc(healthWithCompletion:)
    public func health(
        completion: @escaping @Sendable (ObjectiveCHealthStatus?, NSError?) -> Void
    ) {
        Task {
            do {
                completion(ObjectiveCHealthStatus(try await client.health()), nil)
            } catch {
                completion(nil, objectiveCError(error))
            }
        }
    }

    @objc(chatWithModel:prompt:temperature:maxTokens:completion:)
    public func chat(
        model: String,
        prompt: String,
        temperature: NSNumber?,
        maxTokens: NSNumber?,
        completion: @escaping @Sendable (ObjectiveCChatResult?, NSError?) -> Void
    ) {
        guard !model.isEmpty, model.count <= maximumObjectiveCModelCharacters,
              !prompt.isEmpty, prompt.count <= maximumObjectiveCPromptCharacters,
              temperature.map({ $0.doubleValue.isFinite && (0 ... 2).contains($0.doubleValue) }) ?? true,
              maxTokens.map({ (1 ... 1_048_576).contains($0.intValue) }) ?? true else {
            completion(nil, objectiveCError(RuntimeClientError.invalidResponse))
            return
        }
        let request = ChatRequest(
            model: model,
            messages: [ChatMessage(role: "user", content: prompt)],
            temperature: temperature?.doubleValue,
            maxTokens: maxTokens?.intValue,
            stream: false
        )
        Task {
            do {
                completion(try ObjectiveCChatResult(await client.chat(request)), nil)
            } catch {
                completion(nil, objectiveCError(error))
            }
        }
    }
}

private func objectiveCError(_ error: Error) -> NSError {
    let key = (error as? RuntimeClientError)?.messageKey ?? "runtime.error.server"
    return NSError(
        domain: "VLLMAppleKit",
        code: 1,
        userInfo: [NSLocalizedDescriptionKey: key, "message_key": key]
    )
}
