import Foundation
import XCTest
import VLLMAppleKitObjC

final class ObjectiveCRuntimeClientTests: XCTestCase {
    func testInvalidChatRequestFailsBeforeNetworkAccess() async throws {
        let client = ObjectiveCRuntimeClient(baseURL: try XCTUnwrap(URL(string: "http://127.0.0.1:1")))
        let result: (Bool, String?, String?) = await withCheckedContinuation { continuation in
            client.chat(
                model: "", prompt: "hello", temperature: nil, maxTokens: nil
            ) { response, error in
                continuation.resume(returning: (
                    response == nil,
                    error?.domain,
                    error?.userInfo["message_key"] as? String
                ))
            }
        }
        XCTAssertTrue(result.0)
        XCTAssertEqual(result.1, "VLLMAppleKit")
        XCTAssertEqual(result.2, "runtime.error.invalid_response")
    }
}
