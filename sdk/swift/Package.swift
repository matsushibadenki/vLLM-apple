// swift-tools-version: 6.0
import PackageDescription

let package = Package(
    name: "VLLMAppleKit",
    platforms: [.macOS(.v13)],
    products: [
        .library(name: "VLLMAppleKit", targets: ["VLLMAppleKit"])
    ],
    targets: [
        .target(name: "VLLMAppleKit"),
        .testTarget(name: "VLLMAppleKitTests", dependencies: ["VLLMAppleKit"])
    ]
)

