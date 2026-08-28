// swift-tools-version: 6.0
import PackageDescription

let package = Package(
    name: "VLLMAppleKit",
    platforms: [.macOS(.v13)],
    products: [
        .library(name: "VLLMAppleKit", targets: ["VLLMAppleKit"]),
        .executable(
            name: "VLLMAppleQualificationCheck",
            targets: ["VLLMAppleQualificationCheck"]
        )
    ],
    targets: [
        .target(name: "VLLMAppleKit"),
        .executableTarget(
            name: "VLLMAppleQualificationCheck",
            dependencies: ["VLLMAppleKit"]
        ),
        .testTarget(name: "VLLMAppleKitTests", dependencies: ["VLLMAppleKit"])
    ]
)
