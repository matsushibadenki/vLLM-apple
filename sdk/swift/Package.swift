// swift-tools-version: 6.0
import PackageDescription

let package = Package(
    name: "VLLMAppleKit",
    platforms: [.macOS(.v13)],
    products: [
        .library(name: "VLLMAppleKit", targets: ["VLLMAppleKit"]),
        .library(name: "VLLMAppleKitObjC", type: .dynamic, targets: ["VLLMAppleKitObjC"]),
        .executable(
            name: "VLLMAppleQualificationCheck",
            targets: ["VLLMAppleQualificationCheck"]
        ),
        .executable(
            name: "VLLMAppleModelE2E",
            targets: ["VLLMAppleModelE2E"]
        )
    ],
    targets: [
        .target(name: "VLLMAppleKit"),
        .target(name: "VLLMAppleKitObjC", dependencies: ["VLLMAppleKit"]),
        .executableTarget(
            name: "VLLMAppleQualificationCheck",
            dependencies: ["VLLMAppleKit"]
        ),
        .executableTarget(name: "VLLMAppleModelE2E", dependencies: ["VLLMAppleKit"]),
        .testTarget(name: "VLLMAppleKitTests", dependencies: ["VLLMAppleKit", "VLLMAppleKitObjC"],
                    resources: [.copy("Fixtures")])
    ]
)
