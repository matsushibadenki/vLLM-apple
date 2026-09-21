// swift-tools-version: 6.0
import PackageDescription

let package = Package(
    name: "VLLMAppleOptimizer",
    defaultLocalization: "en",
    platforms: [.macOS(.v13)],
    products: [
        .executable(name: "VLLMAppleOptimizer", targets: ["VLLMAppleOptimizer"])
    ],
    targets: [
        .executableTarget(
            name: "VLLMAppleOptimizer",
            resources: [.process("Resources")]
        ),
        .testTarget(
            name: "VLLMAppleOptimizerTests",
            dependencies: ["VLLMAppleOptimizer"]
        )
    ]
)
