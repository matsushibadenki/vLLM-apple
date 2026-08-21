// swift-tools-version: 6.0
import PackageDescription

let package = Package(
    name: "VLLMAppleChat",
    defaultLocalization: "en",
    platforms: [.macOS(.v13)],
    products: [
        .executable(name: "VLLMAppleChat", targets: ["VLLMAppleChat"])
    ],
    dependencies: [
        .package(path: "../../sdk/swift")
    ],
    targets: [
        .executableTarget(
            name: "VLLMAppleChat",
            dependencies: [
                .product(name: "VLLMAppleKit", package: "swift")
            ],
            resources: [.process("Resources")]
        )
    ]
)
