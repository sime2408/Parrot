// swift-tools-version: 5.10
// Live harness: compiles the app's engine sources (symlinked from the
// app's Parrot/ folder) without SwiftUI/SwiftData, so they can be built and measured
// with Command Line Tools only.
import PackageDescription

let package = Package(
    name: "LiveHarness",
    platforms: [.macOS(.v14)],
    dependencies: [
        .package(url: "https://github.com/argmaxinc/WhisperKit.git", exact: "0.18.0"),
        .package(url: "https://github.com/FluidInference/FluidAudio.git",
                 revision: "5390df9752c8fc583596018360c5fd70d6fa6c75"),
    ],
    targets: [
        .executableTarget(
            name: "LiveHarness",
            dependencies: [
                .product(name: "WhisperKit", package: "WhisperKit"),
                .product(name: "FluidAudio", package: "FluidAudio"),
            ],
            path: "Sources/LiveHarness"
        ),
    ]
)
