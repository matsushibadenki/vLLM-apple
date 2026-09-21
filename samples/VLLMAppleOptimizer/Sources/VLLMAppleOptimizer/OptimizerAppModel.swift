import AppKit
import Combine
import Foundation
import UniformTypeIdentifiers

@MainActor
final class OptimizerAppModel: ObservableObject {
    @Published var modelURL: URL?
    @Published var outputURL: URL?
    @Published var objective: OptimizationObjective = .balanced
    @Published var maximumMemoryGB = 16.0
    @Published var maximumDiskGB = 32.0
    @Published var maximumDurationMinutes = 60.0
    @Published var license = ""
    @Published var plan: OptimizationPlan?
    @Published var errorMessage: String?
    @Published var isRunning = false
    @Published var selectedCandidateID: String?
    @Published var pendingExecution = false
    @Published var exportReport: ExportReport?
    @Published var canResume = false
    @Published var isExporting = false
    @Published var datasetURL: URL?
    @Published var maximumRegressionPercent = 2.0
    @Published var comparisonReport: QualityComparisonReport?
    @Published var progress = 0.0
    @Published var stage = "prepare"
    @Published var isPaused = false
    @Published var generationDatasetURL: URL?
    @Published var minimumTokenAgreementPercent = 100.0
    @Published var maximumExpectationRegressionPercent = 0.0
    @Published var generationComparison: GenerationComparison?

    private var cancellation: ProcessCancellation?
    private let bookmarkStore: SecurityScopedBookmarkStore

    init(bookmarkStore: SecurityScopedBookmarkStore = SecurityScopedBookmarkStore()) {
        self.bookmarkStore = bookmarkStore
        modelURL = bookmarkStore.restore(.model)
        outputURL = bookmarkStore.restore(.output)
        datasetURL = bookmarkStore.restore(.perplexityDataset)
        generationDatasetURL = bookmarkStore.restore(.generationDataset)
    }

    var optimizerExecutable: URL {
        if let configured = ProcessInfo.processInfo.environment["VLLM_APPLE_OPTIMIZER_EXECUTABLE"] {
            return URL(fileURLWithPath: configured)
        }
        for directory in (ProcessInfo.processInfo.environment["PATH"] ?? "").split(separator: ":") {
            let candidate = URL(fileURLWithPath: String(directory))
                .appendingPathComponent("vllm-apple-optimize")
            if FileManager.default.isExecutableFile(atPath: candidate.path) { return candidate }
        }
        let homebrew = URL(fileURLWithPath: "/opt/homebrew/bin/vllm-apple-optimize")
        if FileManager.default.isExecutableFile(atPath: homebrew.path) { return homebrew }
        return URL(fileURLWithPath: "/usr/local/bin/vllm-apple-optimize")
    }

    func chooseModel() {
        let panel = NSOpenPanel()
        panel.canChooseDirectories = true
        panel.canChooseFiles = false
        panel.allowsMultipleSelection = false
        panel.prompt = String(localized: "picker.model.confirm")
        if panel.runModal() == .OK { applySelection(panel.url, as: .model) }
    }

    func chooseOutput() {
        let panel = NSOpenPanel()
        panel.canChooseDirectories = true
        panel.canChooseFiles = false
        panel.canCreateDirectories = true
        panel.allowsMultipleSelection = false
        panel.prompt = String(localized: "picker.output.confirm")
        if panel.runModal() == .OK { applySelection(panel.url, as: .output) }
    }

    func chooseDataset() {
        let panel = NSOpenPanel()
        panel.canChooseDirectories = false
        panel.canChooseFiles = true
        panel.allowsMultipleSelection = false
        panel.allowedContentTypes = [UTType(filenameExtension: "jsonl") ?? .json, .json]
        panel.prompt = String(localized: "picker.dataset.confirm")
        if panel.runModal() == .OK { applySelection(panel.url, as: .perplexityDataset) }
    }

    func chooseGenerationDataset() {
        let panel = NSOpenPanel()
        panel.canChooseDirectories = false
        panel.canChooseFiles = true
        panel.allowsMultipleSelection = false
        panel.allowedContentTypes = [UTType(filenameExtension: "jsonl") ?? .json, .json]
        panel.prompt = String(localized: "picker.generation_dataset.confirm")
        if panel.runModal() == .OK { applySelection(panel.url, as: .generationDataset) }
    }

    private func applySelection(
        _ url: URL?, as selection: SecurityScopedBookmarkStore.Selection
    ) {
        guard let url else { return }
        do {
            try bookmarkStore.save(url, for: selection)
            switch selection {
            case .model: modelURL = url
            case .output: outputURL = url
            case .perplexityDataset: datasetURL = url
            case .generationDataset: generationDatasetURL = url
            }
        } catch {
            errorMessage = error.localizedDescription
        }
    }

    func cancel() { cancellation?.cancel() }
    func pause() { cancellation?.pause() }
    func resume() { cancellation?.resume() }

    func createPlan() {
        guard let modelURL, let outputURL else {
            errorMessage = String(localized: "error.selection_required")
            return
        }
        errorMessage = nil
        plan = nil
        isRunning = true
        isExporting = false
        let cancellation = ProcessCancellation()
        self.cancellation = cancellation
        let executable = optimizerExecutable
        let objective = objective.rawValue
        let memory = maximumMemoryGB
        let disk = maximumDiskGB
        let duration = Int(maximumDurationMinutes * 60)
        let license = license.trimmingCharacters(in: .whitespacesAndNewlines)

        Task {
            let accessModel = modelURL.startAccessingSecurityScopedResource()
            let accessOutput = outputURL.startAccessingSecurityScopedResource()
            defer {
                if accessModel { modelURL.stopAccessingSecurityScopedResource() }
                if accessOutput { outputURL.stopAccessingSecurityScopedResource() }
            }
            do {
                var arguments = [
                    "plan", modelURL.path,
                    "--output", outputURL.appendingPathComponent("optimized-model").path,
                    "--objective", objective,
                    "--max-memory-gb", String(memory),
                    "--max-disk-gb", String(disk),
                    "--max-duration-seconds", String(duration)
                ]
                if !license.isEmpty { arguments += ["--license", license] }
                let result = try await Task.detached {
                    try BoundedProcessRunner.run(
                        executable: executable,
                        arguments: arguments,
                        timeout: 120,
                        cancellation: cancellation
                    )
                }.value
                guard result.status == 0 else {
                    let detail = String(data: result.standardError, encoding: .utf8) ?? ""
                    throw OptimizerUIError.commandFailed(detail)
                }
                plan = try OptimizationPlan.decodeValidated(result.standardOutput)
                selectedCandidateID = plan?.candidates.first(where: { $0.executable && $0.withinBudget })?.id
            } catch {
                errorMessage = error.localizedDescription
            }
            isRunning = false
            self.cancellation = nil
        }
    }

    var selectedCandidate: OptimizationCandidate? {
        plan?.candidates.first { $0.id == selectedCandidateID }
    }

    func requestExecution() {
        guard selectedCandidate?.executable == true, selectedCandidate?.withinBudget == true else {
            errorMessage = String(localized: "error.no_executable_candidate")
            return
        }
        pendingExecution = true
    }

    func execute(resume: Bool = false) {
        guard let plan, let candidate = selectedCandidate,
              candidate.executable, candidate.withinBudget,
              let modelURL, let outputURL else {
            errorMessage = String(localized: "error.no_executable_candidate")
            return
        }
        pendingExecution = false
        errorMessage = nil
        isRunning = true
        isExporting = true
        progress = 0
        stage = "prepare"
        isPaused = false
        exportReport = nil
        let cancellation = ProcessCancellation()
        self.cancellation = cancellation
        let executable = optimizerExecutable
        let processTimeout = maximumDurationMinutes * 60 + 10
        let checkpointURL = outputURL.appendingPathComponent(".vllm-apple-optimizer-checkpoints")
        do {
            try FileManager.default.createDirectory(
                at: checkpointURL,
                withIntermediateDirectories: true,
                attributes: [.posixPermissions: 0o700]
            )
        } catch {
            errorMessage = error.localizedDescription
            isRunning = false
            isExporting = false
            return
        }
        let eventURL = checkpointURL.appendingPathComponent("events-\(UUID().uuidString).jsonl")
        var arguments = [
            "export", modelURL.path,
            "--output", plan.outputPath,
            "--checkpoint-root", checkpointURL.path,
            "--plan-id", plan.planID,
            "--bits", String(candidate.targetWeightBits),
            "--group-size", "64",
            "--max-output-gb", String(Double(candidate.requiredDiskBytes) / 1_073_741_824),
            "--timeout-seconds", String(maximumDurationMinutes * 60),
            "--event-output", eventURL.path,
            "--execute"
        ]
        if !license.trimmingCharacters(in: .whitespacesAndNewlines).isEmpty {
            arguments += ["--license", license.trimmingCharacters(in: .whitespacesAndNewlines)]
        }
        if resume { arguments.append("--resume") }
        let executionArguments = arguments

        Task {
            let accessModel = modelURL.startAccessingSecurityScopedResource()
            let accessOutput = outputURL.startAccessingSecurityScopedResource()
            defer {
                if accessModel { modelURL.stopAccessingSecurityScopedResource() }
                if accessOutput { outputURL.stopAccessingSecurityScopedResource() }
            }
            do {
                let monitor = Task { await monitorEvents(at: eventURL, planID: plan.planID) }
                defer { monitor.cancel() }
                let result = try await Task.detached {
                    try BoundedProcessRunner.run(
                        executable: executable,
                        arguments: executionArguments,
                        timeout: processTimeout,
                        cancellation: cancellation
                    )
                }.value
                guard result.status == 0 else {
                    canResume = true
                    let detail = String(data: result.standardError, encoding: .utf8) ?? ""
                    throw OptimizerUIError.commandFailed(detail)
                }
                exportReport = try ExportReport.decodeValidated(result.standardOutput)
                progress = 1
                stage = "promote"
                canResume = false
            } catch {
                canResume = true
                errorMessage = error.localizedDescription
            }
            isRunning = false
            isExporting = false
            self.cancellation = nil
        }
    }

    func compareQuality() {
        guard let plan, let modelURL, let datasetURL, exportReport != nil else {
            errorMessage = String(localized: "error.evaluation_required")
            return
        }
        errorMessage = nil
        comparisonReport = nil
        isRunning = true
        isExporting = false
        let cancellation = ProcessCancellation()
        self.cancellation = cancellation
        let executable = optimizerExecutable
        let outputModel = plan.outputPath
        guard maximumRegressionPercent.isFinite,
              0...100 ~= maximumRegressionPercent else {
            errorMessage = String(localized: "error.regression_range")
            isRunning = false
            return
        }
        let maximumRegression = maximumRegressionPercent / 100
        let evaluationRoot = URL(fileURLWithPath: plan.outputPath)
            .deletingLastPathComponent()
            .appendingPathComponent(".vllm-apple-evaluations")
            .appendingPathComponent("\(plan.planID)-\(UUID().uuidString)")
        do {
            try FileManager.default.createDirectory(
                at: evaluationRoot,
                withIntermediateDirectories: true,
                attributes: [.posixPermissions: 0o700]
            )
        } catch {
            errorMessage = error.localizedDescription
            isRunning = false
            return
        }
        let baselinePath = evaluationRoot.appendingPathComponent("baseline.json")
        let candidatePath = evaluationRoot.appendingPathComponent("candidate.json")
        let gatePath = evaluationRoot.appendingPathComponent("quality-gate.json")
        let baselineArguments = [
            "evaluate", modelURL.path, "--dataset", datasetURL.path,
            "--output", baselinePath.path, "--max-samples", "256",
            "--max-tokens-per-sample", "512", "--max-total-tokens", "131072"
        ]
        let candidateArguments = [
            "evaluate", outputModel, "--dataset", datasetURL.path,
            "--output", candidatePath.path, "--max-samples", "256",
            "--max-tokens-per-sample", "512", "--max-total-tokens", "131072"
        ]
        let gateArguments = [
            "quality-gate", "--baseline", baselinePath.path,
            "--candidate", candidatePath.path,
            "--max-perplexity-regression", String(maximumRegression),
            "--output", gatePath.path
        ]
        Task {
            let accessModel = modelURL.startAccessingSecurityScopedResource()
            let accessDataset = datasetURL.startAccessingSecurityScopedResource()
            defer {
                if accessModel { modelURL.stopAccessingSecurityScopedResource() }
                if accessDataset { datasetURL.stopAccessingSecurityScopedResource() }
            }
            do {
                for arguments in [baselineArguments, candidateArguments] {
                    let result = try await Task.detached {
                        try BoundedProcessRunner.run(
                            executable: executable,
                            arguments: arguments,
                            timeout: 3600,
                            cancellation: cancellation
                        )
                    }.value
                    guard result.status == 0 else {
                        throw OptimizerUIError.commandFailed(
                            String(data: result.standardError, encoding: .utf8) ?? ""
                        )
                    }
                }
                let gate = try await Task.detached {
                    try BoundedProcessRunner.run(
                        executable: executable,
                        arguments: gateArguments,
                        timeout: 120,
                        cancellation: cancellation
                    )
                }.value
                guard gate.status == 0 || gate.status == 1 else {
                    throw OptimizerUIError.commandFailed(
                        String(data: gate.standardError, encoding: .utf8) ?? ""
                    )
                }
                comparisonReport = try QualityComparisonReport.decodeValidated(gate.standardOutput)
            } catch {
                errorMessage = error.localizedDescription
            }
            isRunning = false
            self.cancellation = nil
        }
    }

    func compareGeneration() {
        guard let plan, let modelURL, let generationDatasetURL, exportReport != nil else {
            errorMessage = String(localized: "error.generation_evaluation_required")
            return
        }
        guard minimumTokenAgreementPercent.isFinite,
              maximumExpectationRegressionPercent.isFinite,
              0...100 ~= minimumTokenAgreementPercent,
              0...100 ~= maximumExpectationRegressionPercent else {
            errorMessage = String(localized: "error.percentage_range")
            return
        }
        errorMessage = nil
        generationComparison = nil
        isRunning = true
        isExporting = false
        let cancellation = ProcessCancellation()
        self.cancellation = cancellation
        let executable = optimizerExecutable
        let root = URL(fileURLWithPath: plan.outputPath).deletingLastPathComponent()
            .appendingPathComponent(".vllm-apple-evaluations")
            .appendingPathComponent("generation-\(plan.planID)-\(UUID().uuidString)")
        do {
            try FileManager.default.createDirectory(
                at: root,
                withIntermediateDirectories: true,
                attributes: [.posixPermissions: 0o700]
            )
        } catch {
            errorMessage = error.localizedDescription
            isRunning = false
            return
        }
        let baselinePath = root.appendingPathComponent("baseline.json")
        let candidatePath = root.appendingPathComponent("candidate.json")
        let gatePath = root.appendingPathComponent("generation-gate.json")
        func evaluateArguments(model: String, output: URL) -> [String] {
            [
                "generate-evaluate", model, "--dataset", generationDatasetURL.path,
                "--output", output.path, "--max-samples", "32",
                "--max-prompt-tokens", "16384", "--max-new-tokens", "32",
                "--chat-template"
            ]
        }
        let commands = [
            evaluateArguments(model: modelURL.path, output: baselinePath),
            evaluateArguments(model: plan.outputPath, output: candidatePath)
        ]
        let gateArguments = [
            "generation-quality-gate", "--baseline", baselinePath.path,
            "--candidate", candidatePath.path,
            "--min-token-agreement", String(minimumTokenAgreementPercent / 100),
            "--max-expectation-regression", String(maximumExpectationRegressionPercent / 100),
            "--output", gatePath.path
        ]
        Task {
            let accessModel = modelURL.startAccessingSecurityScopedResource()
            let accessDataset = generationDatasetURL.startAccessingSecurityScopedResource()
            defer {
                if accessModel { modelURL.stopAccessingSecurityScopedResource() }
                if accessDataset { generationDatasetURL.stopAccessingSecurityScopedResource() }
            }
            do {
                var evaluations: [GenerationEvaluationSummary] = []
                for arguments in commands {
                    let result = try await Task.detached {
                        try BoundedProcessRunner.run(
                            executable: executable,
                            arguments: arguments,
                            timeout: 3600,
                            cancellation: cancellation
                        )
                    }.value
                    guard result.status == 0 else {
                        throw OptimizerUIError.commandFailed(
                            String(data: result.standardError, encoding: .utf8) ?? ""
                        )
                    }
                    evaluations.append(try JSONDecoder().decode(
                        GenerationEvaluationSummary.self,
                        from: result.standardOutput
                    ))
                }
                let result = try await Task.detached {
                    try BoundedProcessRunner.run(
                        executable: executable,
                        arguments: gateArguments,
                        timeout: 120,
                        cancellation: cancellation
                    )
                }.value
                guard result.status == 0 || result.status == 1, evaluations.count == 2 else {
                    throw OptimizerUIError.commandFailed(
                        String(data: result.standardError, encoding: .utf8) ?? ""
                    )
                }
                let gate = try GenerationComparisonReport.decodeValidated(result.standardOutput)
                guard evaluations[0].schemaVersion == 1,
                      evaluations[0].datasetFingerprint == evaluations[1].datasetFingerprint else {
                    throw PlanValidationError.unsupportedContract
                }
                generationComparison = GenerationComparison(
                    baseline: evaluations[0], candidate: evaluations[1], gate: gate
                )
            } catch {
                errorMessage = error.localizedDescription
            }
            isRunning = false
            self.cancellation = nil
        }
    }

    private func monitorEvents(at url: URL, planID: String) async {
        while !Task.isCancelled {
            if let data = try? Data(contentsOf: url), data.count <= 1_048_576,
               let line = data.split(separator: 0x0A).last,
               let event = try? JSONDecoder().decode(OptimizerEventSummary.self, from: Data(line)),
               event.schemaVersion == 1, event.planID == planID,
               event.progress.isFinite, 0...1 ~= event.progress {
                progress = event.progress
                stage = event.stage
                isPaused = event.state == "paused"
            }
            try? await Task.sleep(for: .milliseconds(250))
        }
    }
}

enum OptimizerUIError: LocalizedError {
    case commandFailed(String)

    var errorDescription: String? {
        switch self {
        case let .commandFailed(detail):
            detail.isEmpty ? "Optimizer command failed." : detail
        }
    }
}
