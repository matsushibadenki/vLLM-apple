import AppKit
import Foundation

@MainActor
final class SandboxOptimizerModel: ObservableObject {
    @Published var modelURL: URL?
    @Published var outputURL: URL?
    @Published var endpoint = "http://127.0.0.1:8000/v1/optimizer"
    @Published var objective: OptimizationObjective = .balanced
    @Published var maximumMemoryGB = 16.0
    @Published var maximumDiskGB = 32.0
    @Published var maximumDurationMinutes = 60.0
    @Published var license = ""
    @Published var plan: OptimizationPlan?
    @Published var errorMessage: String?
    @Published var isRunning = false

    private let bookmarks = SecurityScopedBookmarkStore()

    init() {
        modelURL = bookmarks.restore(.model)
        outputURL = bookmarks.restore(.output)
    }

    func chooseModel() { chooseDirectory(selection: .model) }
    func chooseOutput() { chooseDirectory(selection: .output) }

    private func chooseDirectory(selection: SecurityScopedBookmarkStore.Selection) {
        let panel = NSOpenPanel()
        panel.canChooseDirectories = true
        panel.canChooseFiles = false
        panel.canCreateDirectories = selection == .output
        panel.allowsMultipleSelection = false
        guard panel.runModal() == .OK, let url = panel.url else { return }
        do {
            try bookmarks.save(url, for: selection)
            if selection == .model { modelURL = url } else { outputURL = url }
        } catch {
            errorMessage = error.localizedDescription
        }
    }

    func createPlan() {
        guard let modelURL, let outputURL, let endpointURL = URL(string: endpoint) else {
            errorMessage = String(localized: "error.selection")
            return
        }
        isRunning = true
        errorMessage = nil
        plan = nil
        let requestPayload: [String: Any] = [
            "model_path": modelURL.path,
            "output_path": outputURL.appendingPathComponent("optimized-model").path,
            "objective": objective.rawValue,
            "maximum_memory_bytes": Int64(maximumMemoryGB * 1_073_741_824),
            "maximum_disk_bytes": Int64(maximumDiskGB * 1_073_741_824),
            "maximum_duration_seconds": Int(maximumDurationMinutes * 60),
            "license": license.trimmingCharacters(in: .whitespacesAndNewlines).isEmpty
                ? NSNull() : license.trimmingCharacters(in: .whitespacesAndNewlines),
        ]
        Task {
            let modelAccess = modelURL.startAccessingSecurityScopedResource()
            let outputAccess = outputURL.startAccessingSecurityScopedResource()
            defer {
                if modelAccess { modelURL.stopAccessingSecurityScopedResource() }
                if outputAccess { outputURL.stopAccessingSecurityScopedResource() }
            }
            do {
                let payload = try JSONSerialization.data(withJSONObject: requestPayload)
                let transport = try OptimizerDaemonTransport(endpoint: endpointURL)
                let response = try await transport.send(
                    OptimizerDaemonRequest(operation: .plan, payload: payload)
                )
                guard response.succeeded, let payload = response.payload else {
                    throw OptimizerDaemonTransportError.invalidResponse
                }
                plan = try OptimizationPlan.decodeValidated(payload)
            } catch {
                errorMessage = error.localizedDescription
            }
            isRunning = false
        }
    }
}
