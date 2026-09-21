import SwiftUI

struct ContentView: View {
    @ObservedObject var model: OptimizerAppModel

    var body: some View {
        NavigationSplitView {
            Form {
                Section("section.input") {
                    pathRow("label.model", url: model.modelURL, action: model.chooseModel)
                    pathRow("label.output", url: model.outputURL, action: model.chooseOutput)
                }
                Section("section.priority") {
                    Picker("label.objective", selection: $model.objective) {
                        ForEach(OptimizationObjective.allCases) { value in
                            Text("objective.\(value.rawValue)").tag(value)
                        }
                    }
                    TextField("label.license", text: $model.license)
                }
                Section("section.budget") {
                    numberField("label.memory", value: $model.maximumMemoryGB, range: 1...1024)
                    numberField("label.disk", value: $model.maximumDiskGB, range: 1...8192)
                    numberField("label.duration", value: $model.maximumDurationMinutes, range: 1...1440)
                }
                Section {
                    if model.isRunning {
                        HStack {
                            if model.isExporting {
                                ProgressView(value: model.progress).frame(width: 80)
                                Text(localizedStage(model.stage))
                            } else {
                                ProgressView().controlSize(.small)
                                Text("status.planning")
                            }
                            Spacer()
                            if model.isExporting {
                                Button(model.isPaused ? "action.continue" : "action.pause") {
                                    model.isPaused ? model.resume() : model.pause()
                                }
                            }
                            Button("action.cancel", action: model.cancel)
                        }
                    } else {
                        Button("action.create_plan", action: model.createPlan)
                            .buttonStyle(.borderedProminent)
                            .disabled(model.modelURL == nil || model.outputURL == nil)
                    }
                }
            }
            .formStyle(.grouped)
            .navigationTitle("app.title")
            .padding(.horizontal, 16)
        } detail: {
            planView
                .padding(24)
                .frame(maxWidth: .infinity, maxHeight: .infinity, alignment: .topLeading)
        }
        .alert("error.title", isPresented: Binding(
            get: { model.errorMessage != nil },
            set: { if !$0 { model.errorMessage = nil } }
        )) {
            Button("action.ok") { model.errorMessage = nil }
        } message: {
            Text(model.errorMessage ?? "")
        }
        .confirmationDialog(
            "confirmation.title",
            isPresented: $model.pendingExecution,
            titleVisibility: .visible
        ) {
            Button("confirmation.execute", role: .destructive) { model.execute() }
            Button("action.cancel", role: .cancel) {}
        } message: {
            Text("confirmation.message")
        }
    }

    @ViewBuilder
    private var planView: some View {
        if let plan = model.plan {
            ScrollView {
                VStack(alignment: .leading, spacing: 18) {
                    Text("plan.title").font(.title2.bold())
                    LabeledContent("plan.identifier", value: plan.planID)
                    LabeledContent("label.objective", value: localizedObjective(plan.objective))
                    LabeledContent("plan.source_size", value: bytes(plan.source.weightsBytes))
                    if !plan.warnings.isEmpty {
                        GroupBox("plan.warnings") {
                            VStack(alignment: .leading) {
                                ForEach(plan.warnings, id: \.self) { Text("• \($0)") }
                            }.frame(maxWidth: .infinity, alignment: .leading)
                        }
                    }
                    Text("plan.candidates").font(.headline)
                    ForEach(plan.candidates) { candidate in
                        Button {
                            model.selectedCandidateID = candidate.id
                        } label: {
                            candidateView(candidate)
                        }
                        .buttonStyle(.plain)
                    }
                    HStack {
                        if model.canResume {
                            Button("action.resume") { model.execute(resume: true) }
                        }
                        Spacer()
                        Button("action.execute") { model.requestExecution() }
                            .buttonStyle(.borderedProminent)
                            .disabled(model.selectedCandidate?.executable != true || model.isRunning)
                    }
                    if let report = model.exportReport {
                        reportView(report)
                        qualityEvaluationView
                        generationEvaluationView
                    }
                    if model.exportReport == nil {
                        Label("plan.dry_run_notice", systemImage: "shield.checkered")
                            .foregroundStyle(.secondary)
                    }
                }
                .frame(maxWidth: .infinity, alignment: .leading)
            }
        } else {
            VStack(spacing: 12) {
                Image(systemName: "wand.and.stars")
                    .font(.system(size: 36))
                    .foregroundStyle(.secondary)
                Text("plan.empty.title").font(.title2.bold())
                Text("plan.empty.description")
                    .foregroundStyle(.secondary)
                    .multilineTextAlignment(.center)
            }
            .frame(maxWidth: .infinity, maxHeight: .infinity)
            .padding(.horizontal, 16)
        }
    }

    private func pathRow(_ title: LocalizedStringKey, url: URL?, action: @escaping () -> Void) -> some View {
        VStack(alignment: .leading, spacing: 6) {
            Text(title).font(.caption).foregroundStyle(.secondary)
            HStack {
                Text(url?.path ?? String(localized: "picker.none"))
                    .lineLimit(1).truncationMode(.middle)
                Spacer()
                Button("action.choose", action: action)
            }
        }
    }

    private func numberField(_ title: LocalizedStringKey, value: Binding<Double>, range: ClosedRange<Double>) -> some View {
        LabeledContent(title) {
            TextField("", value: value, format: .number.precision(.fractionLength(0...1)))
                .frame(width: 90)
                .multilineTextAlignment(.trailing)
                .onChange(of: value.wrappedValue) { newValue in
                    value.wrappedValue = min(max(newValue, range.lowerBound), range.upperBound)
                }
        }
    }

    private func candidateView(_ candidate: OptimizationCandidate) -> some View {
        GroupBox {
            Grid(alignment: .leading, horizontalSpacing: 20, verticalSpacing: 7) {
                GridRow { Text("candidate.output"); Text(bytes(candidate.estimatedOutputBytes)) }
                GridRow { Text("candidate.memory"); Text(bytes(candidate.estimatedPeakMemoryBytes)) }
                GridRow { Text("candidate.disk"); Text(bytes(candidate.requiredDiskBytes)) }
                GridRow { Text("candidate.duration"); Text(duration(candidate.estimatedDurationSeconds)) }
            }
            .frame(maxWidth: .infinity, alignment: .leading)
            if !candidate.blockingReasons.isEmpty {
                Text(candidate.blockingReasons.joined(separator: " · "))
                    .font(.caption).foregroundStyle(.orange)
            }
        } label: {
            HStack {
                Image(systemName: model.selectedCandidateID == candidate.id ? "largecircle.fill.circle" : "circle")
                Text("\(candidate.strategy) · \(candidate.targetWeightBits)-bit")
                Spacer()
                Image(systemName: candidate.withinBudget && candidate.executable ? "checkmark.circle.fill" : "exclamationmark.triangle.fill")
                    .foregroundStyle(candidate.withinBudget && candidate.executable ? .green : .orange)
            }
        }
    }

    private func reportView(_ report: ExportReport) -> some View {
        GroupBox("report.title") {
            VStack(alignment: .leading, spacing: 8) {
                LabeledContent("report.artifact", value: report.artifactManifest.artifactID)
                LabeledContent("report.output", value: bytes(report.workerResult.outputBytes))
                LabeledContent("report.peak_rss", value: bytes(report.workerResult.peakChildRSSBytes))
                LabeledContent("report.hash", value: report.artifactManifest.outputHash)
                LabeledContent("label.license", value: report.artifactManifest.license ?? String(localized: "value.unrecorded"))
                LabeledContent(
                    "report.evaluation",
                    value: report.artifactManifest.evaluation.isEmpty
                        ? String(localized: "value.not_evaluated")
                        : String(report.artifactManifest.evaluation.count)
                )
                ForEach(report.artifactManifest.transforms, id: \.type) { transform in
                    LabeledContent(
                        "report.transform",
                        value: "\(transform.type) · \(transform.backend) · \(transform.weightBits)-bit · group \(transform.groupSize)"
                    )
                }
                ForEach(report.artifactManifest.toolVersions.sorted(by: { $0.key < $1.key }), id: \.key) { tool in
                    LabeledContent("report.tool", value: "\(tool.key) \(tool.value)")
                }
                Text(report.manifestPath).font(.caption).foregroundStyle(.secondary)
                    .textSelection(.enabled)
            }
            .frame(maxWidth: .infinity, alignment: .leading)
        }
    }

    private var qualityEvaluationView: some View {
        GroupBox("quality.title") {
            VStack(alignment: .leading, spacing: 10) {
                pathRow("quality.dataset", url: model.datasetURL, action: model.chooseDataset)
                LabeledContent("quality.maximum_regression") {
                    TextField(
                        "",
                        value: $model.maximumRegressionPercent,
                        format: .number.precision(.fractionLength(0...2))
                    )
                    .frame(width: 90)
                    .multilineTextAlignment(.trailing)
                }
                HStack {
                    Text("quality.bounded_notice")
                        .font(.caption).foregroundStyle(.secondary)
                    Spacer()
                    Button("quality.compare") { model.compareQuality() }
                        .disabled(model.datasetURL == nil || model.isRunning)
                }
                if let comparison = model.comparisonReport {
                    Label(
                        comparison.approved ? "quality.approved" : "quality.rejected",
                        systemImage: comparison.approved ? "checkmark.seal.fill" : "xmark.octagon.fill"
                    )
                    .foregroundStyle(comparison.approved ? .green : .red)
                    ForEach(comparison.slices) { slice in
                        HStack {
                            Text("\(slice.language) · \(slice.domain)")
                            Spacer()
                            Text(slice.relativeRegression, format: .percent.precision(.fractionLength(2)))
                            Image(systemName: slice.passed ? "checkmark.circle.fill" : "xmark.circle.fill")
                                .foregroundStyle(slice.passed ? .green : .red)
                        }
                    }
                    Divider()
                    Text("quality.untested").font(.caption.bold())
                    Text(comparison.untestedCapabilities.joined(separator: " · "))
                        .font(.caption).foregroundStyle(.secondary)
                }
            }
            .frame(maxWidth: .infinity, alignment: .leading)
        }
    }

    private var generationEvaluationView: some View {
        GroupBox("generation.title") {
            VStack(alignment: .leading, spacing: 10) {
                pathRow(
                    "generation.dataset",
                    url: model.generationDatasetURL,
                    action: model.chooseGenerationDataset
                )
                LabeledContent("generation.minimum_agreement") {
                    TextField("", value: $model.minimumTokenAgreementPercent, format: .number)
                        .frame(width: 90).multilineTextAlignment(.trailing)
                }
                LabeledContent("generation.maximum_regression") {
                    TextField("", value: $model.maximumExpectationRegressionPercent, format: .number)
                        .frame(width: 90).multilineTextAlignment(.trailing)
                }
                HStack {
                    Text("generation.privacy_notice")
                        .font(.caption).foregroundStyle(.secondary)
                    Spacer()
                    Button("generation.compare") { model.compareGeneration() }
                        .disabled(model.generationDatasetURL == nil || model.isRunning)
                }
                if let comparison = model.generationComparison {
                    Label(
                        comparison.gate.approved ? "quality.approved" : "quality.rejected",
                        systemImage: comparison.gate.approved ? "checkmark.seal.fill" : "xmark.octagon.fill"
                    )
                    .foregroundStyle(comparison.gate.approved ? .green : .red)
                    Grid(alignment: .leading, horizontalSpacing: 16, verticalSpacing: 6) {
                        GridRow {
                            Text("generation.model")
                            Text("generation.elapsed")
                            Text("generation.peak_rss")
                        }.font(.caption.bold())
                        GridRow {
                            Text("generation.original")
                            Text(milliseconds(comparison.baseline.elapsedMilliseconds))
                            Text(bytes(comparison.baseline.peakRSSBytes))
                        }
                        GridRow {
                            Text("generation.optimized")
                            Text(milliseconds(comparison.candidate.elapsedMilliseconds))
                            Text(bytes(comparison.candidate.peakRSSBytes))
                        }
                    }
                    ForEach(comparison.gate.samples) { sample in
                        HStack {
                            Text("\(sample.sampleID) · \(sample.language) · \(sample.domain)")
                            Spacer()
                            Text(sample.tokenAgreement, format: .percent.precision(.fractionLength(1)))
                            Image(systemName: sample.passed ? "checkmark.circle.fill" : "xmark.circle.fill")
                                .foregroundStyle(sample.passed ? .green : .red)
                        }
                    }
                    if !comparison.gate.untestedCapabilities.isEmpty {
                        Text("quality.untested").font(.caption.bold())
                        Text(comparison.gate.untestedCapabilities.joined(separator: " · "))
                            .font(.caption).foregroundStyle(.secondary)
                    }
                }
            }
            .frame(maxWidth: .infinity, alignment: .leading)
        }
    }

    private func bytes(_ value: Int64) -> String { ByteCountFormatter.string(fromByteCount: value, countStyle: .file) }
    private func duration(_ seconds: Int?) -> String {
        guard let seconds else { return String(localized: "value.unknown") }
        return Duration.seconds(seconds).formatted(.units(allowed: [.hours, .minutes], width: .abbreviated))
    }
    private func milliseconds(_ value: Int64) -> String {
        Duration.milliseconds(value).formatted(.units(allowed: [.minutes, .seconds], width: .abbreviated))
    }
    private func localizedObjective(_ value: OptimizationObjective) -> String {
        String(localized: String.LocalizationValue("objective.\(value.rawValue)"))
    }
    private func localizedStage(_ stage: String) -> String {
        let supported = ["prepare", "convert", "resume", "validate", "promote"]
        let value = supported.contains(stage) ? stage : "prepare"
        return String(localized: String.LocalizationValue("stage.\(value)"))
    }
}
