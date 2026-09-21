import SwiftUI

struct SandboxContentView: View {
    @ObservedObject var model: SandboxOptimizerModel

    var body: some View {
        ScrollView {
            VStack(alignment: .leading, spacing: 16) {
                Text("app.title").font(.largeTitle.bold())
                Text("app.subtitle").foregroundStyle(.secondary)
                TextField("field.endpoint", text: $model.endpoint)
                    .textFieldStyle(.roundedBorder)
                HStack {
                    Button("button.model") { model.chooseModel() }
                    Text(model.modelURL?.path ?? String(localized: "value.not_selected"))
                        .lineLimit(1).truncationMode(.middle)
                }
                HStack {
                    Button("button.output") { model.chooseOutput() }
                    Text(model.outputURL?.path ?? String(localized: "value.not_selected"))
                        .lineLimit(1).truncationMode(.middle)
                }
                Picker("field.objective", selection: $model.objective) {
                    ForEach(OptimizationObjective.allCases) { value in
                        Text(LocalizedStringKey("objective.\(value.rawValue)"))
                            .tag(value)
                    }
                }
                HStack {
                    TextField("field.memory", value: $model.maximumMemoryGB, format: .number)
                    TextField("field.disk", value: $model.maximumDiskGB, format: .number)
                    TextField(
                        "field.duration", value: $model.maximumDurationMinutes, format: .number
                    )
                }
                TextField("field.license", text: $model.license)
                    .textFieldStyle(.roundedBorder)
                Button("button.plan") { model.createPlan() }
                    .disabled(model.isRunning)
                if model.isRunning { ProgressView() }
                if let error = model.errorMessage {
                    Text(error).foregroundStyle(.red).textSelection(.enabled)
                }
                if let plan = model.plan {
                    Text("plan.title").font(.headline)
                    Text(plan.planID).font(.caption.monospaced()).textSelection(.enabled)
                    ForEach(plan.candidates) { candidate in
                        VStack(alignment: .leading, spacing: 4) {
                            Text("\(candidate.strategy) · INT\(candidate.targetWeightBits)")
                                .font(.headline)
                            Text(candidate.withinBudget ? "plan.within" : "plan.blocked")
                                .foregroundStyle(candidate.withinBudget ? .green : .orange)
                        }
                        .padding(12)
                        .background(.quaternary, in: RoundedRectangle(cornerRadius: 10))
                    }
                }
            }
            .padding(.horizontal, 20)
            .padding(.vertical, 16)
            .frame(maxWidth: .infinity, alignment: .leading)
        }
    }
}
