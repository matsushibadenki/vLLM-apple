import CoreML
import CoreVideo
import CryptoKit
import Darwin
import Foundation

func fail(_ message: String) -> Never {
    FileHandle.standardError.write(Data((message + "\n").utf8))
    exit(2)
}

func embeddingReport(_ values: MLMultiArray, latency: UInt64, index: Int) -> [String: Any] {
    guard values.count == 512 else { fail("invalid_embedding_shape") }
    var bytes = Data(capacity: values.count * 4)
    var squaredNorm = 0.0
    for offset in 0..<values.count {
        let value = values[offset].floatValue
        guard value.isFinite else { fail("nonfinite_embedding") }
        squaredNorm += Double(value * value)
        var bits = value.bitPattern.littleEndian
        withUnsafeBytes(of: &bits) { bytes.append(contentsOf: $0) }
    }
    let digest = SHA256.hash(data: bytes).map { String(format: "%02x", $0) }.joined()
    return [
        "sample_index": index,
        "latency_nanoseconds": latency,
        "output_count": values.count,
        "output_sha256": digest,
        "l2_norm": sqrt(squaredNorm),
    ]
}

func makeImage(index: Int) -> CVPixelBuffer {
    var buffer: CVPixelBuffer?
    let status = CVPixelBufferCreate(
        kCFAllocatorDefault,
        256,
        256,
        kCVPixelFormatType_32BGRA,
        [kCVPixelBufferIOSurfacePropertiesKey: [:]] as CFDictionary,
        &buffer
    )
    guard status == kCVReturnSuccess, let buffer else { fail("pixel_buffer_failed") }
    CVPixelBufferLockBaseAddress(buffer, [])
    defer { CVPixelBufferUnlockBaseAddress(buffer, []) }
    guard let base = CVPixelBufferGetBaseAddress(buffer) else { fail("pixel_buffer_address_failed") }
    let rowBytes = CVPixelBufferGetBytesPerRow(buffer)
    for y in 0..<256 {
        let row = base.advanced(by: y * rowBytes).assumingMemoryBound(to: UInt8.self)
        for x in 0..<256 {
            let offset = x * 4
            row[offset] = UInt8((x + index * 29) % 256)
            row[offset + 1] = UInt8((y * 3 + index * 47) % 256)
            row[offset + 2] = UInt8((x + y + index * 61) % 256)
            row[offset + 3] = 255
        }
    }
    return buffer
}

guard CommandLine.arguments.count == 3 else { fail("expected image and text model paths") }
let configuration = MLModelConfiguration()
configuration.computeUnits = .cpuAndNeuralEngine
guard let imageModel = try? MLModel(
    contentsOf: URL(fileURLWithPath: CommandLine.arguments[1]), configuration: configuration
), let textModel = try? MLModel(
    contentsOf: URL(fileURLWithPath: CommandLine.arguments[2]), configuration: configuration
) else { fail("model_load_failed") }

var imageSamples: [[String: Any]] = []
var textSamples: [[String: Any]] = []
for sampleIndex in 0..<3 {
    guard let imageProvider = try? MLDictionaryFeatureProvider(
        dictionary: ["image": MLFeatureValue(pixelBuffer: makeImage(index: sampleIndex))]
    ) else { fail("image_provider_failed") }
    var started = DispatchTime.now().uptimeNanoseconds
    guard let imageOutput = try? imageModel.prediction(from: imageProvider),
          let imageEmbedding = imageOutput.featureValue(for: "final_emb_1")?.multiArrayValue else {
        fail("image_prediction_failed")
    }
    imageSamples.append(embeddingReport(
        imageEmbedding,
        latency: DispatchTime.now().uptimeNanoseconds - started,
        index: sampleIndex
    ))

    guard let tokens = try? MLMultiArray(shape: [1, 77], dataType: .int32) else {
        fail("token_allocation_failed")
    }
    for offset in 0..<tokens.count {
        tokens[offset] = NSNumber(value: Int32((offset * 97 + sampleIndex * 211) % 49_000))
    }
    guard let textProvider = try? MLDictionaryFeatureProvider(
        dictionary: ["text": MLFeatureValue(multiArray: tokens)]
    ) else { fail("text_provider_failed") }
    started = DispatchTime.now().uptimeNanoseconds
    guard let textOutput = try? textModel.prediction(from: textProvider),
          let textEmbedding = textOutput.featureValue(for: "final_emb_1")?.multiArrayValue else {
        fail("text_prediction_failed")
    }
    textSamples.append(embeddingReport(
        textEmbedding,
        latency: DispatchTime.now().uptimeNanoseconds - started,
        index: sampleIndex
    ))
}

var usage = rusage()
getrusage(RUSAGE_SELF, &usage)
let thermal: String
switch ProcessInfo.processInfo.thermalState {
case .nominal: thermal = "nominal"
case .fair: thermal = "fair"
case .serious: thermal = "serious"
case .critical: thermal = "critical"
@unknown default: thermal = "unknown"
}
let imageDigests = Set(imageSamples.compactMap { $0["output_sha256"] as? String })
let textDigests = Set(textSamples.compactMap { $0["output_sha256"] as? String })
let payload: [String: Any] = [
    "schema_version": 1,
    "scope": "mobileclip_s0_coreml_embedding_qualification",
    "compute_units": "cpu_and_neural_engine",
    "image_input": ["name": "image", "shape": [256, 256, 3]],
    "text_input": ["name": "text", "shape": [1, 77], "dtype": "int32"],
    "output": ["name": "final_emb_1", "shape": [1, 512], "dtype": "float32"],
    "image_samples": imageSamples,
    "text_samples": textSamples,
    "sample_count": 3,
    "peak_rss_bytes": Int(usage.ru_maxrss),
    "thermal_state": thermal,
    "stores_image": false,
    "stores_tokens": false,
    "stores_embedding": false,
    "passed": imageDigests.count == 3 && textDigests.count == 3,
]
guard let encoded = try? JSONSerialization.data(withJSONObject: payload, options: [.sortedKeys]) else {
    fail("json_encoding_failed")
}
FileHandle.standardOutput.write(encoded)
FileHandle.standardOutput.write(Data("\n".utf8))
