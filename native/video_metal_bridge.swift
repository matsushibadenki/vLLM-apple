import AVFoundation
import CoreVideo
import Foundation
import Metal
import VideoToolbox

struct BridgeError: Error, CustomStringConvertible {
    let description: String
    init(_ description: String) { self.description = description }
}

func fail(_ message: String) -> Never {
    FileHandle.standardError.write(Data((message + "\n").utf8))
    exit(1)
}

guard CommandLine.arguments.count == 3,
      let maximumFrames = Int(CommandLine.arguments[2]),
      (1...4096).contains(maximumFrames) else {
    fail("usage: video_metal_bridge INPUT MAXIMUM_FRAMES")
}

let input = URL(fileURLWithPath: CommandLine.arguments[1])
guard input.isFileURL,
      FileManager.default.fileExists(atPath: input.path),
      let attributes = try? FileManager.default.attributesOfItem(atPath: input.path),
      let size = attributes[.size] as? NSNumber,
      size.int64Value > 0,
      size.int64Value <= 512 * 1024 * 1024 else {
    fail("input file is missing or exceeds its byte budget")
}

do {
    let asset = AVURLAsset(url: input)
    guard let track = asset.tracks(withMediaType: .video).first,
          let format = track.formatDescriptions.first else {
        throw BridgeError("input has no video track")
    }
    let codec = CMFormatDescriptionGetMediaSubType(format as! CMFormatDescription)
    let hardwareSupported = VTIsHardwareDecodeSupported(codec)
    guard hardwareSupported else {
        throw BridgeError("hardware decode is unavailable for this codec")
    }
    guard let device = MTLCreateSystemDefaultDevice() else {
        throw BridgeError("Metal device is unavailable")
    }
    var cache: CVMetalTextureCache?
    guard CVMetalTextureCacheCreate(nil, nil, device, nil, &cache) == kCVReturnSuccess,
          let textureCache = cache else {
        throw BridgeError("CVMetalTextureCache creation failed")
    }
    let reader = try AVAssetReader(asset: asset)
    let settings: [String: Any] = [
        kCVPixelBufferPixelFormatTypeKey as String: kCVPixelFormatType_32BGRA,
        kCVPixelBufferMetalCompatibilityKey as String: true,
        kCVPixelBufferIOSurfacePropertiesKey as String: [:],
    ]
    let output = AVAssetReaderTrackOutput(track: track, outputSettings: settings)
    output.alwaysCopiesSampleData = false
    guard reader.canAdd(output) else {
        throw BridgeError("video track output cannot be added")
    }
    reader.add(output)
    guard reader.startReading() else {
        throw reader.error ?? BridgeError("video reader failed to start")
    }

    var decodedFrames = 0
    var textureBindings = 0
    var bindingFailures = 0
    var width = 0
    var height = 0
    while decodedFrames < maximumFrames, let sample = output.copyNextSampleBuffer() {
        autoreleasepool {
            decodedFrames += 1
            guard let pixelBuffer = CMSampleBufferGetImageBuffer(sample) else {
                bindingFailures += 1
                return
            }
            width = CVPixelBufferGetWidth(pixelBuffer)
            height = CVPixelBufferGetHeight(pixelBuffer)
            var wrapped: CVMetalTexture?
            let status = CVMetalTextureCacheCreateTextureFromImage(
                nil,
                textureCache,
                pixelBuffer,
                nil,
                .bgra8Unorm,
                width,
                height,
                0,
                &wrapped
            )
            guard status == kCVReturnSuccess,
                  let wrapped,
                  let texture = CVMetalTextureGetTexture(wrapped),
                  texture.width == width,
                  texture.height == height else {
                bindingFailures += 1
                return
            }
            textureBindings += 1
        }
    }
    reader.cancelReading()
    CVMetalTextureCacheFlush(textureCache, 0)
    guard decodedFrames > 0 else {
        throw BridgeError("decoder returned no frames")
    }
    let report: [String: Any] = [
        "schema_version": 1,
        "hardware_decode_supported": hardwareSupported,
        "decoded_frames": decodedFrames,
        "texture_bindings": textureBindings,
        "binding_failures": bindingFailures,
        "width": width,
        "height": height,
        "pixel_format": "bgra8Unorm",
        "always_copies_sample_data": false,
        "passed": bindingFailures == 0 && textureBindings == decodedFrames,
    ]
    let data = try JSONSerialization.data(withJSONObject: report, options: [.sortedKeys])
    FileHandle.standardOutput.write(data)
    FileHandle.standardOutput.write(Data("\n".utf8))
} catch {
    fail(String(describing: error))
}
