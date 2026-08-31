import CryptoKit
import Darwin
import Foundation
import Security

public enum QualificationCMSError: Error, Sendable, Equatable {
    case invalidInput
    case invalidSignature
    case untrustedSigner
    case signerIdentityMismatch
}

public struct QualificationCMSVerifier: Sendable {
    public init() {}

    public func verify(
        bundleURL: URL,
        signatureURL: URL,
        trustedCAURL: URL,
        expectedSignerSHA256: String
    ) throws {
        guard let bundle = boundedFile(bundleURL, maximumBytes: 65_536),
              let signature = boundedFile(signatureURL, maximumBytes: 1_048_576),
              let caData = boundedFile(trustedCAURL, maximumBytes: 1_048_576),
              let anchors = certificates(from: caData), !anchors.isEmpty,
              isSHA256(expectedSignerSHA256) else { throw QualificationCMSError.invalidInput }
        var decoder: CMSDecoder?
        guard CMSDecoderCreate(&decoder) == errSecSuccess, let decoder else {
            throw QualificationCMSError.invalidSignature
        }
        guard CMSDecoderSetDetachedContent(decoder, bundle as CFData) == errSecSuccess else {
            throw QualificationCMSError.invalidSignature
        }
        let update = signature.withUnsafeBytes { bytes in
            CMSDecoderUpdateMessage(decoder, bytes.baseAddress!, bytes.count)
        }
        guard update == errSecSuccess,
              CMSDecoderFinalizeMessage(decoder) == errSecSuccess else {
            throw QualificationCMSError.invalidSignature
        }
        let policy = SecPolicyCreateBasicX509()
        var signerStatus = CMSSignerStatus(rawValue: 0)!
        guard CMSDecoderCopySignerStatus(
            decoder, 0, policy, false, &signerStatus, nil, nil
        ) == errSecSuccess,
              signerStatus.rawValue == 1 else {
            throw QualificationCMSError.invalidSignature
        }
        var signer: SecCertificate?
        guard CMSDecoderCopySignerCert(decoder, 0, &signer) == errSecSuccess,
              let signer else { throw QualificationCMSError.invalidSignature }
        let signerDigest = SHA256.hash(data: SecCertificateCopyData(signer) as Data)
            .map { String(format: "%02x", $0) }.joined()
        guard signerDigest == expectedSignerSHA256.lowercased() else {
            throw QualificationCMSError.signerIdentityMismatch
        }
        var embedded: CFArray?
        guard CMSDecoderCopyAllCerts(decoder, &embedded) == errSecSuccess,
              let certificates = embedded as? [SecCertificate] else {
            throw QualificationCMSError.invalidSignature
        }
        var trust: SecTrust?
        guard SecTrustCreateWithCertificates(
            certificates as CFArray, policy, &trust
        ) == errSecSuccess, let trust,
              SecTrustSetAnchorCertificates(trust, anchors as CFArray) == errSecSuccess,
              SecTrustSetAnchorCertificatesOnly(trust, true) == errSecSuccess,
              SecTrustEvaluateWithError(trust, nil) else {
            throw QualificationCMSError.untrustedSigner
        }
    }

    private func certificates(from data: Data) -> [SecCertificate]? {
        if let certificate = SecCertificateCreateWithData(nil, data as CFData) {
            return [certificate]
        }
        guard let text = String(data: data, encoding: .utf8) else { return nil }
        let blocks = text.components(separatedBy: "-----BEGIN CERTIFICATE-----").dropFirst()
        let certificates = blocks.compactMap { block -> SecCertificate? in
            guard let encoded = block.components(separatedBy: "-----END CERTIFICATE-----").first else {
                return nil
            }
            let compact = encoded.components(separatedBy: .whitespacesAndNewlines).joined()
            guard let der = Data(base64Encoded: compact) else { return nil }
            return SecCertificateCreateWithData(nil, der as CFData)
        }
        return certificates.isEmpty ? nil : certificates
    }

    private func boundedFile(_ url: URL, maximumBytes: Int) -> Data? {
        let descriptor = Darwin.open(url.path, O_RDONLY | O_NOFOLLOW)
        guard descriptor >= 0 else { return nil }
        defer { Darwin.close(descriptor) }
        var before = stat()
        guard fstat(descriptor, &before) == 0,
              before.st_mode & S_IFMT == S_IFREG,
              before.st_size > 0,
              before.st_size <= off_t(maximumBytes) else { return nil }
        var data = Data()
        var buffer = [UInt8](repeating: 0, count: min(65_536, maximumBytes + 1))
        while true {
            let count = Darwin.read(descriptor, &buffer, buffer.count)
            guard count >= 0 else { return nil }
            if count == 0 { break }
            guard data.count + count <= maximumBytes else { return nil }
            data.append(buffer, count: count)
        }
        var after = stat()
        guard fstat(descriptor, &after) == 0,
              before.st_dev == after.st_dev,
              before.st_ino == after.st_ino,
              before.st_size == after.st_size,
              before.st_mtimespec.tv_sec == after.st_mtimespec.tv_sec,
              before.st_mtimespec.tv_nsec == after.st_mtimespec.tv_nsec,
              data.count == Int(before.st_size) else { return nil }
        return data
    }

    private func isSHA256(_ value: String) -> Bool {
        value.utf8.count == 64 && value.allSatisfy { $0.isHexDigit }
    }
}
