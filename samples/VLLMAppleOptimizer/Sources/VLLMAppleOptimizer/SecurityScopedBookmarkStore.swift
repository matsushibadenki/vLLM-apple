import Foundation

struct SecurityScopedBookmarkStore {
    enum Selection: String, CaseIterable {
        case model
        case output
        case perplexityDataset
        case generationDataset

        var defaultsKey: String { "vllm-apple.optimizer.bookmark.\(rawValue)" }
    }

    private let defaults: UserDefaults

    init(defaults: UserDefaults = .standard) {
        self.defaults = defaults
    }

    func save(_ url: URL, for selection: Selection) throws {
        let data = try url.bookmarkData(
            options: [.withSecurityScope],
            includingResourceValuesForKeys: [.isDirectoryKey, .fileSizeKey],
            relativeTo: nil
        )
        guard !data.isEmpty, data.count <= 1_048_576 else {
            throw BookmarkError.invalidBookmark
        }
        defaults.set(data, forKey: selection.defaultsKey)
    }

    func restore(_ selection: Selection) -> URL? {
        guard let data = defaults.data(forKey: selection.defaultsKey),
              !data.isEmpty, data.count <= 1_048_576 else {
            defaults.removeObject(forKey: selection.defaultsKey)
            return nil
        }
        var stale = false
        do {
            let url = try URL(
                resolvingBookmarkData: data,
                options: [.withSecurityScope, .withoutUI],
                relativeTo: nil,
                bookmarkDataIsStale: &stale
            )
            if stale {
                try save(url, for: selection)
            }
            return url
        } catch {
            defaults.removeObject(forKey: selection.defaultsKey)
            return nil
        }
    }

    func clear(_ selection: Selection) {
        defaults.removeObject(forKey: selection.defaultsKey)
    }
}

enum BookmarkError: LocalizedError {
    case invalidBookmark

    var errorDescription: String? {
        String(localized: "error.bookmark_invalid")
    }
}
