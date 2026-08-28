/* Hallmark · pre-emit critique: P5 H4 E4 S5 R5 V4
   genre: modern-minimal · macrostructure: Workbench · theme: Cobalt
   enrichment: none · navigation: native sidebar
   contrast: adaptive system colors · slop: pass (native macOS adaptation) */
import SwiftUI

enum DesignTokens {
    static let page = Color(nsColor: .windowBackgroundColor)
    static let surface = Color(nsColor: .controlBackgroundColor)
    static let ink = Color(nsColor: .labelColor)
    static let secondaryInk = Color(nsColor: .secondaryLabelColor)
    static let rule = Color(nsColor: .separatorColor)
    static let cobalt = Color(nsColor: .systemBlue)
    static let success = Color(nsColor: .systemGreen)
    static let warning = Color(nsColor: .systemOrange)
    static let danger = Color(nsColor: .systemRed)

    static let micro: CGFloat = 4
    static let compact: CGFloat = 8
    static let dense: CGFloat = 12
    static let standard: CGFloat = 16
    static let section: CGFloat = 24
    static let reading: CGFloat = 32
    static let spacious: CGFloat = 48
    static let controlRadius: CGFloat = 6

    static func display(size: CGFloat, weight: Font.Weight = .semibold) -> Font {
        .custom("Avenir Next", size: size).weight(weight)
    }
}
