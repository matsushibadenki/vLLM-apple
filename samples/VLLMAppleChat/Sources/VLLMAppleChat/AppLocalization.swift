import Foundation

enum AppLocalization {
    static var bundle: Bundle {
        #if SWIFT_PACKAGE
        .module
        #else
        .main
        #endif
    }
}
