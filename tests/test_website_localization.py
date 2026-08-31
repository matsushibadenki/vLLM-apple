import re
import unittest
from html.parser import HTMLParser
from pathlib import Path
from urllib.parse import urlparse


WEBSITE = Path("website")
LOCALES = {
    "jp": "ja",
    "en": "en",
    "zh": "zh-CN",
}
ALTERNATES = {
    "ja": "../jp/",
    "en": "../en/",
    "zh-Hans": "../zh/",
    "x-default": "../en/",
}


class PageParser(HTMLParser):
    def __init__(self) -> None:
        super().__init__()
        self.html_lang: str | None = None
        self.ids: set[str] = set()
        self.links: list[dict[str, str]] = []
        self.scripts: list[str] = []
        self.stylesheets: list[str] = []
        self.images: list[tuple[str, str]] = []
        self.meta_description: str | None = None
        self.title_parts: list[str] = []
        self._in_title = False

    def handle_starttag(
        self,
        tag: str,
        attrs: list[tuple[str, str | None]],
    ) -> None:
        values = {key: value or "" for key, value in attrs}
        if tag == "html":
            self.html_lang = values.get("lang")
        if element_id := values.get("id"):
            self.ids.add(element_id)
        if tag == "meta" and values.get("name") == "description":
            self.meta_description = values.get("content")
        if tag == "link":
            self.links.append(values)
            if values.get("rel") == "stylesheet":
                self.stylesheets.append(values.get("href", ""))
        if tag == "script":
            self.scripts.append(values.get("src", ""))
        if tag == "img":
            self.images.append((values.get("src", ""), values.get("alt", "")))
        if tag == "title":
            self._in_title = True

    def handle_endtag(self, tag: str) -> None:
        if tag == "title":
            self._in_title = False

    def handle_data(self, data: str) -> None:
        if self._in_title:
            self.title_parts.append(data)

    @property
    def title(self) -> str:
        return "".join(self.title_parts).strip()


def parse_page(locale: str) -> tuple[Path, str, PageParser]:
    page = WEBSITE / locale / "index.html"
    source = page.read_text(encoding="utf-8")
    parser = PageParser()
    parser.feed(source)
    parser.close()
    return page, source, parser


def local_asset(page: Path, reference: str) -> Path | None:
    parsed = urlparse(reference)
    if parsed.scheme or parsed.netloc or reference.startswith("#"):
        return None
    return (page.parent / parsed.path).resolve()


class WebsiteLocalizationTests(unittest.TestCase):
    def test_pages_declare_language_metadata_and_alternates(self) -> None:
        for locale, expected_lang in LOCALES.items():
            with self.subTest(locale=locale):
                _, _, parser = parse_page(locale)
                self.assertEqual(parser.html_lang, expected_lang)
                self.assertTrue(parser.title.startswith("vLLM Apple — "))
                self.assertIsNotNone(parser.meta_description)
                self.assertGreaterEqual(len(parser.meta_description or ""), 50)
                alternates = {
                    link.get("hreflang"): link.get("href")
                    for link in parser.links
                    if link.get("rel") == "alternate"
                }
                self.assertEqual(alternates, ALTERNATES)

    def test_localized_pages_keep_the_same_navigation_contract(self) -> None:
        expected_ids = {"main", "site-nav", "top", "features", "how", "safety"}
        for locale in LOCALES:
            with self.subTest(locale=locale):
                _, source, parser = parse_page(locale)
                self.assertTrue(expected_ids.issubset(parser.ids))
                for marker in (
                    "data-header",
                    "data-menu-button",
                    "data-nav",
                    "data-model-select",
                    "data-model-label",
                    "data-memory-status",
                    "data-optimize",
                    "data-optimize-label",
                ):
                    self.assertIn(marker, source)

    def test_every_local_asset_reference_resolves_to_a_regular_file(self) -> None:
        for locale in LOCALES:
            page, _, parser = parse_page(locale)
            references = parser.scripts + parser.stylesheets + [src for src, _ in parser.images]
            for reference in references:
                with self.subTest(locale=locale, reference=reference):
                    asset = local_asset(page, reference)
                    self.assertIsNotNone(asset)
                    self.assertTrue(asset.is_file())
                    self.assertFalse(asset.is_symlink())
                    self.assertGreater(asset.stat().st_size, 0)
            for _, alt in parser.images:
                self.assertTrue(alt.strip())

    def test_english_page_and_runtime_states_do_not_regress_to_japanese(self) -> None:
        page, source, _ = parse_page("en")
        script = (page.parent / "assets/js/script.js").read_text(encoding="utf-8")
        kana = re.compile(r"[\u3040-\u30ff]")
        self.assertIsNone(kana.search(source))
        self.assertIsNone(kana.search(script))
        for text in ("Checking your Mac…", "Optimized", "Safe plan applied"):
            self.assertIn(text, script)

    def test_simplified_chinese_runtime_states_are_localized(self) -> None:
        page, source, _ = parse_page("zh")
        script = (page.parent / "assets/js/script.js").read_text(encoding="utf-8")
        for text in ("让你的 Mac", "前往 GitHub", "工作原理"):
            self.assertIn(text, source)
        for text in ("正在检查 Mac…", "优化完成", "已应用安全方案"):
            self.assertIn(text, script)


if __name__ == "__main__":
    unittest.main()
