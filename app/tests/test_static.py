"""Static UI contract checks that do not require a browser runtime."""
from html.parser import HTMLParser
from pathlib import Path
import unittest


STATIC = Path(__file__).resolve().parents[1] / "static"


class StructureParser(HTMLParser):
    VOID = {"area", "base", "br", "col", "embed", "hr", "img", "input", "link", "meta", "param", "source", "track", "wbr"}

    def __init__(self):
        super().__init__()
        self.ids = set()
        self.stack = []

    def handle_starttag(self, tag, attrs):
        attributes = dict(attrs)
        if "id" in attributes:
            self.assert_unique(attributes["id"])
        if tag not in self.VOID:
            self.stack.append(tag)

    def handle_endtag(self, tag):
        if not self.stack or self.stack[-1] != tag:
            raise AssertionError(f"closing {tag}; expected {self.stack[-1] if self.stack else None}")
        self.stack.pop()

    def assert_unique(self, identifier):
        if identifier in self.ids:
            raise AssertionError(f"duplicate id: {identifier}")
        self.ids.add(identifier)


class StaticUiTests(unittest.TestCase):
    def test_html_is_balanced_and_has_technician_workflow(self):
        html = (STATIC / "index.html").read_text(encoding="utf-8")
        parser = StructureParser()
        parser.feed(html)
        self.assertEqual(parser.stack, [])
        self.assertTrue({
            "asset-id", "asset-location", "collected-at", "work-order", "file-input",
            "results", "history-content", "history-filter", "model-content",
        } <= parser.ids)
        self.assertNotIn("research scores", html.lower())
        self.assertIn("Peer comparisons are descriptive", html)

    def test_frontend_exposes_evidence_review_and_partial_failure_states(self):
        javascript = (STATIC / "app.js").read_text(encoding="utf-8")
        for phrase in (
            "comparisonCards", "outlier_high", "Compared only with this recording or batch",
            "review-status", "run.failures", "asset_id", "files: {door: []",
        ):
            self.assertIn(phrase, javascript)

    def test_styles_include_touch_keyboard_and_reduced_motion_support(self):
        css = (STATIC / "style.css").read_text(encoding="utf-8")
        self.assertIn("min-height: 44px", css)
        self.assertIn(":focus-visible", css)
        self.assertIn("prefers-reduced-motion", css)
        self.assertIn(".comparison.outlier_high", css)


if __name__ == "__main__":
    unittest.main()
