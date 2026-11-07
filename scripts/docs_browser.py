"""Render original Mermaid SVGs or verify the actual built static site in Chromium."""

from __future__ import annotations

import argparse
import hashlib
import json
import threading
import xml.etree.ElementTree as ET
from functools import partial
from http.server import SimpleHTTPRequestHandler, ThreadingHTTPServer
from importlib.metadata import version
from pathlib import Path
from typing import Any

from playwright.sync_api import sync_playwright

ROOT = Path(__file__).resolve().parents[1]


class QuietHandler(SimpleHTTPRequestHandler):
    def log_message(self, format: str, *args: Any) -> None:
        pass


def check(
    site: Path, output: Path, executable: str | None, mermaid_js: Path | None
) -> dict[str, Any]:
    output.mkdir(parents=True, exist_ok=False)
    with sync_playwright() as playwright:
        browser = playwright.chromium.launch(headless=True, executable_path=executable)
        try:
            page = browser.new_page(viewport={"width": 1280, "height": 900})
            diagrams = []
            if mermaid_js is not None:
                page.goto("about:blank")
                page.add_script_tag(path=str(mermaid_js))
                page.evaluate(
                    "mermaid.initialize({startOnLoad:false,securityLevel:'strict',htmlLabels:false,deterministicIds:true,deterministicIDSeed:'qvelt'})"
                )
                for source in sorted((ROOT / "docs/site/diagrams").glob("*.mmd")):
                    result = page.evaluate(
                        "async ({id,source}) => await mermaid.render(id,source)",
                        {"id": source.stem, "source": source.read_text()},
                    )
                    svg = result["svg"]
                    assert "<title" in svg and "<desc" in svg
                    ET.register_namespace("", "http://www.w3.org/2000/svg")
                    node = ET.fromstring(svg)
                    node.set("style", node.get("style", "") + ";background-color:white")
                    svg = ET.tostring(node, encoding="unicode")
                    destination = ROOT / "docs/site/assets" / (source.stem + ".svg")
                    destination.parent.mkdir(parents=True, exist_ok=True)
                    destination.write_text(svg)
                    diagrams.append(
                        {
                            "source": source.name,
                            "source_sha256": hashlib.sha256(
                                source.read_bytes()
                            ).hexdigest(),
                            "svg_sha256": hashlib.sha256(svg.encode()).hexdigest(),
                        }
                    )
            if not site.is_dir():
                assert mermaid_js is not None
                return {
                    "rendered_diagrams": diagrams,
                    "playwright": version("playwright"),
                    "browser": browser.version,
                }
            server = ThreadingHTTPServer(
                ("127.0.0.1", 0), partial(QuietHandler, directory=str(site.resolve()))
            )
            worker = threading.Thread(target=server.serve_forever, daemon=True)
            worker.start()
            errors: list[str] = []
            page.on("pageerror", lambda error: errors.append(str(error)))
            try:
                base = f"http://127.0.0.1:{server.server_address[1]}"
                for route in (
                    "",
                    "architecture/",
                    "tutorial/",
                    "status/",
                    "responsible-release/",
                ):
                    response = page.goto(base + "/" + route, wait_until="networkidle")
                    assert response is not None and response.status == 200
                    assert page.locator("h1").count() == 1
                    if route == "architecture/":
                        images = page.locator('img[src$=".svg"]')
                        assert images.count() == 2
                        for image in images.all():
                            assert image.get_attribute("alt") and image.evaluate(
                                "image => image.naturalWidth > 0"
                            )
                        page.screenshot(
                            path=str(output / "architecture-light.png"), full_page=True
                        )
                        page.emulate_media(color_scheme="dark")
                        page.screenshot(
                            path=str(output / "architecture-dark.png"), full_page=True
                        )
                page.set_viewport_size({"width": 390, "height": 844})
                page.goto(base + "/tutorial/", wait_until="networkidle")
                page.screenshot(
                    path=str(output / "tutorial-mobile.png"), full_page=True
                )
                assert not errors, errors
            finally:
                server.shutdown()
                server.server_close()
                worker.join(timeout=5)
            return {
                "pages_checked": 5,
                "browser_errors": errors,
                "rendered_diagrams": diagrams,
                "playwright": version("playwright"),
                "browser": browser.version,
                "scope": "actual local static site navigation,images and screenshots;not a screen-reader human audit",
            }
        finally:
            browser.close()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--site", type=Path, default=Path("site"))
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--browser-executable")
    parser.add_argument("--mermaid-js", type=Path)
    args = parser.parse_args()
    result = check(args.site, args.output, args.browser_executable, args.mermaid_js)
    (args.output / "result.json").write_text(
        json.dumps(result, sort_keys=True, indent=2) + "\n"
    )
    print(json.dumps(result, sort_keys=True))


if __name__ == "__main__":
    main()
