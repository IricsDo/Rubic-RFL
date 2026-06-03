from __future__ import annotations

import re
from html.parser import HTMLParser
from pathlib import Path


FRONTEND_ROOT = Path(__file__).resolve().parents[1]
REPO_ROOT = FRONTEND_ROOT.parent
INDEX_HTML = FRONTEND_ROOT / "index.html"
APP_JS = FRONTEND_ROOT / "src" / "app.js"
STYLES_CSS = FRONTEND_ROOT / "styles.css"
BACKEND_MAIN = REPO_ROOT / "backend" / "app" / "main.py"


class FrontendHtmlParser(HTMLParser):
    def __init__(self) -> None:
        super().__init__()
        self.elements_by_id: dict[str, dict[str, str | None]] = {}
        self.links: list[dict[str, str | None]] = []
        self.scripts: list[dict[str, str | None]] = []

    def handle_starttag(self, tag: str, attrs: list[tuple[str, str | None]]) -> None:
        attributes = dict(attrs)
        if "id" in attributes and attributes["id"]:
            attributes["tag"] = tag
            self.elements_by_id[attributes["id"]] = attributes
        if tag == "link":
            self.links.append(attributes)
        if tag == "script":
            self.scripts.append(attributes)


def parse_index() -> FrontendHtmlParser:
    parser = FrontendHtmlParser()
    parser.feed(INDEX_HTML.read_text(encoding="utf-8"))
    return parser


def normalized_asset_path(value: str) -> Path:
    return FRONTEND_ROOT / value.split("?", 1)[0].removeprefix("./")


def test_frontend_entrypoint_assets_exist() -> None:
    parser = parse_index()

    stylesheets = [
        link["href"]
        for link in parser.links
        if link.get("rel") == "stylesheet" and link.get("href")
    ]
    module_scripts = [
        script["src"]
        for script in parser.scripts
        if script.get("type") == "module" and script.get("src")
    ]

    assert any(asset.startswith("./styles.css?v=") for asset in stylesheets)
    assert any(asset.startswith("./src/app.js?v=") for asset in module_scripts)
    assert all(normalized_asset_path(asset).is_file() for asset in stylesheets)
    assert all(normalized_asset_path(asset).is_file() for asset in module_scripts)


def test_app_dom_selectors_have_matching_html_ids() -> None:
    parser = parse_index()
    app_source = APP_JS.read_text(encoding="utf-8")
    queried_ids = set(re.findall(r'document\.querySelector\("#([A-Za-z0-9_-]+)"\)', app_source))

    assert queried_ids, "app.js should declare DOM selectors for the static UI"
    assert queried_ids <= set(parser.elements_by_id), queried_ids - set(parser.elements_by_id)


def test_workspace_exposes_required_user_flows() -> None:
    parser = parse_index()
    required_ids = {
        "workspace",
        "cubeStage",
        "cube",
        "moveButtons",
        "languageEnButton",
        "languageVnButton",
        "apiBaseInput",
        "apiCheckButton",
        "depthInput",
        "depthDecreaseButton",
        "depthIncreaseButton",
        "seedInput",
        "scrambleButton",
        "resetButton",
        "classicalModeButton",
        "rlModeButton",
        "solveButton",
        "playButton",
        "stepBackButton",
        "stepForwardButton",
        "historyList",
        "solutionList",
        "rlDecisionPanel",
        "rlDecisionStats",
        "rlDecisionList",
        "rlSearchTraceList",
        "savedSessionsList",
        "saveReplayButton",
        "refreshSessionsButton",
        "stateJson",
        "exportReplayButton",
    }

    missing_ids = sorted(required_ids - set(parser.elements_by_id))
    assert not missing_ids
    assert parser.elements_by_id["apiBaseInput"].get("type") == "url"
    assert parser.elements_by_id["depthInput"].get("type") == "number"
    assert parser.elements_by_id["depthInput"].get("min") == "0"
    assert parser.elements_by_id["depthInput"].get("max") == "30"
    assert parser.elements_by_id["depthInput"].get("step") == "1"
    assert parser.elements_by_id["depthInput"].get("inputmode") == "numeric"
    assert "readonly" in parser.elements_by_id["depthInput"]
    assert parser.elements_by_id["depthInput"].get("tabindex") == "-1"
    assert parser.elements_by_id["depthDecreaseButton"].get("type") == "button"
    assert parser.elements_by_id["depthIncreaseButton"].get("type") == "button"
    assert parser.elements_by_id["classicalModeButton"].get("aria-pressed") == "true"
    assert parser.elements_by_id["rlModeButton"].get("aria-pressed") == "false"
    assert "hidden" in parser.elements_by_id["rlDecisionPanel"]


def test_frontend_backend_route_contracts_are_referenced() -> None:
    app_source = APP_JS.read_text(encoding="utf-8")
    backend_source = BACKEND_MAIN.read_text(encoding="utf-8")
    backend_routes = set(
        re.findall(r'@app\.(?:get|post|delete|websocket)\("([^"]+)"\)', backend_source)
    )

    route_contracts = {
        "/health": "/health",
        "/cube/validate": "/cube/validate",
        "/cube/scramble": "/cube/scramble",
        "/cube/apply-move": "/cube/apply-move",
        "/solve/classical": "/solve/classical",
        "/solve/rl/replay-package": "/solve/rl/replay-package",
        "/sessions": "/sessions",
        "/sessions/{session_id}": "/sessions/${encodeURIComponent(sessionId)}",
        "/ws/solve/{session_id}": "/ws/solve/",
    }

    missing_backend_routes = sorted(set(route_contracts) - backend_routes)
    assert not missing_backend_routes
    for frontend_reference in route_contracts.values():
        assert frontend_reference in app_source
    assert "new WebSocket" in app_source


def test_css_covers_responsive_replay_and_trace_surfaces() -> None:
    css_source = STYLES_CSS.read_text(encoding="utf-8")
    required_selectors = {
        ".cube-stage",
        ".progress-bar",
        ".panel-section",
        ".decision-list",
        ".search-trace-list",
        ".session-list",
        "@media (max-width: 760px)",
        "@media (prefers-reduced-motion: reduce)",
    }

    missing_selectors = sorted(
        selector for selector in required_selectors if selector not in css_source
    )
    assert not missing_selectors
