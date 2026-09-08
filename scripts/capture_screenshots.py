#!/usr/bin/env python3
"""Capture the README screenshots from the running system.

Every image in `docs/screenshots/` is a photograph of this system serving real
data. None is a mockup, and none is retouched. Regenerating them is one command,
which is the only way a screenshot in a README stays true.

Why this exists rather than `msedge --screenshot`: headless Chrome's
`--screenshot` fires when the page load event settles, and the two most
interesting surfaces here finish well after that. The copilot streams its answer
over Server-Sent Events, so a load-time snapshot catches "Classifying intent…"
forever. This drives the same browser over the DevTools protocol instead, waits
for a condition that means *the content is actually on screen*, and only then
captures.

    python scripts/capture_screenshots.py                # all surfaces
    python scripts/capture_screenshots.py copilot field  # just these
    python scripts/capture_screenshots.py --keep-open    # leave the browser up

Requires the stack to be running and a Chromium-family browser installed.
"""

from __future__ import annotations

import argparse
import base64
import json
import shutil
import socket
import subprocess
import time
import urllib.parse
import urllib.request
from dataclasses import dataclass, field
from pathlib import Path

from websockets.sync.client import connect

REPO_ROOT = Path(__file__).resolve().parents[1]
OUT_DIR = REPO_ROOT / "docs" / "screenshots"
BASE = "http://localhost:8000/ui"

BROWSER_CANDIDATES = [
    r"C:\Program Files (x86)\Microsoft\Edge\Application\msedge.exe",
    r"C:\Program Files\Microsoft\Edge\Application\msedge.exe",
    r"C:\Program Files\Google\Chrome\Application\chrome.exe",
    r"C:\Program Files (x86)\Google\Chrome\Application\chrome.exe",
    "google-chrome",
    "chromium",
    "chromium-browser",
]


@dataclass(frozen=True)
class Shot:
    name: str
    url: str
    width: int
    height: int
    #: JavaScript that must evaluate truthy before the capture. This is the
    #: whole point of the script: "the page loaded" and "the page has something
    #: on it" are different moments, and only the second is worth a screenshot.
    ready: str
    timeout_s: float = 90.0
    warm: tuple[str, dict] | None = None
    settle_s: float = 1.2


QUESTION = "Why did the mechanical seal on P-101B fail after startup?"

SHOTS: list[Shot] = [
    Shot(
        "index",
        f"{BASE}/index.html",
        1500,
        1150,
        ready="document.querySelectorAll('[data-assets] tbody tr').length > 0",
    ),
    Shot(
        "ingestion",
        f"{BASE}/ingestion.html",
        1500,
        1250,
        ready="!!document.querySelector('[data-jobs] .job, [data-jobs] tbody tr, [data-jobs] .state')",
    ),
    Shot(
        "graph",
        f"{BASE}/graph.html?asset=P-101B",
        1500,
        1150,
        # [data-graph] *is* the <svg>, and the view renders one <g> wrapper
        # into it -- so counting its children never gets past 1. Count the nodes
        # instead: a drawn node means the neighbourhood really arrived.
        ready="document.querySelectorAll('[data-graph] [data-node]').length > 0",
        settle_s=2.0,
    ),
    Shot(
        "copilot",
        f"{BASE}/copilot.html?q={urllib.parse.quote(QUESTION)}&asset_tag=P-101B",
        1500,
        1500,
        # Citations appear only once the stream has delivered them.
        ready="document.querySelectorAll('.citation-row').length > 0",
        warm=("/api/v1/query", {"question": QUESTION, "asset_tag": "P-101B"}),
        settle_s=2.0,
    ),
    Shot(
        "reliability",
        f"{BASE}/reliability.html?asset=P-101B",
        1500,
        1600,
        ready="!!document.querySelector('[data-dossier] .tag, [data-dossier] table')",
        settle_s=2.0,
    ),
    Shot(
        "compliance",
        f"{BASE}/compliance.html?asset=V-102",
        1500,
        1500,
        ready="document.querySelectorAll('[data-findings] .finding, [data-findings] li').length > 0",
        settle_s=2.0,
    ),
    Shot(
        # Tall enough that the whole page fits the viewport. The ask bar is
        # position:sticky, and in a full-page capture of a short viewport it
        # renders across the middle of the image -- an artefact of the capture,
        # not of the page, but an artefact a reader would read as a bug.
        "field",
        f"{BASE}/field.html?asset=P-101B",
        430,
        1900,
        ready="!!document.querySelector('[data-asset-hero] .tag, [data-asset-hero] h1, [data-asset-hero] strong')",
        settle_s=2.0,
    ),
]


# ---------------------------------------------------------------------------
# A very small DevTools protocol client
# ---------------------------------------------------------------------------


@dataclass
class Devtools:
    ws_url: str
    _next_id: int = field(default=1, init=False)

    def __enter__(self) -> Devtools:
        self._ws = connect(self.ws_url, max_size=64 * 1024 * 1024, open_timeout=30)
        return self

    def __exit__(self, *exc: object) -> None:
        self._ws.close()

    def call(self, method: str, params: dict | None = None, timeout: float = 60.0) -> dict:
        message_id = self._next_id
        self._next_id += 1
        self._ws.send(json.dumps({"id": message_id, "method": method, "params": params or {}}))
        deadline = time.monotonic() + timeout
        while time.monotonic() < deadline:
            raw = self._ws.recv(timeout=max(1.0, deadline - time.monotonic()))
            payload = json.loads(raw)
            if payload.get("id") != message_id:
                continue  # an event, not our reply
            if "error" in payload:
                raise RuntimeError(f"{method}: {payload['error']}")
            return payload.get("result", {})
        raise TimeoutError(f"{method} did not answer within {timeout}s")

    def evaluate(self, expression: str) -> object:
        result = self.call(
            "Runtime.evaluate",
            {"expression": expression, "returnByValue": True, "awaitPromise": True},
        )
        return result.get("result", {}).get("value")


def free_port() -> int:
    with socket.socket() as sock:
        sock.bind(("127.0.0.1", 0))
        return int(sock.getsockname()[1])


def find_browser() -> str:
    for candidate in BROWSER_CANDIDATES:
        if Path(candidate).exists():
            return candidate
        found = shutil.which(candidate)
        if found:
            return found
    raise SystemExit(
        "No Chromium-family browser found. Install Edge or Chrome, or add its path "
        "to BROWSER_CANDIDATES in this script."
    )


def start_browser(port: int, profile: Path) -> subprocess.Popen:
    return subprocess.Popen(
        [
            find_browser(),
            "--headless=new",
            "--disable-gpu",
            "--hide-scrollbars",
            "--no-first-run",
            "--no-default-browser-check",
            f"--remote-debugging-port={port}",
            f"--user-data-dir={profile}",
            "about:blank",
        ],
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )


def page_target(port: int, timeout: float = 30.0) -> str:
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        try:
            with urllib.request.urlopen(f"http://127.0.0.1:{port}/json", timeout=3) as response:
                for target in json.load(response):
                    if target.get("type") == "page":
                        return str(target["webSocketDebuggerUrl"])
        except Exception:
            time.sleep(0.4)
    raise SystemExit("the browser never exposed a debugging endpoint")


def warm(path: str, payload: dict) -> None:
    """Issue the request the page will make, so the page does not wait on a cold model.

    Not a shortcut: it is the same endpoint returning the same answer. The
    reranker cache makes the second call fast, which keeps the capture inside a
    sane timeout.
    """
    request = urllib.request.Request(
        f"http://localhost:8000{path}",
        data=json.dumps(payload).encode(),
        headers={"Content-Type": "application/json"},
    )
    try:
        with urllib.request.urlopen(request, timeout=180):
            pass
    except Exception as exc:  # pragma: no cover - warming is best effort
        print(f"    (warm-up failed: {type(exc).__name__}: {exc})")


def capture(dt: Devtools, shot: Shot) -> bool:
    dt.call(
        "Emulation.setDeviceMetricsOverride",
        {
            "width": shot.width,
            "height": shot.height,
            "deviceScaleFactor": 1,
            "mobile": shot.width < 700,
        },
    )
    # Blank the tab between captures. Several surfaces hold an open
    # Server-Sent Events connection to /api/v1/events/stream, and one tab
    # accumulating them across seven navigations is asking for trouble with
    # Chrome's six-connections-per-host limit.
    dt.call("Page.navigate", {"url": "about:blank"})
    time.sleep(0.4)
    dt.call("Page.navigate", {"url": shot.url})

    deadline = time.monotonic() + shot.timeout_s
    ready = False
    while time.monotonic() < deadline:
        try:
            if dt.evaluate(f"!!({shot.ready})"):
                ready = True
                break
        except RuntimeError:
            pass  # the document is mid-navigation
        time.sleep(0.5)

    if not ready:
        # Say why, rather than only that. A capture script that reports "not
        # ready" and nothing else leaves you guessing between a broken page, a
        # wrong selector and a slow backend -- three very different problems.
        print(f"    NOT READY after {shot.timeout_s:.0f}s: {shot.ready}")
        for label, expression in (
            ("url", "location.href"),
            ("readyState", "document.readyState"),
            ("visible error", "document.querySelector('.state.error')?.innerText?.slice(0,160)"),
            ("still loading", "!!document.querySelector('.spinner')"),
            ("last console error", "window.__lastError || null"),
        ):
            try:
                print(f"      {label:18} {dt.evaluate(expression)!r}")
            except RuntimeError as exc:
                print(f"      {label:18} <{exc}>")
        return False

    time.sleep(shot.settle_s)  # let the last paint land
    result = dt.call(
        "Page.captureScreenshot",
        {"format": "png", "captureBeyondViewport": True},
        timeout=120,
    )
    target = OUT_DIR / f"{shot.name}.png"
    target.write_bytes(base64.b64decode(result["data"]))
    print(f"    {target.relative_to(REPO_ROOT)}  {target.stat().st_size:,} bytes")
    return True


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("names", nargs="*", help="surfaces to capture (default: all)")
    parser.add_argument("--keep-open", action="store_true")
    args = parser.parse_args()

    try:
        with urllib.request.urlopen("http://localhost:8000/health/live", timeout=5):
            pass
    except Exception:
        print("The stack is not answering on :8000. Start it with: docker compose up -d")
        return 2

    wanted = [s for s in SHOTS if not args.names or s.name in args.names]
    if not wanted:
        print(f"no such surface. known: {', '.join(s.name for s in SHOTS)}")
        return 2

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    port = free_port()
    profile = Path(REPO_ROOT / ".screenshot-profile")
    browser = start_browser(port, profile)
    failures = 0
    try:
        ws_url = page_target(port)
        with Devtools(ws_url) as dt:
            dt.call("Page.enable")
            dt.call("Runtime.enable")
            for shot in wanted:
                print(f"  {shot.name}")
                if shot.warm:
                    warm(*shot.warm)
                if not capture(dt, shot):
                    failures += 1
    finally:
        if not args.keep_open:
            browser.terminate()
            try:
                browser.wait(timeout=15)
            except subprocess.TimeoutExpired:  # pragma: no cover
                browser.kill()
            shutil.rmtree(profile, ignore_errors=True)

    print(f"\n{len(wanted) - failures}/{len(wanted)} captured")
    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
