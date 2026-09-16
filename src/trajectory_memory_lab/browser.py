from __future__ import annotations

import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from playwright.sync_api import Browser, Page, Playwright, sync_playwright


@dataclass
class Observation:
    url: str
    title: str
    text: str
    elements: list[dict[str, Any]]

    def prompt_text(self, *, max_text_chars: int = 6_000) -> str:
        element_lines = []
        for item in self.elements:
            attrs = []
            if item.get("role"):
                attrs.append(f"role={item['role']!r}")
            if item.get("name"):
                attrs.append(f"name={item['name']!r}")
            if item.get("placeholder"):
                attrs.append(f"placeholder={item['placeholder']!r}")
            element_lines.append(
                f"[{item['ref']}] <{item['tag']}> " + " ".join(attrs)
            )
        return (
            f"URL: {self.url}\nTITLE: {self.title}\n\n"
            f"INTERACTIVE ELEMENTS:\n" + "\n".join(element_lines) + "\n\n"
            f"VISIBLE PAGE TEXT:\n{self.text[:max_text_chars]}"
        )


class BrowserSession:
    def __init__(self, *, headless: bool, screenshot_dir: Path) -> None:
        self.headless = headless
        self.screenshot_dir = screenshot_dir
        self._playwright: Playwright | None = None
        self._browser: Browser | None = None
        self.page: Page | None = None

    def __enter__(self) -> "BrowserSession":
        self.screenshot_dir.mkdir(parents=True, exist_ok=True)
        self._playwright = sync_playwright().start()
        self._browser = self._playwright.chromium.launch(headless=self.headless)
        context = self._browser.new_context(
            viewport={"width": 1280, "height": 900},
            locale="en-US",
            user_agent=(
                "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 "
                "Chrome/131.0 Safari/537.36 trajectory-memory-lab/0.1"
            ),
        )
        self.page = context.new_page()
        return self

    def __exit__(self, exc_type: Any, exc: Any, tb: Any) -> None:
        if self._browser:
            self._browser.close()
        if self._playwright:
            self._playwright.stop()

    def goto(self, url: str) -> None:
        assert self.page
        self.page.goto(url, wait_until="domcontentloaded", timeout=60_000)
        self.page.wait_for_timeout(1200)

    def observe(self, screenshot_index: int) -> Observation:
        assert self.page
        raw_elements = self.page.evaluate(
            r"""
            () => {
              const candidates = Array.from(document.querySelectorAll(
                'a,button,input,textarea,select,[role="button"],[role="link"],[role="textbox"],[contenteditable="true"]'
              ));
              let ref = 0;
              const result = [];
              for (const el of candidates) {
                const rect = el.getBoundingClientRect();
                const style = window.getComputedStyle(el);
                if (rect.width < 1 || rect.height < 1 || style.visibility === 'hidden' || style.display === 'none') continue;
                ref += 1;
                el.setAttribute('data-memory-lab-ref', String(ref));
                const name = (
                  el.getAttribute('aria-label') ||
                  el.getAttribute('title') ||
                  el.innerText ||
                  el.value ||
                  el.getAttribute('alt') ||
                  ''
                ).replace(/\s+/g, ' ').trim().slice(0, 240);
                result.push({
                  ref,
                  tag: el.tagName.toLowerCase(),
                  role: el.getAttribute('role') || '',
                  name,
                  placeholder: el.getAttribute('placeholder') || ''
                });
                if (result.length >= 180) break;
              }
              return result;
            }
            """
        )
        try:
            text = self.page.locator("body").inner_text(timeout=10_000)
        except Exception:
            text = ""
        text = re.sub(r"\n{3,}", "\n\n", text)
        self.page.screenshot(
            path=str(self.screenshot_dir / f"step_{screenshot_index:02d}.png"),
            full_page=False,
        )
        return Observation(
            url=self.page.url,
            title=self.page.title(),
            text=text,
            elements=raw_elements,
        )

    def execute(self, action: dict[str, Any]) -> str:
        assert self.page
        kind = str(action.get("action", "")).lower()
        ref = action.get("ref")
        if kind == "click":
            self._by_ref(ref).click(timeout=15_000)
            self.page.wait_for_timeout(1200)
            return f"clicked [{ref}]"
        if kind == "fill":
            value = str(action.get("text", ""))
            self._by_ref(ref).fill(value, timeout=15_000)
            return f"filled [{ref}] with {value!r}"
        if kind == "press":
            key = str(action.get("key", "Enter"))
            if ref is None:
                self.page.keyboard.press(key)
            else:
                self._by_ref(ref).press(key, timeout=15_000)
            self.page.wait_for_timeout(1200)
            return f"pressed {key!r} on {ref!r}"
        if kind == "scroll":
            amount = int(action.get("amount", 700))
            self.page.mouse.wheel(0, amount)
            self.page.wait_for_timeout(500)
            return f"scrolled {amount} pixels"
        if kind == "back":
            self.page.go_back(wait_until="domcontentloaded", timeout=30_000)
            self.page.wait_for_timeout(800)
            return "went back"
        if kind == "goto":
            url = str(action.get("url", ""))
            if not url.startswith(("http://", "https://")):
                raise ValueError("goto requires an absolute http(s) URL")
            self.goto(url)
            return f"navigated to {url}"
        if kind == "finish":
            return "finished"
        raise ValueError(f"Unsupported action: {json.dumps(action, ensure_ascii=False)}")

    def _by_ref(self, ref: Any):
        if not isinstance(ref, int):
            raise ValueError("Action requires integer ref")
        assert self.page
        return self.page.locator(f'[data-memory-lab-ref="{ref}"]').first
