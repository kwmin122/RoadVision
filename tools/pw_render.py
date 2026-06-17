"""Playwright로 11개 슬라이드를 1280×720 PNG로 렌더(검수용).
사용: python -m tools.pw_render
출력: presentation/pw/slideNN.png
"""
from __future__ import annotations
import os
from playwright.sync_api import sync_playwright

BASE = os.path.abspath("presentation/slides")
OUT = "presentation/pw"


def main():
    os.makedirs(OUT, exist_ok=True)
    with sync_playwright() as p:
        b = p.chromium.launch()
        pg = b.new_page(viewport={"width": 1280, "height": 720}, device_scale_factor=1)
        for i in range(1, 12):
            n = f"{i:02d}"
            path = f"{BASE}/slide-{n}.html"
            if not os.path.exists(path):
                print(f"없음: {path}"); continue
            pg.goto(f"file://{path}")
            pg.wait_for_timeout(450)
            pg.screenshot(path=f"{OUT}/slide{n}.png")
            print(f"렌더: slide{n}.png")
        b.close()
    print("완료:", OUT)


if __name__ == "__main__":
    main()
