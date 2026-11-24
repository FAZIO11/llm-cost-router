#!/usr/bin/env python3
"""Capture a screenshot of the Streamlit dashboard."""

import argparse
import asyncio
from pathlib import Path
from playwright.async_api import async_playwright


async def capture_screenshot(output_path: Path, width: int = 1440, height: int = 1200):
    """Capture dashboard screenshot."""
    async with async_playwright() as p:
        browser = await p.chromium.launch()
        page = await browser.new_page(viewport={"width": width, "height": height})
        
        # Navigate to dashboard
        await page.goto("http://localhost:8501", wait_until="networkidle", timeout=30000)
        
        # Wait for content to load
        await asyncio.sleep(3)
        
        # Ensure output directory exists
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        # Take screenshot
        await page.screenshot(path=str(output_path), full_page=False)
        
        await browser.close()
        
        print(f"Screenshot saved to: {output_path}")


def main():
    parser = argparse.ArgumentParser(description="Capture dashboard screenshot")
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("docs/dashboard.png"),
        help="Output screenshot path (default: docs/dashboard.png)",
    )
    parser.add_argument(
        "--width",
        type=int,
        default=1440,
        help="Viewport width (default: 1440)",
    )
    parser.add_argument(
        "--height",
        type=int,
        default=1200,
        help="Viewport height (default: 1200)",
    )
    args = parser.parse_args()
    
    asyncio.run(capture_screenshot(args.output, args.width, args.height))


if __name__ == "__main__":
    main()
