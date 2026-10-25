"""Record docs/demo.gif: the Streamlit app answering one question, step by step.

    streamlit run streamlit_app.py           # in one terminal (uses your .env model)
    python scripts/record_demo.py            # in another

Needs `pip install -e ".[demo]"` and Google Chrome. Frames are screenshots taken
while the agent works, so the GIF shows real tool calls, not a mock-up.
"""

import argparse
import io
import sys
import time
from pathlib import Path

from PIL import Image
from playwright.sync_api import sync_playwright

ROOT = Path(__file__).resolve().parents[1]
QUESTION = ("I want a furnished 1-bedroom in JVC under AED 90k. Compare the two cheapest, "
            "and tell me how much notice is needed before changing the lease terms.")
WIDTH, HEIGHT = 1000, 1150
FRAME_SECONDS = 0.5


def grab(page) -> Image.Image:
    return Image.open(io.BytesIO(page.screenshot())).convert("RGB")


def record(url: str, timeout: float) -> list[tuple[Image.Image, int]]:
    """Return (frame, milliseconds) pairs."""
    frames = []
    with sync_playwright() as p:
        browser = p.chromium.launch(channel="chrome", headless=True)
        page = browser.new_page(viewport={"width": WIDTH, "height": HEIGHT})
        page.goto(url)
        page.get_by_text("Try one:").wait_for(timeout=60_000)
        time.sleep(1)
        frames.append((grab(page), 1500))
        page.get_by_role("button", name=QUESTION).click()
        # The intro also contains this sentence, so look only inside chat messages.
        answer = page.locator("[data-testid='stChatMessage']").filter(
            has_text="Information only, not legal or financial advice.")
        deadline = time.time() + timeout
        while time.time() < deadline:
            frames.append((grab(page), int(FRAME_SECONDS * 1000)))
            if answer.count():
                break
            time.sleep(FRAME_SECONDS)
        else:
            raise SystemExit("No answer appeared; is the model reachable?")
        time.sleep(1)
        frames.append((grab(page), 2500))
        # Open the first tool step to show its arguments and result.
        page.locator("[data-testid='stExpander'] summary").first.click()
        time.sleep(1)
        frames.append((grab(page), 3500))
        browser.close()
    return frames


def save_gif(frames: list[tuple[Image.Image, int]], path: Path, width: int = 800) -> None:
    images, durations = [], []
    for frame, ms in frames:
        if images and frame.tobytes() == images[-1].tobytes():
            durations[-1] += ms  # merge identical frames
            continue
        images.append(frame)
        durations.append(ms)
    height = round(images[0].height * width / images[0].width)
    small = [im.resize((width, height), Image.LANCZOS).quantize(colors=128) for im in images]
    small[0].save(path, save_all=True, append_images=small[1:], duration=durations, loop=0, optimize=True)


def main(argv: list[str]) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--url", default="http://localhost:8501")
    parser.add_argument("--out", type=Path, default=ROOT / "docs" / "demo.gif")
    parser.add_argument("--timeout", type=float, default=90)
    args = parser.parse_args(argv)
    frames = record(args.url, args.timeout)
    save_gif(frames, args.out)
    print(f"Wrote {args.out.relative_to(ROOT)} ({len(frames)} frames, {args.out.stat().st_size / 1e6:.1f} MB)")
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
