"""Generate seedtrace brand assets (logo + GitHub social preview).

The mark is the product's core idea drawn literally: two execution runs
travel identical (one line), hit a checkpoint (dashed marker), then
diverge - one run stays on the healthy trajectory (emerald node), one
strays (rose node). Reproducible branding: run this script, get the
assets.

    python docs/generate_brand_assets.py

Needs Pillow (dev-only). Outputs:
    docs/assets/logo.png            1024x1024, dark rounded tile
    docs/assets/logo-light.png      1024x1024, light rounded tile
    docs/assets/social-preview.png  1280x640, GitHub Open Graph card
"""

from __future__ import annotations

import os

from PIL import Image, ImageDraw, ImageFont

# -- palette -----------------------------------------------------------
NAVY = (15, 23, 42)          # #0f172a  background (dark)
PAPER = (248, 250, 252)     # #f8fafc  background (light)
INK = (226, 232, 240)        # #e2e8f0  shared-path line (dark theme)
INK_DIM = (148, 163, 184)   # #94a3b8  secondary
SLATE_LINE = (71, 85, 105)  # #475569  shared-path line (light theme)
EMERALD = (16, 185, 129)    # #10b981  deterministic outcome
ROSE = (244, 63, 94)        # #f43f5e  divergent outcome
GRID_DARK = (51, 65, 85)    # dashed checkpoint marker (dark)
GRID_LIGHT = (148, 163, 184)

SS = 4                        # supersampling factor for crisp curves
HERE = os.path.dirname(os.path.abspath(__file__))
ASSETS = os.path.join(HERE, "assets")


def _cubic(p0, p1, p2, p3, n=28):
    """Sample a cubic Bezier curve into a polyline (logical units).

    Keep the sample count low: dense polylines with joint="curve" leave
    hairline gaps between segment quads that show up as tick artifacts.
    """
    pts = []
    for i in range(n + 1):
        t = i / n
        mt = 1.0 - t
        x = mt**3 * p0[0] + 3 * mt**2 * t * p1[0] + 3 * mt * t**2 * p2[0] + t**3 * p3[0]
        y = mt**3 * p0[1] + 3 * mt**2 * t * p1[1] + 3 * mt * t**2 * p2[1] + t**3 * p3[1]
        pts.append((x, y))
    return pts


def _draw_trace(d: ImageDraw.ImageDraw, dark: bool) -> None:
    """Draw the divergence mark in a 1024x1024 logical space, pre-scaled."""
    shared = INK if dark else SLATE_LINE
    grid = GRID_DARK if dark else GRID_LIGHT
    origin = INK_DIM if dark else SLATE_LINE
    w = 26

    # shared path enters from the left, gently rising to the checkpoint
    shared_pts = _cubic((150, 560), (330, 560), (410, 505), (512, 505))
    d.line(shared_pts, fill=shared, width=w, joint="curve")

    # dashed vertical checkpoint marker where the runs split
    y = 215
    while y < 795:
        d.line([(512, y), (512, y + 14)], fill=grid, width=7)
        y += 26

    # branch A: stays on the healthy trajectory -> emerald node
    a_pts = _cubic((512, 505), (640, 505), (724, 470), (820, 452))
    d.line(a_pts, fill=shared, width=w, joint="curve")

    # branch B: strays upward into divergence -> rose node
    b_pts = _cubic((512, 505), (652, 488), (706, 352), (820, 302))
    d.line(b_pts, fill=ROSE, width=w, joint="curve")

    # terminal nodes with halo rings
    for cx, cy, color in ((820, 452, EMERALD), (820, 302, ROSE)):
        r = 50
        d.ellipse((cx - r - 20, cy - r - 20, cx + r + 20, cy + r + 20),
                  outline=color, width=9)
        d.ellipse((cx - r, cy - r, cx + r, cy + r), fill=color)

    # origin node on the shared path
    r0 = 32
    d.ellipse((150 - r0, 560 - r0, 150 + r0, 560 + r0), fill=origin)


def make_logo(dark: bool, size: int = 1024) -> Image.Image:
    """Render the rounded-tile logo at `size` px square.

    The trace is drawn in a clean 1024 logical space, then upscaled with
    Lanczos before compositing - giving perfectly smooth curves without
    threading scale factors through every coordinate.
    """
    img = Image.new("RGBA", (1024 * SS, 1024 * SS), (0, 0, 0, 0))
    d = ImageDraw.Draw(img, "RGBA")
    d.rounded_rectangle((0, 0, 1024 * SS - 1, 1024 * SS - 1),
                        radius=190 * SS, fill=NAVY if dark else PAPER)
    _draw_trace_scaled(img, dark)
    return img.resize((size, size), Image.LANCZOS)


def _draw_trace_scaled(img: Image.Image, dark: bool) -> None:
    """Wrapper that maps the 1024-space trace onto the supersampled canvas."""
    canvas = Image.new("RGBA", (1024, 1024), (0, 0, 0, 0))
    d = ImageDraw.Draw(canvas, "RGBA")
    _draw_trace(d, dark)
    canvas = canvas.resize((1024 * SS, 1024 * SS), Image.LANCZOS)
    img.alpha_composite(canvas)


def _font(px: int, bold: bool = True):
    candidates = []
    if bold:
        candidates.append(r"C:\Windows\Fonts\consolab.ttf")
    candidates.append(r"C:\Windows\Fonts\consola.ttf")
    for path in candidates:
        if os.path.exists(path):
            try:
                return ImageFont.truetype(path, px)
            except OSError:
                continue
    return ImageFont.load_default()


def make_social(path: str, size=(1280, 640)) -> None:
    """Compose the 1280x640 GitHub social preview card."""
    w, h = size
    img = Image.new("RGB", (w, h), NAVY)
    d = ImageDraw.Draw(img, "RGB")

    # top accent rule
    d.rectangle((0, 0, w, 5), fill=EMERALD)

    # logo panel on the left
    logo = make_logo(dark=True, size=440).convert("RGBA")
    panel = Image.new("RGBA", (440 + 48, 440 + 48), (0, 0, 0, 0))
    pd = ImageDraw.Draw(panel, "RGBA")
    pd.rounded_rectangle((0, 0, panel.width - 1, panel.height - 1),
                         radius=70, fill=(30, 41, 59, 255))
    panel.alpha_composite(logo, (24, 24))
    img.paste(panel.convert("RGB"), (60, (h - panel.height) // 2), panel)

    # wordmark + tagline + URL
    x = 640
    d.text((x, 140), "seedtrace", font=_font(84), fill=(248, 250, 252))
    tag = _font(30, bold=False)
    for i, line in enumerate(
        [
            "Find out why two runs with the",
            "same seed gave different results.",
        ]
    ):
        d.text((x, 278 + i * 44), line, font=tag, fill=INK_DIM)
    d.text((x, 408), "github.com/realgauravvyas/seedtrace",
           font=_font(28, bold=False), fill=EMERALD)

    img.save(path)


def main() -> None:
    os.makedirs(ASSETS, exist_ok=True)
    make_logo(dark=True).save(os.path.join(ASSETS, "logo.png"))
    make_logo(dark=False).save(os.path.join(ASSETS, "logo-light.png"))
    make_social(os.path.join(ASSETS, "social-preview.png"))
    print("wrote docs/assets/logo.png, logo-light.png, social-preview.png")


if __name__ == "__main__":
    main()
