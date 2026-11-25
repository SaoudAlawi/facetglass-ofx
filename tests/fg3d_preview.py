"""Render a contact sheet of FacetGlass 3D looks with the CPU renderer.

Usage: python tests/fg3d_preview.py <fg3d_render> [W H] [look ...]
Writes tests/out/fg3d/<look>.png and tests/out/fg3d/contact_sheet.png.
"""
import math
import os
import subprocess
import sys

import numpy as np
from PIL import Image, ImageDraw

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "..", "reference", "kernel-harness"))
import run  # test image

exe = os.path.abspath(sys.argv[1])
args = sys.argv[2:]
W, H = (int(args[0]), int(args[1])) if len(args) >= 2 and args[0].isdigit() else (640, 360)
which = [a for a in args if not a.isdigit()]
OUT = os.path.join(HERE, "out", "fg3d")
os.makedirs(OUT, exist_ok=True)

LOOKS = {
    "00_faceted_plate": dict(shape=0, facets=3, prismAngle=6, size=0.9, depth=0.12, center=(0.42, 0.5), rotZ=8),
    "01_ghost_both_sides": dict(shape=0, facets=3, bothSides=1, prismAngle=5, size=0.7, depth=0.12, rotZ=15),
    "02_wedge_prism": dict(shape=1, size=0.55, depth=0.15, center=(0.62, 0.5), rotZ=10, rotY=20, fstop=2.8),
    "03_radial_prism": dict(shape=2, facets=6, prismAngle=7, clearCentre=0.5, size=1.6, depth=0.15),
    "04_split_diopter": dict(shape=3, power=4, size=2.5, depth=0.15, center=(0.5, 0.5), rotZ=180, focus=0.5,
                             glassFocus=1.0, fstop=2.0),
    "05_round_diopter": dict(shape=4, power=6, size=0.7, depth=0.2, center=(0.5, 0.6), dispersion=0.6),
    "06_geodesic_sphere": dict(shape=5, detail=1, size=0.8, depth=0.2, center=(0.62, 0.45), rotX=20, rotY=30,
                               ior=1.6, fstop=2.8, light=(0.25, 0.8, 0.3, 0.12, 5600, 3)),
    "07_gem": dict(shape=6, size=0.7, depth=0.15, center=(0.35, 0.45), rotX=-35, rotY=15, ior=2.0, dispersion=0.8,
                   fstop=5.6, focus=0.05, maxBounces=16, light=(0.2, 0.8, 0.3, 0.1, 6500, 3)),
    "08_crystal_cluster": dict(shape=7, facets=7, size=1.2, depth=0.15, center=(0.3, 0.4), ior=1.7, dispersion=0.7,
                               fstop=5.6, focus=0.05, seed=3),
    "09_box_focus_on_glass": dict(shape=8, size=0.6, depth=0.25, rotX=25, rotY=35, focus=0.0, fstop=8,
                                  light=(0.7, 0.8, 0.4, 0.15, 3200, 4)),
    "10_pyramid": dict(shape=9, size=0.9, depth=0.15, rotZ=20, ior=1.55, fstop=2.8),
    "11_rod_stretch": dict(shape=10, size=0.6, depth=0.12, center=(0.65, 0.5), rotZ=-12, fstop=2.0),
}


def fmt(v):
    return ",".join(repr(float(x)) for x in v) if isinstance(v, tuple) else repr(v)


def to8(x):
    x = x[..., :3]
    x = x / (1.0 + 0.15 * np.maximum(x - 1.0, 0.0))
    return (np.clip(x, 0, 1) ** (1 / 2.2) * 255 + 0.5).astype(np.uint8)


img = run.test_image()  # 960x540, display-ish values with HDR lights; treated as linear here
img = np.asarray(Image.fromarray((np.clip(img, 0, 1) * 255).astype(np.uint8)).resize((W, H), Image.LANCZOS),
                 np.float32) / 255.0
hdr = run.test_image()
small = np.asarray(Image.fromarray(np.zeros((1, 1), np.uint8)))  # unused
# keep the HDR fairy lights: max-pool the original's overbright parts into the resized image
ys = (np.arange(H) * hdr.shape[0] / H).astype(int)
xs = (np.arange(W) * hdr.shape[1] / W).astype(int)
img = np.maximum(img ** 2.2, np.where(hdr[ys][:, xs] > 1.0, hdr[ys][:, xs], 0.0))
rgba = np.concatenate([img, np.ones((H, W, 1), np.float32)], -1)[::-1].astype(np.float32).copy()
inp = os.path.join(OUT, "in.raw")
rgba.tofile(inp)

tiles = []
for name in (which or list(LOOKS)):
    kw = LOOKS[name]
    o = os.path.join(OUT, name + ".raw")
    cmd = [exe, inp, o, str(W), str(H)] + [f"{k}={fmt(v)}" for k, v in kw.items()]
    r = subprocess.run(cmd, capture_output=True, text=True)
    if r.returncode != 0:
        print(r.stdout, r.stderr)
        raise SystemExit(f"render failed: {name}")
    res = np.fromfile(o, np.float32).reshape(H, W, 4)[::-1]
    t8 = to8(res)
    Image.fromarray(t8).save(os.path.join(OUT, name + ".png"))
    tiles.append((name, t8))
    print(f"{name:24s} {r.stdout.strip()}")

if len(tiles) > 1:
    cols = 3
    tw, th = W // 2, H // 2
    rows = math.ceil(len(tiles) / cols)
    sheet = Image.new("RGB", (cols * tw, rows * (th + 18)), (18, 18, 18))
    d = ImageDraw.Draw(sheet)
    for i, (n, t) in enumerate(tiles):
        x, y = (i % cols) * tw, (i // cols) * (th + 18)
        sheet.paste(Image.fromarray(t).resize((tw, th), Image.LANCZOS), (x, y + 18))
        d.text((x + 5, y + 3), n[3:].replace("_", " "), fill=(225, 225, 225))
    sheet.save(os.path.join(OUT, "contact_sheet.png"))
    print("sheet:", os.path.join(OUT, "contact_sheet.png"))
