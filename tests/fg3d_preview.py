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

SHARP = dict(focus=0.05, fstop=5.6)   # focus on the glass itself
LOOKS = {
    # Prisms & filters
    "00_faceted_plate": dict(shape=0, facets=3, prismAngle=6, size=0.9, depth=0.12, center=(0.42, 0.5), rotZ=8),
    "01_wedge_prism": dict(shape=1, size=0.55, depth=0.15, center=(0.62, 0.5), rotZ=10, rotY=20, fstop=2.8),
    "02_radial_prism": dict(shape=2, facets=6, prismAngle=7, clearCentre=0.5, size=1.6, depth=0.15),
    "03_half_radial": dict(shape=3, facets=8, prismAngle=7, clearCentre=0.45, size=1.6, depth=0.15),
    "04_kaleidoscope": dict(shape=4, facets=3, size=1.2, depth=0.2, maxBounces=24),
    "05_halo_ring": dict(shape=5, clearCentre=0.55, prismAngle=6, size=1.4, depth=0.15),
    "06_ripple_rings": dict(shape=6, facets=4, prismAngle=8, size=1.3, depth=0.15),
    "07_half_ripple": dict(shape=7, facets=4, prismAngle=8, size=1.6, depth=0.15, rotZ=180),
    "08_honeycomb": dict(shape=8, facets=3, prismAngle=6, size=1.0, depth=0.15, fstop=11),
    "09_fresnel_lens": dict(shape=9, facets=3, power=4, size=1.0, depth=0.2, fstop=8),
    # Lenses
    "10_round_diopter": dict(shape=10, power=6, size=0.7, depth=0.2, center=(0.5, 0.6), dispersion=0.6),
    "11_split_diopter": dict(shape=11, power=4, size=2.5, depth=0.15, rotZ=180, focus=0.5, glassFocus=1.0, fstop=2.0),
    "12_double_convex": dict(shape=12, power=7, size=0.8, depth=0.2, center=(0.45, 0.55)),
    "13_cylinder_lens": dict(shape=13, power=6, size=0.8, depth=0.15, rotZ=90),
    "14_rod": dict(shape=14, size=0.6, depth=0.12, center=(0.65, 0.5), rotZ=-12, fstop=2.0),
    # Gems
    "15_round_brilliant": dict(shape=15, size=0.7, depth=0.15, center=(0.4, 0.45), rotX=-35, rotY=15, ior=2.0,
                               dispersion=0.8, maxBounces=16, light=(0.2, 0.8, 0.3, 0.1, 6500, 3), **SHARP),
    "16_emerald_cut": dict(shape=16, size=0.7, depth=0.15, rotX=-30, rotY=20, ior=1.9, dispersion=0.6, **SHARP),
    "17_princess_cut": dict(shape=17, size=0.7, depth=0.15, rotX=-30, rotY=25, ior=2.0, dispersion=0.8, **SHARP),
    "18_pear_cut": dict(shape=18, size=0.7, depth=0.15, rotX=-25, rotZ=90, ior=2.0, dispersion=0.8, **SHARP),
    "19_heart_cut": dict(shape=19, size=0.7, depth=0.15, rotX=-20, ior=2.0, dispersion=0.8, **SHARP),
    "20_marquise_cut": dict(shape=20, size=0.8, depth=0.15, rotX=-25, rotZ=20, ior=2.0, dispersion=0.8, **SHARP),
    "21_crystal_cluster": dict(shape=21, facets=7, size=1.2, depth=0.15, center=(0.3, 0.4), ior=1.7, dispersion=0.7,
                               seed=3, **SHARP),
    # Solids
    "22_tetrahedron": dict(shape=22, size=0.7, depth=0.2, rotX=20, rotY=30, **SHARP),
    "23_cube": dict(shape=23, size=0.6, depth=0.25, rotX=25, rotY=35, focus=0.0, fstop=8,
                    light=(0.7, 0.8, 0.4, 0.15, 3200, 4)),
    "24_octahedron": dict(shape=24, size=0.7, depth=0.2, rotX=15, rotY=25, **SHARP),
    "25_dodecahedron": dict(shape=25, size=0.7, depth=0.2, rotX=20, rotY=30, **SHARP),
    "26_icosahedron": dict(shape=26, size=0.7, depth=0.2, rotX=20, rotY=30, **SHARP),
    "27_geodesic_sphere": dict(shape=27, detail=1, size=0.8, depth=0.2, center=(0.62, 0.45), rotX=20, rotY=30, ior=1.6,
                               fstop=2.8, light=(0.25, 0.8, 0.3, 0.12, 5600, 3)),
    "28_glass_ball": dict(shape=28, size=0.7, depth=0.2, **SHARP),
    "29_pyramid": dict(shape=29, size=0.9, depth=0.15, rotZ=20, ior=1.55, fstop=2.8),
    "30_hexagonal_prism": dict(shape=30, size=0.7, depth=0.2, rotX=30, rotY=20, **SHARP),
    "31_ring": dict(shape=31, size=0.9, depth=0.2, rotX=35, **SHARP),
    # Organic
    "32_wavy_glass": dict(shape=32, facets=3, prismAngle=8, size=1.2, depth=0.12),
    "33_fluted_glass": dict(shape=33, facets=5, prismAngle=8, size=1.4, depth=0.12),
    "34_bubble_glass": dict(shape=34, facets=4, prismAngle=8, size=1.4, depth=0.12),
    "35_water_drops": dict(shape=35, facets=4, prismAngle=8, size=1.4, depth=0.12, **SHARP),
    "36_ice_cube": dict(shape=36, size=0.7, depth=0.2, rotX=20, rotY=25, ior=1.31, dispersion=0.2, **SHARP),
    "37_melted_blob": dict(shape=37, size=0.8, depth=0.2, seed=2, **SHARP),
    "38_shards": dict(shape=38, facets=4, size=1.2, depth=0.15, seed=4, **SHARP),
    "39_twisted_bar": dict(shape=39, prismAngle=6, size=0.6, depth=0.15, rotZ=15, **SHARP),
    "40_bent_slab": dict(shape=40, prismAngle=6, size=0.9, depth=0.12, rotZ=10),
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
    cols = 6
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
