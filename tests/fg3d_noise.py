"""Grain measurement: render looks at the plugin's quality presets and compare with
a high-sample reference. Reports RMS error in display space (lower = less grain)
and writes side-by-side crops to tests/out/noise/.

Usage: python tests/fg3d_noise.py <fg3d_render> [tag]
"""
import os
import subprocess
import sys

import numpy as np
from PIL import Image

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
exe = os.path.abspath(sys.argv[1])
tag = sys.argv[2] if len(sys.argv) > 2 else "run"
OUT = os.path.join(HERE, "out", "noise")
os.makedirs(OUT, exist_ok=True)
W, H = 480, 270
inp = os.path.join(HERE, "out", "fg3d", "in.raw")  # written by fg3d_preview.py at 480x270
if not os.path.exists(inp) or os.path.getsize(inp) != W * H * 16:
    raise SystemExit("run tests/fg3d_preview.py <fg3d_render> 480 270 first (it writes the input image)")

SHARP = dict(focus=0.05, fstop=5.6)
LOOKS = {
    "gem": dict(shape=15, size=0.7, depth=0.15, center=(0.4, 0.45), rotX=-35, rotY=15, ior=2.0, dispersion=0.8,
                maxBounces=16, light=(0.2, 0.8, 0.3, 0.1, 6500, 3), **SHARP),
    "radial_prism": dict(shape=2, facets=6, prismAngle=7, clearCentre=0.5, size=1.6, depth=0.15),
    "round_diopter": dict(shape=10, power=6, size=0.7, depth=0.2, center=(0.5, 0.6), dispersion=0.6),
    "crystals_defocused": dict(shape=21, facets=7, size=1.2, depth=0.15, center=(0.3, 0.4), ior=1.7, dispersion=0.7,
                               seed=3, fstop=2.0),
    "water_drops": dict(shape=35, facets=4, prismAngle=8, size=1.4, depth=0.12, **SHARP),
}
QUALITY = {"draft": (4, 8, 0.05), "good": (16, 64, 0.025), "final": (48, 384, 0.008),
           "good+denoise": (16, 64, 0.025, 0.5), "draft+denoise": (4, 8, 0.05, 0.5)}
REF = (512, 512, 0.0)


def fmt(v):
    return ",".join(repr(float(x)) for x in v) if isinstance(v, tuple) else repr(v)


def render(name, kw, q):
    o = os.path.join(OUT, f"{name}_{'_'.join(str(v) for v in q)}.raw")
    args = [f"{k}={fmt(v)}" for k, v in kw.items()] + [f"minSamples={q[0]}", f"maxSamples={q[1]}", f"noise={q[2]}"]
    if len(q) > 3:
        args.append(f"denoise={q[3]}")
    r = subprocess.run([exe, inp, o, str(W), str(H)] + args, capture_output=True, text=True, check=True)
    ms = float(r.stdout.split("render")[1].split("ms")[0])
    return np.fromfile(o, np.float32).reshape(H, W, 4)[::-1], ms


def disp(x):
    x = x[..., :3]
    x = x / (1.0 + 0.15 * np.maximum(x - 1.0, 0.0))
    return np.clip(x, 0, 1) ** (1 / 2.2)


ref_cache = os.path.join(OUT, "ref")
os.makedirs(ref_cache, exist_ok=True)
print(f"{'look':20s} " + " ".join(f"{q:>16s}" for q in QUALITY))
totals = {q: [] for q in QUALITY}
for name, kw in LOOKS.items():
    refp = os.path.join(ref_cache, f"{tag}_{name}.npy")
    if os.path.exists(refp):
        ref = np.load(refp)
    else:
        ref, _ = render(name, kw, REF)
        np.save(refp, ref)
    rd = disp(ref)
    row, crops = [], [rd]
    for qn, q in QUALITY.items():
        img, ms = render(name, kw, q)
        e = float(np.sqrt(np.mean((disp(img) - rd) ** 2)))
        totals[qn].append(e)
        row.append(f"{e:.4f} {ms:6.0f}ms")
        crops.append(disp(img))
    print(f"{name:20s} " + " ".join(f"{c:>16s}" for c in row))
    strip = np.concatenate([c[60:210, 120:360] for c in crops], 1)
    Image.fromarray((strip * 255 + 0.5).astype(np.uint8)).save(os.path.join(OUT, f"{tag}_{name}.png"))
print(f"{'mean RMS':20s} " + " ".join(f"{np.mean(v):>16.4f}" for v in totals.values()))
print("crops (reference | " + " | ".join(QUALITY) + "):", OUT)
