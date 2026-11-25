"""Render FacetGlass presets through the CPU harness (same kernel source as the Fuse)."""
import subprocess, sys, os, math
import numpy as np
from PIL import Image, ImageDraw, ImageFont
from skimage import data, transform

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(HERE, "out")
os.makedirs(OUT, exist_ok=True)
W, H = 960, 540


def test_image():
    a = data.astronaut().astype(np.float32) / 255.0  # 512x512
    a = transform.resize(a, (540, 540), anti_aliasing=True)
    img = np.zeros((H, W, 3), np.float32)
    # dark gradient background with a warm window
    yy, xx = np.mgrid[0:H, 0:W]
    img[..., 0] = 0.08 + 0.10 * xx / W
    img[..., 1] = 0.06 + 0.05 * yy / H
    img[..., 2] = 0.12
    img[:, 210:750] = a
    # HDR fairy lights on the right and left (values > 1)
    rng = np.random.default_rng(3)
    for _ in range(40):
        x = rng.integers(20, 200) if rng.random() < 0.5 else rng.integers(770, 940)
        y = rng.integers(20, 520)
        r = rng.integers(2, 4)
        col = np.array([4.0, 3.2, 1.8]) * rng.uniform(0.6, 1.4)
        img[max(0, y - r):y + r, max(0, x - r):x + r] = col
    return img


# UI-level values (degrees, f-stop, 0..1 focus) used by the golden presets.
DEFAULTS = dict(shape=0, facets=1, symmetric=0, samples=32, edgeMode=1, useSource=0, seed=1,
                center=(0.5, 0.5), angle=0.0, size=0.35, inner=0.35, prism=10.0, power=1.0,
                ior=1.5, dispersion=0.3, fstop=4.0, focus=1.0, glassDist=0.05, dfocus=1.0,
                boost=0.5, threshold=0.8, refl=0.0, light=(0.3, 0.75), lightSize=0.15,
                lightInt=0.0, mix=1.0)


def ui(**kw):
    p = dict(DEFAULTS)
    p.update(kw)
    return p


def params(**kw):
    p = ui(**kw)
    zg = p.pop("glassDist")
    fstop = p.pop("fstop")
    focus = p.pop("focus")
    dfocus = p.pop("dfocus")
    # same conversions the Fuse's Lua does
    out = dict(p)
    out["glassDist"] = zg
    out["aperture"] = zg * 0.3 / fstop
    fmap = lambda f: 1.0 + (1.0 / zg - 1.0) * (1.0 - f) ** 2
    out["focusInv"] = fmap(focus)
    out["focusInvD"] = fmap(dfocus)
    out["angle"] = math.radians(p["angle"])
    out["prism"] = math.radians(p["prism"])
    return out


def render(name, img, **kw):
    inp = os.path.join(OUT, "in.raw")
    rgba = np.concatenate([img, np.ones((H, W, 1), np.float32)], -1)[::-1].copy()  # bottom-up like Fusion
    rgba.astype(np.float32).tofile(inp)
    args = []
    for k, v in params(**kw).items():
        args.append(f"{k}={v[0]},{v[1]}" if isinstance(v, tuple) else f"{k}={v}")
    o = os.path.join(OUT, name + ".raw")
    subprocess.run([os.path.join(HERE, "harness"), inp, o, str(W), str(H)] + args, check=True)
    res = np.fromfile(o, np.float32).reshape(H, W, 4)[::-1]
    return res


def to8(x):
    x = x[..., :3] / (1 + 0.15 * np.maximum(x[..., :3] - 1, 0))
    return (np.clip(x, 0, 1) ** (1 / 1.0) * 255).astype(np.uint8)


PRESETS = {
    "00_original": dict(mix=0.0),
    "01_linear_prism_split": dict(shape=0, facets=1, prism=8, angle=0, center=(0.55, 0.5)),
    "02_linear_ghost_repeat": dict(shape=0, facets=3, symmetric=1, prism=4, size=0.25, angle=15),
    "03_radial_prism_6": dict(shape=1, facets=6, prism=6, inner=0.55, size=0.4),
    "04_split_diopter": dict(shape=2, power=2.0, angle=180, center=(0.5, 0.5), focus=0.45, dfocus=1.0, fstop=2.0, samples=64),
    "05_round_diopter_magnify": dict(shape=3, power=14.0, fstop=8, size=0.28, center=(0.5, 0.62), dispersion=0.6),
    "06_crystal": dict(shape=4, facets=4, prism=6, inner=0.5, size=0.5, dispersion=0.8, boost=1.0),
    "07_focus_on_glass": dict(shape=1, facets=5, prism=6, inner=0.45, size=0.4, focus=0.0, fstop=2.0, samples=64),
    "08_reflection_glint": dict(shape=0, facets=2, symmetric=1, prism=6, size=0.3, refl=0.6, lightInt=2.0, light=(0.35, 0.8)),
}

if __name__ == "__main__":
    img = test_image()
    which = sys.argv[1:] or list(PRESETS)
    tiles = []
    for name in which:
        res = render(name, img, **PRESETS[name])
        Image.fromarray(to8(res)).save(os.path.join(OUT, name + ".png"))
        tiles.append((name, to8(res)))
        print("ok", name, float(res[..., :3].mean()))
    if len(tiles) > 1:
        tw, th = W // 2, H // 2
        cols = 3
        rows = math.ceil(len(tiles) / cols)
        sheet = Image.new("RGB", (cols * tw, rows * (th + 22)), (20, 20, 20))
        d = ImageDraw.Draw(sheet)
        for i, (n, t) in enumerate(tiles):
            x, y = (i % cols) * tw, (i // cols) * (th + 22)
            sheet.paste(Image.fromarray(t).resize((tw, th), Image.LANCZOS), (x, y + 22))
            d.text((x + 6, y + 5), n[3:].replace("_", " "), fill=(230, 230, 230))
        sheet.save(os.path.join(OUT, "contact_sheet.png"))
