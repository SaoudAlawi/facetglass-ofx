"""OFX plugin test: the built FacetGlass.ofx, loaded by the HostSupport test host,
must reproduce the golden renders and handle tiles, render scale, negative row
bytes, the Refract Source clip, wobble from time, quality presets and identity.

Usage:
  python tests/compare_ofx.py <fg_ofxhost> <plugin dir> <fg_fuse_harness> <fg_core_equiv>
The plugin dir is the folder containing FacetGlass.ofx.bundle.
Needs numpy, pillow, scikit-image. Exit code 0 = pass.
"""
import math
import os
import subprocess
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "..", "reference", "kernel-harness"))
import run  # test image, presets, defaults and Lua-equivalent param conversion

host, plugdir, harness, core = (os.path.abspath(a) for a in sys.argv[1:5])
OUT = os.path.join(HERE, "out", "ofx")
os.makedirs(OUT, exist_ok=True)
W, H = run.W, run.H
TOL = 1e-3

# UI name (run.py / Fuse) -> OFX script name
OFX_NAME = dict(shape="shape", facets="facets", symmetric="bothSides", edgeMode="outsideFrame", seed="seed",
                center="center", angle="angle", size="size", inner="clearCentre", prism="prismAngle",
                power="diopterPower", ior="ior", dispersion="chromaticAberration", fstop="fStop", focus="focus",
                glassDist="glassDistance", dfocus="diopterFocus", boost="highlightBoost",
                threshold="highlightThreshold", refl="reflection", light="lightPosition", lightSize="lightSize",
                lightInt="lightIntensity", mix="mix")


def fmt(v):
    return f"{v[0]!r},{v[1]!r}" if isinstance(v, tuple) else repr(v)


def ofx_args(ui, quality=None):
    args = []
    for k, v in ui.items():
        if k in ("useSource", "samples"):
            continue
        args.append(f"{OFX_NAME[k]}={fmt(v)}")
    if quality is None:
        args += ["quality=3", f"samples={ui['samples']}"]  # Custom
    else:
        args.append(f"quality={quality}")
    return args


def core_args(params):
    return [f"{k}={v[0]},{v[1]}" if isinstance(v, tuple) else f"{k}={v}" for k, v in params.items()]


def save(path, img):
    np.ascontiguousarray(img, np.float32).tofile(path)


def load(path, w, h):
    return np.fromfile(path, np.float32).reshape(h, w, 4)


def run_host(name, inp, w, h, args, extra=()):
    o = os.path.join(OUT, name + ".raw")
    cmd = [host, "--plugins", plugdir, "--in", inp, str(w), str(h), "--out", o] + list(extra) + args
    r = subprocess.run(cmd, capture_output=True, text=True)
    if r.returncode != 0:
        print(r.stdout, r.stderr)
        raise SystemExit(f"host failed on {name}")
    info = {}
    for line in r.stdout.splitlines():
        if ": " in line:
            k, v = line.split(": ", 1)
            info[k] = v
    return load(o, w, h), info


def run_ref(exe, name, inp, w, h, params):
    o = os.path.join(OUT, name + ".ref.raw")
    subprocess.run([exe, inp, o, str(w), str(h)] + core_args(params), check=True)
    return load(o, w, h)


def wobble(t, seed, amount, speed):  # same as the Fuse's Lua wobble()
    if amount <= 0:
        return 0.0, 0.0, 0.0
    s = seed * 1.7319
    tt = t * speed / 24.0
    ox = amount * 0.03 * (math.sin(1.31 * tt + s) + 0.5 * math.sin(2.97 * tt + 2.1 * s))
    oy = amount * 0.03 * (math.sin(1.07 * tt + 3.3 * s) + 0.5 * math.sin(2.41 * tt + 0.7 * s))
    oa = amount * 6.0 * (math.sin(0.83 * tt + 5.1 * s) + 0.4 * math.sin(2.13 * tt + 1.3 * s))
    return ox, oy, oa


results = []


def report(name, diff, limit, info=None, extra_ok=True, note=""):
    ok = diff <= limit and extra_ok
    if info is not None and "roi_full" in info:
        ok = ok and info["roi_full"] == "1" and info.get("unwritten") == "0"
        note = (note + f" roi_full={info['roi_full']} unwritten={info.get('unwritten')}").strip()
    results.append(ok)
    print(f"{'ok  ' if ok else 'FAIL'} {name:34s} max abs diff {diff:.2e} (limit {limit:.0e}) {note}")


# ---------------------------------------------------------------- inputs
img = run.test_image()
rgba = np.concatenate([img, np.ones((H, W, 1), np.float32)], -1)[::-1].copy()  # bottom-up (OFX y-up)
inp = os.path.join(OUT, "in.raw")
save(inp, rgba)

# ---------------------------------------------------------------- 1. golden presets
golden = {}
for name, kw in run.PRESETS.items():
    ui = run.ui(**kw)
    ref = run_ref(harness, name, inp, W, H, run.params(**kw))
    out, info = run_host(name, inp, W, H, ofx_args(ui))
    golden[name] = out
    if kw.get("mix", 1.0) == 0.0:
        ident = info.get("identity") == "Source"
        report(f"golden {name}", float(np.abs(out - ref).max()), TOL, extra_ok=ident,
               note=f"identity={info.get('identity')}")
    else:
        report(f"golden {name}", float(np.abs(out - ref).max()), TOL, info)

# ---------------------------------------------------------------- 2. tiles
kw = run.PRESETS["03_radial_prism_6"]
out, info = run_host("tiles3x3", inp, W, H, ofx_args(run.ui(**kw)), ["--tiles", "3"])
report("tiles 3x3 == full frame", float(np.abs(out - golden["03_radial_prism_6"]).max()), 0.0, info,
       extra_ok=info.get("tiles") == "9", note=f"tiles={info.get('tiles')}")

# ---------------------------------------------------------------- 3. negative row bytes
kw = run.PRESETS["06_crystal"]
out, info = run_host("negstride", inp, W, H, ofx_args(run.ui(**kw)), ["--negstride"])
report("negative rowBytes == normal", float(np.abs(out - golden["06_crystal"]).max()), 0.0, info)

# ---------------------------------------------------------------- 4. render scale 0.5
half = rgba.reshape(H // 2, 2, W // 2, 2, 4).mean(axis=(1, 3)).astype(np.float32)
inp_half = os.path.join(OUT, "in_half.raw")
save(inp_half, half)
for name in ("01_linear_prism_split", "05_round_diopter_magnify"):
    kw = run.PRESETS[name]
    ref = run_ref(core, name + "_half", inp_half, W // 2, H // 2, run.params(**kw))
    out, info = run_host(name + "_half", inp_half, W // 2, H // 2, ofx_args(run.ui(**kw)), ["--scale", "0.5"])
    report(f"render scale 0.5 {name[3:]}", float(np.abs(out - ref).max()), TOL, info)

# ---------------------------------------------------------------- 5. quality presets
for q, n in ((0, 8), (2, 64)):
    kw = dict(run.PRESETS["02_linear_ghost_repeat"], samples=n)
    ref = run_ref(core, f"quality{q}", inp, W, H, run.params(**kw))
    out, info = run_host(f"quality{q}", inp, W, H, ofx_args(run.ui(**kw), quality=q))
    report(f"quality preset {q} ({n} samples)", float(np.abs(out - ref).max()), TOL, info)

# ---------------------------------------------------------------- 6. wobble from time
t, seed, amount, speed = 37.0, 3, 0.6, 1.5
kw = dict(run.PRESETS["01_linear_prism_split"], seed=seed)
ox, oy, oa = wobble(t, seed, amount, speed)
ui = run.ui(**kw)
refkw = dict(kw, center=(ui["center"][0] + ox, ui["center"][1] + oy), angle=ui["angle"] + oa)
ref = run_ref(core, "wobble", inp, W, H, run.params(**refkw))
out, info = run_host("wobble", inp, W, H, ofx_args(ui) + [f"wobble={amount}", f"wobbleSpeed={speed}"],
                     ["--time", str(t)])
still = run_host("wobble_t0", inp, W, H, ofx_args(ui) + [f"wobble={amount}", f"wobbleSpeed={speed}"])[0]
moved = float(np.abs(out - still).max())
report("wobble at frame 37", float(np.abs(out - ref).max()), TOL, info, extra_ok=moved > 0.05,
       note=f"(moves vs frame 0: {moved:.2f})")

# ---------------------------------------------------------------- 7. Refract Source
kw = run.PRESETS["03_radial_prism_6"]
args = ofx_args(run.ui(**kw))
out, info = run_host("refract_same", inp, W, H, args, ["--refract", inp, str(W), str(H)])
report("refract = source (General ctx)", float(np.abs(out - golden["03_radial_prism_6"]).max()), 0.0, info)


def gradient(w, h):
    yy, xx = np.mgrid[0:h, 0:w].astype(np.float32)
    g = np.zeros((h, w, 4), np.float32)
    g[..., 0] = (xx + 0.5) / w
    g[..., 1] = (yy + 0.5) / h
    g[..., 3] = 1.0
    return g


g1, g2 = os.path.join(OUT, "grad1x.raw"), os.path.join(OUT, "grad2x.raw")
save(g1, gradient(W, H))
save(g2, gradient(2 * W, 2 * H))
o1, i1 = run_host("refract_grad1x", inp, W, H, args, ["--refract", g1, str(W), str(H)])
o2, i2 = run_host("refract_grad2x", inp, W, H, args, ["--refract", g2, str(2 * W), str(2 * H)])
glass = np.abs(o1 - golden["03_radial_prism_6"]).max(-1) > 1e-3  # pixels that see the refract source
report("refract 2x res == 1x res", float(np.abs(o2 - o1).max()), 2e-3, i2,
       extra_ok=glass.mean() > 0.05, note=f"(glass covers {glass.mean() * 100:.0f}%)")

# ---------------------------------------------------------------- summary
passed = all(results)
print("PASS" if passed else "FAIL", f"{sum(results)}/{len(results)}")
sys.exit(0 if passed else 1)
