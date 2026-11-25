"""OFX plugin test (3D core): the built FacetGlass.ofx, loaded by the HostSupport
test host, must reproduce the standalone CPU renderer (same core and host library)
exactly, and handle tiles, render scale, negative row bytes, the Refract Source
clip, quality presets, wobble from time, lights, colour spaces and identity.

Usage:
  python tests/compare_ofx.py <fg_ofxhost> <plugin dir> <fg3d_render>
The plugin dir is the folder containing FacetGlass.ofx.bundle.
Exit code 0 = pass.
"""
import os
import subprocess
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "..", "reference", "kernel-harness"))
import run  # test image

host, plugdir, render = (os.path.abspath(a) for a in sys.argv[1:4])
OUT = os.path.join(HERE, "out", "ofx3d")
os.makedirs(OUT, exist_ok=True)
W, H = 480, 270

# Full UI state: name -> (fg3d_render key, OFX param name). Every look sets all of
# them explicitly, so the plugin and renderer defaults cannot drift apart unseen.
MAP = dict(shape=("shape", "shape"), facets=("facets", "facets"), bothSides=("bothSides", "bothSides"),
           prismAngle=("prismAngle", "prismAngle"), clearCentre=("clearCentre", "clearCentre"),
           power=("power", "diopterPower"), thickness=("thickness", "thickness"), detail=("detail", "detail"),
           seed=("seed", "seed"), center=("center", "center"), depth=("depth", "depth"), rotX=("rotX", "tilt"),
           rotY=("rotY", "turn"), rotZ=("rotZ", "roll"), size=("size", "size"), ior=("ior", "ior"),
           dispersion=("dispersion", "chromaticAberration"), reflection=("reflection", "reflection"),
           roughness=("roughness", "roughness"), bevelWidth=("bevelWidth", "bevelWidth"),
           bevelAngle=("bevelAngle", "bevelAngle"), maxBounces=("maxBounces", "bounceLimit"),
           fstop=("fstop", "fStop"), focus=("focus", "focus"), glassFocus=("glassFocus", "glassFocus"),
           world=("world", "outsideFrame"), worldBlur=("worldBlur", "worldBlur"), feather=("feather", "feather"),
           wobble=("wobble", "wobble"), wobbleSpeed=("wobbleSpeed", "wobbleSpeed"), boost=("boost", "highlightBoost"),
           threshold=("threshold", "highlightThreshold"), mix=("mix", "mix"),
           colorSpace=("colorSpace", "inputColorSpace"))
BASE = dict(shape=15, facets=3, bothSides=0, prismAngle=6.0, clearCentre=0.4, power=4.0, thickness=0.3, detail=1,
            seed=1, center=(0.5, 0.5), depth=0.15, rotX=0.0, rotY=0.0, rotZ=0.0, size=0.7, ior=1.55,
            dispersion=0.4, reflection=1.0, roughness=0.0, bevelWidth=0.02, bevelAngle=20.0, maxBounces=12,
            fstop=4.0, focus=1.0, glassFocus=1.0, world=0, worldBlur=4.0, feather=0.05, wobble=0.0,
            wobbleSpeed=1.0, boost=0.5, threshold=1.0, mix=1.0, colorSpace=0)
SAMPLES = dict(minSamples=8, maxSamples=24, noise=0.03)
QUALITY = {0: (4, 8, 0.05), 1: (12, 48, 0.03), 2: (48, 384, 0.008)}

LOOKS = {
    "gem": dict(rotX=-35, rotY=15, ior=2.0, focus=0.05, fstop=5.6),
    "faceted_plate": dict(shape=0, facets=3, bothSides=1, size=0.9, depth=0.12, rotZ=12),
    "radial_prism": dict(shape=2, facets=6, prismAngle=7, clearCentre=0.5, size=1.6),
    "round_diopter": dict(shape=10, power=6, center=(0.5, 0.6), dispersion=0.6, depth=0.2),
    "split_diopter": dict(shape=11, power=4, size=2.5, rotZ=180, focus=0.5, glassFocus=1.0, fstop=2.0),
    "crystal_cluster": dict(shape=21, facets=7, size=1.2, center=(0.3, 0.4), ior=1.7, seed=3, roughness=0.2,
                            focus=0.05, fstop=5.6),
    "box_slog3": dict(shape=23, size=0.6, depth=0.25, rotX=25, rotY=35, focus=0.0, fstop=8, colorSpace=3),
    "sphere_transparent": dict(shape=27, detail=2, size=0.8, depth=0.2, world=5, worldBlur=2.0),
}


def fmt(v):
    return ",".join(repr(float(x)) for x in v) if isinstance(v, tuple) else repr(v)


def save(path, img):
    np.ascontiguousarray(img, np.float32).tofile(path)


def load(path, w, h):
    return np.fromfile(path, np.float32).reshape(h, w, 4)


def ref_args(ui, samples, lights=(), time=0.0):
    a = [f"{MAP[k][0]}={fmt(v)}" for k, v in ui.items()]
    a += [f"minSamples={samples[0]}", f"maxSamples={samples[1]}", f"noise={samples[2]}", f"time={time}"]
    a += ["light=" + ",".join(repr(float(x)) for x in L) for L in lights]
    return a


def ofx_args(ui, samples=None, quality=None, lights=()):
    a = [f"{MAP[k][1]}={fmt(v)}" for k, v in ui.items()]
    if quality is None:
        a += ["quality=3", f"minSamples={samples[0]}", f"maxSamples={samples[1]}", f"noiseThreshold={samples[2]}"]
    else:
        a.append(f"quality={quality}")
    for i, L in enumerate(lights):
        n = i + 1
        a += [f"light{n}Position={L[0]!r},{L[1]!r}", f"light{n}Depth={L[2]!r}", f"light{n}Size={L[3]!r}",
              f"light{n}Kelvin={L[4]!r}", f"light{n}Intensity={L[5]!r}"]
    return a


def run_ref(name, inp, w, h, args):
    o = os.path.join(OUT, name + ".ref.raw")
    r = subprocess.run([render, inp, o, str(w), str(h)] + args, capture_output=True, text=True)
    if r.returncode != 0 or "unknown" in r.stderr:
        print(r.stdout, r.stderr)
        raise SystemExit(f"renderer failed on {name}")
    return load(o, w, h)


def run_host(name, inp, w, h, args, extra=()):
    o = os.path.join(OUT, name + ".raw")
    r = subprocess.run([host, "--plugins", plugdir, "--in", inp, str(w), str(h), "--out", o] + list(extra) + args,
                       capture_output=True, text=True)
    if r.returncode != 0:
        print(r.stdout, r.stderr)
        raise SystemExit(f"host failed on {name}")
    info = dict(line.split(": ", 1) for line in r.stdout.splitlines() if ": " in line)
    return load(o, w, h), info


results = []


def report(name, out, ref, info, limit=0.0, extra_ok=True, note=""):
    diff = float(np.abs(out - ref).max())
    ok = diff <= limit and extra_ok
    if "roi_full" in info:
        ok = ok and info["roi_full"] == "1" and info.get("unwritten") == "0"
        note = (note + f" roi_full={info['roi_full']} unwritten={info.get('unwritten')}").strip()
    results.append(ok)
    print(f"{'ok  ' if ok else 'FAIL'} {name:34s} max abs diff {diff:.2e} {note}")


# ---------------------------------------------------------------- inputs
img = run.test_image()[::2, ::2]  # 480x270
rgba = np.concatenate([img, np.ones((H, W, 1), np.float32)], -1)[::-1].copy()
inp = os.path.join(OUT, "in.raw")
save(inp, rgba)

# 1. Looks: plugin == standalone renderer.
full = {}
for name, kw in LOOKS.items():
    ui = dict(BASE, **kw)
    ref = run_ref(name, inp, W, H, ref_args(ui, (SAMPLES["minSamples"], SAMPLES["maxSamples"], SAMPLES["noise"])))
    out, info = run_host(name, inp, W, H, ofx_args(ui, (SAMPLES["minSamples"], SAMPLES["maxSamples"], SAMPLES["noise"])))
    full[name] = out
    moved = float(np.abs(out - rgba).max())
    report(f"look {name}", out, ref, info, extra_ok=moved > 0.05, note=f"(changes image: {moved:.2f})")

S = (SAMPLES["minSamples"], SAMPLES["maxSamples"], SAMPLES["noise"])

# 2. Tiles, negative row bytes.
ui = dict(BASE, **LOOKS["radial_prism"])
out, info = run_host("tiles", inp, W, H, ofx_args(ui, S), ["--tiles", "3"])
report("tiles 3x3 == full frame", out, full["radial_prism"], info, extra_ok=info.get("tiles") == "9")
ui = dict(BASE, **LOOKS["gem"])
out, info = run_host("negstride", inp, W, H, ofx_args(ui, S), ["--negstride"])
report("negative rowBytes == normal", out, full["gem"], info)

# 3. Render scale 0.5 (host gives a half-size image) == renderer at half size.
half = rgba.reshape(H // 2, 2, W // 2, 2, 4).mean(axis=(1, 3)).astype(np.float32)
inp_half = os.path.join(OUT, "in_half.raw")
save(inp_half, half)
for name in ("gem", "round_diopter"):
    ui = dict(BASE, **LOOKS[name])
    ref = run_ref(name + "_half", inp_half, W // 2, H // 2, ref_args(ui, S))
    out, info = run_host(name + "_half", inp_half, W // 2, H // 2, ofx_args(ui, S), ["--scale", "0.5"])
    report(f"render scale 0.5 {name}", out, ref, info)

# 4. Quality presets.
for q in (0, 1):
    ui = dict(BASE, **LOOKS["faceted_plate"])
    ref = run_ref(f"quality{q}", inp, W, H, ref_args(ui, QUALITY[q]))
    out, info = run_host(f"quality{q}", inp, W, H, ofx_args(ui, quality=q))
    report(f"quality preset {q}", out, ref, info)

# 5. Wobble from time, and lights.
ui = dict(BASE, **LOOKS["gem"], wobble=0.8, wobbleSpeed=1.5)
lights = [(0.2, 0.8, 0.3, 0.1, 6500.0, 3.0), (0.7, 0.3, 0.5, 0.2, 3200.0, 1.5)]
ref = run_ref("wobble_lights", inp, W, H, ref_args(ui, S, lights, time=37.0))
out, info = run_host("wobble_lights", inp, W, H, ofx_args(ui, S, lights=lights), ["--time", "37"])
still = run_host("wobble_t0", inp, W, H, ofx_args(ui, S, lights=lights))[0]
report("wobble at frame 37 + 2 lights", out, ref, info, extra_ok=float(np.abs(out - still).max()) > 0.05)

# 6. Refract Source: same image == no refract; a different image changes only the glass.
ui = dict(BASE, **LOOKS["gem"])
out, info = run_host("refract_same", inp, W, H, ofx_args(ui, S), ["--refract", inp, str(W), str(H)])
report("refract = source (General ctx)", out, full["gem"], info)
red = np.zeros((2 * H, 2 * W, 4), np.float32)
red[..., 0] = 1.0
red[..., 3] = 1.0
inp_red = os.path.join(OUT, "red2x.raw")
save(inp_red, red)
out, info = run_host("refract_red", inp, W, H, ofx_args(ui, S), ["--refract", inp_red, str(2 * W), str(2 * H)])
changed = np.abs(out - full["gem"]).max(-1) > 1e-3
report("refract 2x red: only glass changes", out, out, info, extra_ok=0.02 < changed.mean() < 0.6,
       note=f"(glass {changed.mean() * 100:.0f}% of frame)")

# 7. Custom OBJ (a small faceted solid written here, quads and a negative index).
obj = os.path.join(OUT, "test_shape.obj")
with open(obj, "w") as f:
    f.write("# test\nv 0 0 1.2\nv 1 0 0\nv 0 1 0\nv -1 0 0\nv 0 -1 0\nv 0 0 -0.8\n")
    f.write("f 1 2 3\nf 1 3 4\nf 1 4 5\nf 1 5 2\nf 6 3 2\nf 6 4 3\nf 6 5 4\nf -1 2/1/1 5//1\n")
ui = dict(BASE, shape=41, rotX=20, rotY=30, focus=0.05, fstop=5.6)
ref = run_ref("obj", inp, W, H, ref_args(ui, S) + [f"obj={obj}", "smoothAngle=30"])
out, info = run_host("obj", inp, W, H, ofx_args(ui, S) + [f"objFile:={obj}", "smoothAngle=30"])
report("custom OBJ", out, ref, info, extra_ok=float(np.abs(out - rgba).max()) > 0.05)

# 8. A host without the message suite V2 (like Resolve) must still render.
ui = dict(BASE, **LOOKS["gem"])
out, info = run_host("minimal_host", inp, W, H, ofx_args(ui, S), ["--minimal-host"])
report("host without message suite", out, full["gem"], info)

# 9. Identity at mix 0.
out, info = run_host("identity", inp, W, H, ofx_args(dict(BASE, mix=0.0), S))
report("mix 0 is identity", out, rgba, info, extra_ok=info.get("identity") == "Source")

passed = all(results)
print("PASS" if passed else "FAIL", f"{sum(results)}/{len(results)}")
sys.exit(0 if passed else 1)
