"""OFX plugin test (3D core): the built FacetGlass.ofx, loaded by the HostSupport
test host, must reproduce the standalone CPU renderer (same core and host library)
exactly, and handle tiles, render scale, negative row bytes, the Refract Source
clip, quality presets, wobble from time, lights, colour spaces and identity.

Usage:
  python tests/compare_ofx.py <fg_ofxhost> <plugin dir> <fg3d_render> [--metal]
--metal (macOS) runs every render through the plugin's Metal path, as Resolve does.
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
HOST_EXTRA = [a for a in sys.argv[4:] if a == "--metal"]
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
           roughness=("roughness", "roughness"), innerReflect=("innerReflect", "innerReflections"), glassType=("glassType", "glassType"), lensMount=("lensMount", "mount"), tint=("tint", "tint"), tintDensity=("tintDensity", "tintDensity"), bevelWidth=("bevelWidth", "bevelWidth"),
           bevelAngle=("bevelAngle", "bevelAngle"), maxBounces=("maxBounces", "bounceLimit"),
           fstop=("fstop", "fStop"), focus=("focus", "focus"), glassFocus=("glassFocus", "glassFocus"),
           world=("world", "outsideFrame"), worldBlur=("worldBlur", "worldBlur"), feather=("feather", "feather"),
           wobble=("wobble", "wobble"), wobbleSpeed=("wobbleSpeed", "wobbleSpeed"), boost=("boost", "highlightBoost"),
           threshold=("threshold", "highlightThreshold"), mix=("mix", "mix"),
           colorSpace=("colorSpace", "inputColorSpace"), denoise=("denoise", "denoise"))
BASE = dict(shape=15, facets=3, bothSides=0, prismAngle=6.0, clearCentre=0.4, power=4.0, thickness=0.3, detail=1,
            seed=1, center=(0.5, 0.5), depth=0.15, rotX=0.0, rotY=0.0, rotZ=0.0, size=0.7, ior=1.55,
            dispersion=0.4, reflection=1.0, roughness=0.0, innerReflect=1.0, glassType=0, lensMount=0, tint=(1.0, 1.0, 1.0), tintDensity=1.0, bevelWidth=0.02, bevelAngle=20.0, maxBounces=12,
            fstop=4.0, focus=1.0, glassFocus=1.0, world=0, worldBlur=4.0, feather=0.05, wobble=0.0,
            wobbleSpeed=1.0, boost=0.5, threshold=1.0, mix=1.0, colorSpace=0, denoise=0.0)
SAMPLES = dict(minSamples=8, maxSamples=24, noise=0.03)
QUALITY = {0: (4, 8, 0.05), 1: (16, 64, 0.025), 2: (48, 384, 0.008)}

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
    "ghost_plate": dict(shape=0, facets=3, bothSides=1, size=0.9, depth=0.12, rotZ=12, innerReflect=3.0),
    "gem_no_inner": dict(rotX=-35, rotY=15, ior=2.0, focus=0.05, fstop=5.6, innerReflect=0.0),
    "diamond_amber": dict(rotX=-35, rotY=15, focus=0.05, fstop=5.6, glassType=8, tint=(1.0, 0.72, 0.35),
                          tintDensity=1.5),
    "emerald_crystals": dict(shape=21, facets=7, size=1.2, center=(0.3, 0.4), seed=3, glassType=2,
                             tint=(0.3, 0.85, 0.5), dispersion=1.0),
    "lens_row_ghost": dict(lensMount=1, shape=42, facets=7, prismAngle=1.5, size=1.1, thickness=0.2, fstop=2.8),
    "lens_circular_ghost": dict(lensMount=1, shape=2, facets=6, prismAngle=3, clearCentre=0.4, size=1.1, fstop=2.8,
                                denoise=0.5),
    "gem_denoised": dict(rotX=-35, rotY=15, ior=2.0, focus=0.05, fstop=5.6, denoise=0.5),
    "crystals_denoised": dict(shape=21, facets=7, size=1.2, center=(0.3, 0.4), fstop=2.0, dispersion=0.7, seed=3,
                              denoise=1.0),
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
    r = subprocess.run([host, "--plugins", plugdir, "--in", inp, str(w), str(h), "--out", o] + HOST_EXTRA + list(extra) + args,
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
ui = dict(BASE, **LOOKS["crystals_denoised"])
out, info = run_host("tiles_denoised", inp, W, H, ofx_args(ui, S), ["--tiles", "4"])
report("tiles 4x4 == full frame (denoised)", out, full["crystals_denoised"], info, extra_ok=info.get("tiles") == "16")
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
# Look presets: picking one in the plugin == the renderer's look=N (several, incl. On Lens).
for n in (1, 3, 5, 11):
    ui = dict(BASE)
    ref = run_ref(f"look{n}", inp, W, H, ref_args(ui, S) + [f"look={n}"])
    out, info = run_host(f"look{n}", inp, W, H, ofx_args(ui, S), ["--look", str(n)])
    report(f"look preset {n}", out, ref, info)
# Playback cache: the 2nd identical render records (and outputs the replay), the 3rd
# replays; both must equal the renderer's record-then-replay (cache=1).
for name, reps in (("halo", 2), ("halo", 3), ("echo_denoised", 3), ("gem_tiles", 3)):
    ui = dict(BASE, **(LOOKS["gem"] if name == "gem_tiles" else {}))
    extra_ref = [] if name == "gem_tiles" else [f"look={1 if name == 'halo' else 3}"]
    extra_ofx = [] if name == "gem_tiles" else ["--look", "1" if name == "halo" else "3"]
    if name == "echo_denoised":
        ui["denoise"] = 0.5
    ref = run_ref(f"cache_{name}_{reps}", inp, W, H, ref_args(ui, S) + extra_ref + ["cache=1"])
    out, info = run_host(f"cache_{name}_{reps}", inp, W, H, ofx_args(ui, S) + ["playbackCache=2"],
                         extra_ofx + ["--repeat", str(reps)] + (["--tiles", "1"] if name != "gem_tiles" else []))
    report(f"playback cache {name} x{reps}", out, ref, info)
# Preview Quality: interactive renders use Draft, final renders keep Final.
ui = dict(BASE, **LOOKS["faceted_plate"])
ref = run_ref("preview_draft", inp, W, H, ref_args(ui, QUALITY[0]))
out, info = run_host("preview_draft", inp, W, H, ofx_args(ui, quality=2) + ["previewQuality=1"], ["--interactive"])
report("preview quality (interactive)", out, ref, info)

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
