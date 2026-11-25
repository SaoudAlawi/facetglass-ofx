"""Golden test: the portable core (src/core) must match the Fuse v1 kernel.

Usage:  python tests/compare_core.py <path to fg_fuse_harness> <path to fg_core_equiv>
Needs numpy, pillow, scikit-image. Exit code 0 = pass.
"""
import os, sys, subprocess
import numpy as np
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "..", "reference", "kernel-harness"))
import run  # test image, presets and Lua-equivalent param conversion

harness, core = sys.argv[1], sys.argv[2]
out = os.path.join(HERE, "out"); os.makedirs(out, exist_ok=True)
W, H = run.W, run.H
img = run.test_image()
rgba = np.concatenate([img, np.ones((H, W, 1), np.float32)], -1)[::-1].copy()
inp = os.path.join(out, "in.raw"); rgba.tofile(inp)
worst = 0.0
for name, kw in run.PRESETS.items():
    args = [f"{k}={v[0]},{v[1]}" if isinstance(v, tuple) else f"{k}={v}" for k, v in run.params(**kw).items()]
    a_p, b_p = os.path.join(out, "a.raw"), os.path.join(out, "b.raw")
    subprocess.run([harness, inp, a_p, str(W), str(H)] + args, check=True)
    subprocess.run([core, inp, b_p, str(W), str(H)] + args, check=True)
    d = float(np.abs(np.fromfile(a_p, np.float32) - np.fromfile(b_p, np.float32)).max())
    worst = max(worst, d)
    print(f"{name:28s} max abs diff {d:.2e}")
print("PASS" if worst < 1e-3 else "FAIL", worst)
sys.exit(0 if worst < 1e-3 else 1)
