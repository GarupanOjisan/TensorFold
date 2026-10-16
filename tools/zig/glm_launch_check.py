#!/usr/bin/env python3
"""Print each captured GLM Triton function's variants: runtime params, constexprs, grids and call sites."""

from __future__ import annotations

HELP = """Print, per captured GLM Triton function, each variant's runtime params, constexprs, grids and call sites (host-only, no torch).

Compare with zig/src/families/glm/cuda/glm_triton.zig: launches.json comes from tools/zig/triton_aot_manifest.py's Recorder,
aot.json from the AOT pack; a variant is one specialization (hash) of a function.
"""

import argparse
import json
from collections import defaultdict
from pathlib import Path


def value(v):
    return v["float"] if isinstance(v, dict) and "float" in v else v


def b16_split_k(n: int, k: int) -> int:
    """qmm.b16_split_k, to check every captured _bmm's SK."""
    tiles, steps, sk = -(-n // 64), k // 64, 1
    while sk < 8 and tiles * sk < 192 and steps % (sk * 2) == 0 and steps // (sk * 2) >= 4:
        sk *= 2
    return sk


def main() -> int:
    ap = argparse.ArgumentParser(description=HELP)
    ap.add_argument("--launches", default=str(Path.home() / "tf-native/cap/c2/launches.json"))
    ap.add_argument("--aot", default=str(Path.home() / "tf-native/aot/c2/aot.json"))
    a = ap.parse_args()
    rec = json.loads(Path(a.launches).read_text())
    spec = {k["hash"]: k for k in json.loads(Path(a.aot).read_text())["kernels"]}
    by = defaultdict(list)
    for h, k in rec["kernels"].items():
        by[k["name"]].append(h)
    missing = [h for h in rec["kernels"] if h not in spec]
    problems = 0
    for name in sorted(by):
        print(f"== {name}: {len(by[name])} variant(s)")
        for h in sorted(by[name], key=lambda h: json.dumps(rec["kernels"][h]["constexprs"], sort_keys=True)):
            k = rec["kernels"][h]
            consts = {c: value(v) for c, v in k["constexprs"].items()}
            s = spec.get(h)
            params = " ".join(f"{p['name']}:{p['type']}{'/16' if p['div16'] else ''}" for p in s["params"]) if s else "?"
            print(f"  consts {consts}")
            print(f"    params {params}  (/16 = divisible by 16; absent ints were 1)")
            print(f"    grids  {rec['grids'].get(h, [])}")
            for site in rec["sites"].get(h, []):
                print(f"    site   {site}")
            if name == "_bmm" and b16_split_k(consts["N"], consts["K"]) != consts["SK"]:
                problems += 1
                print(f"    MISMATCH b16_split_k({consts['N']}, {consts['K']}) = {b16_split_k(consts['N'], consts['K'])}")
    print(f"{len(by)} functions, {len(rec['kernels'])} variants launched, {len(spec)} in aot.json, "
          f"{len(missing)} launched but not packed, {problems} formula mismatches")
    return 1 if missing or problems else 0


if __name__ == "__main__":
    raise SystemExit(main())
