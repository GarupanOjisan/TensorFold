"""Check that each kernel copy in zig/kernels/cuda matches the Python source lines its first line names.

A copy's first line reads "// Device code of src/tensorfold/<path> (lines A-B, ...)". Lines A-B of that file, without
comment-only lines and ATen/c10/torch includes, with ``namespace {`` named as the copy names it, must be the copy's lines
after the first, compared the same way; what follows them (the namespace close, explicit instantiations) is the copy's own.
The Python checkout is $TF_PYTHON_REPO (default ~/tf-native/py066, the v0.6.6 tree with src/tensorfold).
"""

import os
import re
import sys
from pathlib import Path

HEADER = re.compile(r"^// Device code of (src/tensorfold/\S+) \(lines (\d+)-(\d+),")
# ATen brought in <stdint.h>; a copy may include it in their place
HOST_INCLUDE = re.compile(r"^#include <((ATen|c10|torch)/|stdint\.h>|cstdint>)")
NAMED = re.compile(r"^namespace (\w+) \{$")


def norm(lines: list[str], ns: str | None) -> list[str]:
    out = []
    for line in lines:
        s = line.rstrip()
        if s.lstrip().startswith("//") or HOST_INCLUDE.match(s):
            continue
        if ns and s == "namespace {":
            s = f"namespace {ns} {{"
        out.append(s)
    while out and out[0] == "":
        out.pop(0)
    # blank runs left by dropped comments count as one blank line
    return [s for i, s in enumerate(out) if s or (i and out[i - 1])]


def check(copy: Path, repo: Path) -> str | None:
    lines = copy.read_text().splitlines()
    m = HEADER.match(lines[0]) if lines else None
    if m is None:
        return None
    src, a, b = repo / m[1], int(m[2]), int(m[3])
    if not src.is_file():
        return f"{copy.name}: {src} missing"
    body = lines[1:]
    ns = next((n[1] for n in map(NAMED.match, (s.rstrip() for s in body)) if n), None)
    want = norm(src.read_text().splitlines()[a - 1:b], ns)
    have = norm(body, None)
    while want and want[-1] == "":
        want.pop()
    for i, w in enumerate(want):
        if i >= len(have) or have[i] != w:
            got = have[i] if i < len(have) else "<end of copy>"
            return f"{copy.name}: differs from {m[1]} at normalized line {i + 1}:\n  python: {w!r}\n  copy:   {got!r}"
    return None


def main() -> int:
    kernels = Path(__file__).resolve().parents[2] / "kernels" / "cuda"
    repo = Path(os.environ.get("TF_PYTHON_REPO", Path.home() / "tf-native" / "py066"))
    checked, bad = 0, 0
    for copy in sorted([*kernels.glob("*.cu"), *kernels.glob("*.cuh")]):
        if not HEADER.match(copy.read_text().split("\n", 1)[0]):
            continue
        checked += 1
        err = check(copy, repo)
        if err:
            bad += 1
            print("FAIL", err)
        else:
            print("ok  ", copy.name)
    print(f"{checked - bad}/{checked} copies match {repo}")
    return 1 if bad or checked == 0 else 0


if __name__ == "__main__":
    sys.exit(main())
