#!/usr/bin/env python3
"""GLM-5.3-Flash's Python CUDA engine (v0.6.6, two ranks) as the Zig engine's oracle: Triton launches and tokens.

Rank 0 runs the prompt set, serial and MTP-drafted, and writes launches.json, jit.json and tokens.json; rank 1
mirrors it and is stopped once rank 0 has written them. The set covers what the native engine must replay: one-row
and drafted windows of every width, prompts past one 2048-row chunk and past the 2,051-token dense limit (the
indexer and sparse attention), and a three-chunk prompt.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
import time
from pathlib import Path

TEXTS = {
    "story": "Write a short story about a lighthouse keeper who finds a message in a bottle.",
    "code": "Write a Python function that merges two sorted lists, with a docstring and three tests.",
    "facts": "Explain how a refrigerator moves heat out of its cabinet, step by step.",
    "json": "Return a JSON object listing three European capitals with their population and one landmark each.",
}
BODIES = {"long": 2900, "huge": 5200}      # tokens of module source: past one prompt chunk and the dense limit; three chunks


def chat(user: str) -> str:
    return f"[gMASK]<sop><|user|>\n{user}<|assistant|>\n<think></think>\n"


def prompts(model: Path) -> dict[str, list[int]]:
    from tokenizers import Tokenizer
    import textwrap

    tok = Tokenizer.from_file(str(model / "tokenizer.json"))
    out = {name: tok.encode(chat(text), add_special_tokens=False).ids for name, text in TEXTS.items()}
    words = Path(textwrap.__file__).read_text()
    for name, size in BODIES.items():
        body = tok.encode(words * (1 + size // max(1, len(words) // 3)), add_special_tokens=False).ids[:size]
        head = tok.encode("[gMASK]<sop><|user|>\nReview this module and list its public functions:\n\n",
                          add_special_tokens=False).ids
        tail = tok.encode("<|assistant|>\n<think></think>\n", add_special_tokens=False).ids
        out[name] = head + body + tail
    return out


def lengths(model: Path, sizes: list[int]) -> dict[str, list[int]]:
    """Prompts of exactly these many tokens (module source), so every chunk remainder class gets launched."""

    from tokenizers import Tokenizer
    import textwrap

    tok = Tokenizer.from_file(str(model / "tokenizer.json"))
    words = Path(textwrap.__file__).read_text()
    body = tok.encode(words * (2 + max(sizes) // max(1, len(words) // 4)), add_special_tokens=False).ids
    return {f"len{n}": body[:n] for n in sizes}


def specialization() -> dict:
    """Each launched JIT function's parameter names and which ones Triton never specializes."""

    from triton.runtime.jit import JITFunction
    import gc

    out = {}
    for fn in gc.get_objects():
        if isinstance(fn, JITFunction):
            name = f"{fn.fn.__module__}.{fn.fn.__qualname__}"
            out[name] = {"params": [p.name for p in fn.params],
                         "do_not_specialize": [p.name for p in fn.params if p.do_not_specialize],
                         "no_align": [p.name for p in fn.params if p.do_not_specialize_on_alignment]}
    return out


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--model", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--rank", type=int, required=True)
    ap.add_argument("--master", required=True)
    ap.add_argument("--port", type=int, default=29551)
    ap.add_argument("--tools", default="", help="folder of triton_aot_manifest.py: record every Triton launch")
    ap.add_argument("--context", type=int, default=65536, help="explicit, so the indexer and sparse attention run")
    ap.add_argument("--max-tokens", type=int, default=128)
    ap.add_argument("--only", default="", help="comma-separated prompt names (default: all)")
    ap.add_argument("--lengths", default="", help="comma-separated prompt lengths in tokens, instead of the prompt set")
    a = ap.parse_args()
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    rec = None
    if a.tools and a.rank == 0:
        sys.path.insert(0, a.tools)
        import triton_aot_manifest as aot

        rec = aot.Recorder().install()
    from tensorfold.families.glm5_next.cuda.engine import GlmEngine

    t0 = time.perf_counter()
    eng = GlmEngine(Path(a.model), rank=a.rank, master=a.master, port=a.port, drafter=None, context=a.context,
                    context_explicit=True)
    print(f"[capture] rank {a.rank} ready in {time.perf_counter() - t0:.1f} s, window {eng.limit}", flush=True)
    if a.rank == 1:
        eng.follow()
        return 0
    ids = lengths(Path(a.model), [int(n) for n in a.lengths.split(",")]) if a.lengths else prompts(Path(a.model))
    if a.only:
        ids = {k: v for k, v in ids.items() if k in a.only.split(",")}
    (out / "prompts.json").write_text(json.dumps(ids) + "\n")
    results = {}
    for name, prompt in ids.items():
        row = {"prompt_len": len(prompt)}
        for mode, draft in (("serial", False), ("mtp", True)):
            got: list[int] = []
            t = time.perf_counter()
            stats = eng.generate(prompt, a.max_tokens, None, got.extend, draft=draft)
            row[mode] = got
            row[mode + "_s"] = round(time.perf_counter() - t, 3)
            row[mode + "_stats"] = {k: v for k, v in stats.items() if isinstance(v, (int, float, str))}
        row["equal"] = row["serial"] == row["mtp"]
        row["sha"] = hashlib.sha256(json.dumps(row["serial"]).encode()).hexdigest()[:12]
        results[name] = row
        print(f"[capture] {name}: {len(prompt)} tokens, serial {row['serial_s']} s, mtp {row['mtp_s']} s, "
              f"equal {row['equal']}, {row['sha']}", flush=True)
    (out / "tokens.json").write_text(json.dumps(results) + "\n")
    if rec is not None:
        rec.dump(out / "launches.json")
        (out / "jit.json").write_text(json.dumps(specialization(), indent=1, sort_keys=True) + "\n")
    print("[capture] done", flush=True)
    os._exit(0)                                  # rank 1 idles in follow(); the launcher stops it


if __name__ == "__main__":
    raise SystemExit(main())
