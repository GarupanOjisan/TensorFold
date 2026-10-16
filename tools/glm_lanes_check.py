"""Kept prompt states across lanes, against a running OpenAI-compatible server (GLM CUDA lanes, docs/glm-lanes-design.md).

Two long conversations A and B (system prompts past prompt_cache's 4,096-token floor) take turns: A1, B1, A2, B2.
Then C, with A's system prompt and its own question, starts while A3 decodes, so A's lane is busy and C copies its
rows. Every request is greedy and 64 tokens long; each reply's token-id hash (``tensorfold.token_sha``) and cached
prompt tokens (``usage.prompt_tokens_details.cached_tokens``) are recorded.

  # the reference: one lane, no kept states (every request prefills from 0)
  python3 tools/glm_lanes_check.py http://127.0.0.1:8000 MODEL --save ref.json
  # the server under test (--parallel 2 or more): same replies, and A2, B2 and C resume from kept states
  python3 tools/glm_lanes_check.py http://127.0.0.1:8000 MODEL --compare ref.json

With one lane and kept states on, A2 and B2 report 0 cached tokens (B1 overwrote A1's rows): the check tells the
two apart. Concurrent replies against lone ones: tools/bench_concurrent.py --alone. Standard library only.
"""

import argparse
import json
import sys
import threading
import time
import urllib.request


def system_prompt(tag: str) -> str:
    # about 6,000 tokens of distinct numbered facts
    return "\n".join(f"Fact {tag}-{i}: the {tag} registry entry {i} maps to code {(i * 7919) % 10007}." for i in range(400))


def ask(base: str, model: str, messages: list[dict]) -> dict:
    body = {"model": model, "messages": messages, "max_tokens": 64, "temperature": 0,
            "chat_template_kwargs": {"enable_thinking": False}}
    req = urllib.request.Request(base + "/v1/chat/completions", data=json.dumps(body).encode(),
                                 headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=3600) as resp:
        out = json.load(resp)
    usage = out.get("usage") or {}
    return {"text": out["choices"][0]["message"].get("content") or "",
            "token_sha": (out.get("tensorfold") or {}).get("token_sha"),
            "prompt_tokens": usage.get("prompt_tokens", 0),
            "cached": (usage.get("prompt_tokens_details") or {}).get("cached_tokens", 0)}


def turn(history: list[dict], reply: dict, question: str) -> list[dict]:
    return history + [{"role": "assistant", "content": reply["text"]}, {"role": "user", "content": question}]


def run(base: str, model: str) -> dict:
    sa, sb = system_prompt("alpha"), system_prompt("beta")
    a1 = [{"role": "system", "content": sa}, {"role": "user", "content": "What code does alpha entry 17 map to?"}]
    b1 = [{"role": "system", "content": sb}, {"role": "user", "content": "What code does beta entry 23 map to?"}]
    r = {}
    r["A1"] = ask(base, model, a1)
    r["B1"] = ask(base, model, b1)
    a2 = turn(a1, r["A1"], "And alpha entry 301?")
    b2 = turn(b1, r["B1"], "And beta entry 302?")
    r["A2"] = ask(base, model, a2)
    r["B2"] = ask(base, model, b2)
    a3 = turn(a2, r["A2"], "List alpha entries 40 to 45 with their codes.")
    c1 = [{"role": "system", "content": sa}, {"role": "user", "content": "What code does alpha entry 99 map to?"}]
    out: dict = {}
    t = threading.Thread(target=lambda: out.update(A3=ask(base, model, a3)))
    t.start()
    time.sleep(2)  # A3 is past its prefill (it resumes from A2's state) and decoding when C arrives
    r["C"] = ask(base, model, c1)
    t.join()
    r["A3"] = out["A3"]
    return r


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("base")
    p.add_argument("model")
    g = p.add_mutually_exclusive_group(required=True)
    g.add_argument("--save", help="write the replies here (the reference run)")
    g.add_argument("--compare", help="check the replies against a reference run's file")
    o = p.parse_args()
    r = run(o.base.rstrip("/"), o.model)
    for name, x in r.items():
        print(f"{name}: {x['prompt_tokens']} prompt tokens, {x['cached']} cached, token_sha {x['token_sha']}")
    if o.save:
        with open(o.save, "w") as f:
            json.dump(r, f, indent=1)
        return 0
    with open(o.compare) as f:
        ref = json.load(f)
    bad = [n for n in r if r[n]["token_sha"] != ref[n]["token_sha"]]
    cold = [n for n in ("A2", "B2", "C") if r[n]["cached"] == 0]
    for n in bad:
        print(f"FAIL {n}: reply differs from the reference")
    for n in cold:
        print(f"FAIL {n}: no cached prompt tokens (its kept state was not resumed)")
    if not bad and not cold:
        print("PASS: replies equal the reference; A2, B2 and C resumed from kept states")
    return 1 if bad or cold else 0


if __name__ == "__main__":
    sys.exit(main())
