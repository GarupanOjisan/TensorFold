"""v1.8's TF_GLM_DENSE=q4 load of a few tensors of one rank, dumped as raw bytes for the Zig port's `tf-glm qdump`:
qdump.py MODEL_DIR RANK OUT_DIR [NAME ...]. Run on the GPU in the v1.8 image (PYTHONPATH=py18/src): the quantizer's
float sums depend on torch's CUDA reduction order. Each tensor NAME gets NAME.w, NAME.s, NAME.b (Q4: the tiled int32
words, bf16 scales and biases) or NAME.q8, NAME.sc (F8: e4m3 bytes, fp32 scales); a stack is the loader's
(qmm.stack_dense_q4)."""
import sys
from pathlib import Path

import torch

from tensorfold.families.glm5_next.cuda import qmm
from tensorfold.families.glm5_next.cuda.split import RankReader

PREFIX = "model.language_model."
SA = "self_attn."
SETS = {
    "l0_kda_proj": [f"layers.0.{SA}{n}_proj" for n in ("q", "k", "v", "f_a", "g_a", "b")],
    "l0_kda_fb": [f"layers.0.{SA}f_b_proj"],
    "l0_kda_o": [f"layers.0.{SA}o_proj"],
    "l0_mlp_gu": ["layers.0.mlp.gate_proj", "layers.0.mlp.up_proj"],
    "l0_mlp_down": ["layers.0.mlp.down_proj"],
    "l3_dsa_proj": [f"layers.3.{SA}q_a_proj", f"layers.3.{SA}kv_a_proj_with_mqa"],
    "l3_dsa_qb": [f"layers.3.{SA}q_b_proj"],
    "l3_dsa_o": [f"layers.3.{SA}o_proj"],
    "l3_ix_kw": [f"layers.3.{SA}indexer.wk", f"layers.3.{SA}indexer.weights_proj"],
    "l3_ix_qb": [f"layers.3.{SA}indexer.wq_b"],
    "l3_sh_gu": ["layers.3.mlp.shared_experts.gate_proj", "layers.3.mlp.shared_experts.up_proj"],
    "l3_sh_down": ["layers.3.mlp.shared_experts.down_proj"],
    "mtp_eh": ["layers.45.eh_proj"],
}


def main() -> None:
    model, rank, out = Path(sys.argv[1]), int(sys.argv[2]), Path(sys.argv[3])
    only = set(sys.argv[4:])
    out.mkdir(parents=True, exist_ok=True)
    rd = RankReader(model, rank, 2)
    dev = torch.device("cuda")
    for name, parts in SETS.items():
        if only and name not in only:
            continue
        w = [rd.get(PREFIX + p + ".weight").to(dev) for p in parts]
        q = qmm.stack_dense_q4(w)
        assert isinstance(q, qmm.Q4), name
        for ext, t in (("w", q.weight), ("s", q.scales), ("b", q.biases)):
            (out / f"{name}.{ext}").write_bytes(t.contiguous().view(torch.uint8).cpu().numpy().tobytes())
        print(name, q.n, q.k, tuple(q.weight.shape), tuple(q.scales.shape), flush=True)
    if not only or "head" in only or "draft" in only:
        raw = rd.get("lm_head.weight")
        v = raw.shape[0] // 2
        h = raw[rank * v:(rank + 1) * v].to(dev)
    if not only or "head" in only:
        f = qmm.make_f8(h)
        (out / "head.q8").write_bytes(f.weight.view(torch.uint8).cpu().numpy().tobytes())
        (out / "head.sc").write_bytes(f.scale.contiguous().view(torch.uint8).cpu().numpy().tobytes())
        print("head", f.n, f.k, flush=True)
    if not only or "draft" in only:                    # the draft head: TF_GLM_DRAFT_QUANT's default, min/max ranges
        q = qmm.draft_quantize4(h)
        for ext, t in (("w", q.weight), ("s", q.scales), ("b", q.biases)):
            (out / f"draft.{ext}").write_bytes(t.contiguous().view(torch.uint8).cpu().numpy().tobytes())
        print("draft", q.n, q.k, flush=True)


main()
