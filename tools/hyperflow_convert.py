#!/usr/bin/env python3
"""HyperFlow H3 Flash int8_convrot (diffusers layout) -> ComfyUI's int8 layout.

  python tools/hyperflow_convert.py INT8_DIR GGUF OUT.safetensors

Dense int8 linears become ComfyUI `int8_tensorwise` + convrot layers (weight, weight_scale [out, 1], comfy_quant),
which is what the stock pruned H3 int8 file uses. Both use the same regular Hadamard (kron of H4/2, group 256) and
store W·R with one scale per output row, so the int8 values copy over untouched; q, k and v rows are concatenated.
Low-rank pairs stay bf16 under the GGUF's names (`attn.q_proj.down.weight` ...), dropped blocks are renumbered, and
`endpoint_time_embedder.*`, `hyperflow_gate` and `hyperflow_sigmas` are kept for the HyperFlow int8 loader.

Every tensor written is checked against the same tensor in the GGUF (Q8_0, the same weights): cosine > 0.999
for quantized ones, near-exact for float ones. A wrong name map or row order fails loudly.
"""
import json, os, re, sys

import gguf
import numpy as np
import torch
from safetensors import safe_open
from safetensors.torch import save_file

QUANT = json.dumps({"format": "int8_tensorwise", "convrot": True, "convrot_groupsize": 256}).encode()
TOP = {
    "proj_in": "video_patch_proj", "audio_proj_in": "audio_patch_proj", "context_embedder": "condition_proj",
    "time_embedder.linear_1": "time_embedder.proj_in", "time_embedder.linear_2": "time_embedder.proj_out",
    "endpoint_time_embedder.linear_1": "endpoint_time_embedder.proj_in",
    "endpoint_time_embedder.linear_2": "endpoint_time_embedder.proj_out",
    "norm_out.linear": "final_layer.adaln_proj.linear", "norm_out.norm": "final_layer.norm",
    "proj_out": "final_layer.video_out", "audio_proj_out": "final_layer.audio_out",
}
LINEAR = {"attn.to_q": "attn.q_proj", "attn.to_k": "attn.k_proj", "attn.to_v": "attn.v_proj",
          "attn.to_out.0": "attn.out_proj", "ff.net.0.proj": "mlp.fc1", "ff.net.2": "mlp.fc2"}
NORM = {"norm1": "norm1", "norm2": "norm2", "attn.norm_q": "attn.q_norm", "attn.norm_k": "attn.k_norm",
        "adaln_proj.linear": "adaln_proj.linear"}


def swap_halves(t):
    """Diffusers' SwiGLU proj is [hidden; gate], ComfyUI's fc1 chunks [gate; hidden]."""
    a, b = t.chunk(2, dim=0)
    return torch.cat([b, a])


def pad_rank(t, leaf, multiple=256):
    """Zero-pad a low-rank pair's rank to a multiple of 256, as the GGUF does: up @ down is unchanged."""
    rank = t.shape[0] if leaf == "down" else t.shape[1]
    extra = -rank % multiple
    if not extra:
        return t
    if leaf == "down":
        return torch.cat([t, t.new_zeros(extra, t.shape[1])])
    return torch.cat([t, t.new_zeros(t.shape[0], extra)], dim=1)


def hadamard(n):
    h4 = torch.tensor([[1, 1, 1, -1], [1, 1, -1, 1], [1, -1, 1, 1], [-1, 1, 1, 1]], dtype=torch.float64) / 2
    h = torch.ones(1, 1, dtype=torch.float64)
    while h.shape[0] < n:
        h = torch.kron(h, h4)
    return h.float()


class Reference:
    def __init__(self, path):
        self.tensors = {t.name: t for t in gguf.GGUFReader(path).tensors}

    def get(self, name):
        t = self.tensors[name]
        a = gguf.quants.dequantize(t.data, t.tensor_type)
        return torch.from_numpy(np.ascontiguousarray(a)).float().reshape([int(s) for s in reversed(t.shape)])


def main():
    src, ref_path, out = sys.argv[1:4]
    config = json.load(open(os.path.join(src, "config.json")))
    dropped = sorted(config.get("dropped_blocks", []))
    index = json.load(open(os.path.join(src, "diffusion_pytorch_model.safetensors.index.json")))["weight_map"]
    files = {f: safe_open(os.path.join(src, f), "pt") for f in set(index.values())}
    get = lambda name: files[index[name]].get_tensor(name)
    ref = Reference(ref_path)
    H = hadamard(256)
    out_sd, checks = {}, []

    def renumber(i):
        return i - sum(d < i for d in dropped)

    def check(name, ours, quantized):
        want = ref.get(name)
        if tuple(want.shape) != tuple(ours.shape):
            raise SystemExit(f"{name}: shape {tuple(ours.shape)} vs GGUF {tuple(want.shape)}")
        a, b = ours.float().flatten(), want.flatten()
        cos = float(torch.nn.functional.cosine_similarity(a, b, dim=0)) if a.norm() > 0 else 1.0
        rel = float((a - b).norm() / max(b.norm(), 1e-12))
        floor = 0.999 if quantized else 0.9999
        checks.append((name, cos, rel))
        if cos < floor or (not quantized and rel > 0.01):
            raise SystemExit(f"{name}: cosine {cos:.6f} rel {rel:.4f} against the GGUF")

    def dequant(q, s):
        w = q.float() * s.float().reshape(-1, 1)
        k = w.shape[1]
        return (w.reshape(-1, k // 256, 256) @ H).reshape(w.shape[0], k)  # W·R·R = W

    def put_int8(name, qs, ss):
        q, s = torch.cat(qs), torch.cat(ss).float().reshape(-1, 1)
        out_sd[f"{name}.weight"], out_sd[f"{name}.weight_scale"] = q, s
        out_sd[f"{name}.comfy_quant"] = torch.tensor(list(QUANT), dtype=torch.uint8)
        check(f"{name}.weight", dequant(q, s), True)

    def put(name, t, quantized=False):
        out_sd[name] = t
        check(name, t, quantized)

    # Top level.
    for d, c in TOP.items():
        for leaf in ("weight", "bias"):
            if f"{d}.{leaf}" in index:
                put(f"{c}.{leaf}", get(f"{d}.{leaf}"))
    for name in ("rope.inv_freq", "hyperflow_sigmas"):
        out_sd[name] = ref.get(name).contiguous()
    out_sd["hyperflow_gate"] = torch.tensor([float(config["hyperflow_gate"])], dtype=torch.float32)

    # Token refiner: plain weights, q/k/v concatenated.
    for i in range(config["num_refiner_layers"]):
        d, c = f"token_refiner.refiner_blocks.{i}", f"token_refiner.blocks.{i}"
        put(f"{c}.attn.qkv_proj.weight", torch.cat([get(f"{d}.attn.to_{x}.weight") for x in "qkv"]))
        put(f"{c}.attn.out_proj.weight", get(f"{d}.attn.to_out.0.weight"))
        put(f"{c}.mlp.fc1.weight", swap_halves(get(f"{d}.ff.net.0.proj.weight")))
        put(f"{c}.mlp.fc2.weight", get(f"{d}.ff.net.2.weight"))
        for dn, cn in (("norm1", "norm1"), ("norm2", "norm2"), ("attn.norm_q", "attn.q_norm"), ("attn.norm_k", "attn.k_norm")):
            put(f"{c}.{cn}.weight", get(f"{d}.{dn}.weight"))
    put("token_refiner.final_norm.weight", get("token_refiner.final_norm.weight"))

    def put_low_rank(d, c, dn):
        for leaf in ("down", "up"):
            t = get(f"{d}.{dn}.{leaf}.weight").to(torch.bfloat16)
            if dn == "ff.net.0.proj" and leaf == "up":
                t = swap_halves(t)
            put(f"{c}.{LINEAR[dn]}.{leaf}.weight", pad_rank(t, leaf), True)

    # Main blocks. Each linear is either int8 or a low-rank pair, and a block can mix them (fl2va's block 2 keeps
    # an int8 fc2, block 49 factors only fc2). q, k and v are always the same kind.
    factorized = 0
    for i in range(config["num_layers"]):
        if i in dropped:
            continue
        d, c = f"transformer_blocks.{i}", f"blocks.{renumber(i)}"
        for dn, cn in NORM.items():
            for leaf in ("weight", "bias"):
                if f"{d}.{dn}.{leaf}" in index:
                    put(f"{c}.{cn}.{leaf}", get(f"{d}.{dn}.{leaf}"))
        int8 = {dn: f"{d}.{dn}.qweight" in index for dn in LINEAR}
        if len({int8[f"attn.to_{x}"] for x in "qkv"}) != 1:
            raise SystemExit(f"{d}: q, k and v are not all int8 or all low-rank")
        if int8["attn.to_q"]:
            put_int8(f"{c}.attn.qkv_proj", [get(f"{d}.attn.to_{x}.qweight") for x in "qkv"],
                     [get(f"{d}.attn.to_{x}.wscale") for x in "qkv"])
        for dn in ("attn.to_out.0", "ff.net.2"):
            if int8[dn]:
                put_int8(f"{c}.{LINEAR[dn]}", [get(f"{d}.{dn}.qweight")], [get(f"{d}.{dn}.wscale")])
        if int8["ff.net.0.proj"]:
            put_int8(f"{c}.mlp.fc1", [swap_halves(get(f"{d}.ff.net.0.proj.qweight"))],
                     [swap_halves(get(f"{d}.ff.net.0.proj.wscale"))])
        low = [dn for dn, q in int8.items() if not q]
        for dn in low:
            put_low_rank(d, c, dn)
        factorized += len(low)
        print(f"block {i} -> {c}" + (f", low-rank: {len(low)} of {len(LINEAR)}" if low else ""), flush=True)

    missing = set(ref.tensors) - set(out_sd)
    if missing:
        raise SystemExit(f"in the GGUF but not written: {sorted(missing)[:20]}")
    worst = min(checks, key=lambda x: x[1])
    print(f"{len(out_sd)} tensors, {factorized} low-rank linears, {len(checks)} checked; "
          f"worst cosine {worst[1]:.6f} ({worst[0]}), worst rel {max(c[2] for c in checks):.4f}")
    save_file(out_sd, out, metadata={"hyperflow": json.dumps({"source": f"VideoRebirthAI/hyperflow-h3-flash {os.path.basename(os.path.normpath(src))}",
                                                               "dropped_blocks": dropped})})


if __name__ == "__main__":
    main()
