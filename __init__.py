"""HyperFlow H3 Flash on ComfyUI's native int8 ConvRot path.

Loads the file tools/hyperflow_convert.py writes (nodevorg/studio-models-h3 on Hugging Face): dense blocks are ComfyUI `int8_tensorwise` + convrot layers (loaded by
ComfyUI's mixed-precision ops, int8 matmul), factorized blocks keep bf16 low-rank pairs. The low-rank modules and
the two-time (t, r) embedder are ComfyUI-HyperFlow-Flash's own classes, so that pack must be installed; its
HyperFlowSigmas node gives the schedule.
"""
import logging
import re
import sys
from collections import defaultdict

import comfy.ops
import comfy.sd
import comfy.utils
import folder_paths
import nodes as comfy_nodes
import torch

PART = re.compile(r"^blocks\.(\d+)\.(attn\.[qkv]_proj|attn\.out_proj|mlp\.fc[12])\.(down|up)\.weight$")
QKV = ("attn.q_proj", "attn.k_proj", "attn.v_proj")
ENDPOINT = "endpoint_time_embedder."


def hyperflow():
    loader = comfy_nodes.NODE_CLASS_MAPPINGS.get("HyperFlowGGUFLoader")
    if loader is None:
        raise RuntimeError("HyperFlow int8 loading needs ComfyUI-HyperFlow-Flash installed.")
    package = loader.__module__.rsplit(".", 1)[0]
    return sys.modules[f"{package}.model_patch"], sys.modules[f"{package}.loader_nodes"]


def low_rank(parts, name, dtype, ops, patch):
    down, up = parts[name]["down"], parts[name]["up"]
    a = ops.Linear(down.shape[1], down.shape[0], bias=False, dtype=dtype, device="cpu")
    b = ops.Linear(up.shape[1], up.shape[0], bias=False, dtype=dtype, device="cpu")
    a.weight = torch.nn.Parameter(down.to(dtype), requires_grad=False)
    b.weight = torch.nn.Parameter(up.to(dtype), requires_grad=False)
    return patch.LowRankLinear(a, b)


def load(path):
    patch, loader_nodes = hyperflow()
    sd = comfy.utils.load_torch_file(path)
    gate = float(sd.pop("hyperflow_gate").flatten()[0])
    sigmas = sd.pop("hyperflow_sigmas").float().flatten()
    endpoint_sd = {k.removeprefix(ENDPOINT): sd.pop(k) for k in list(sd) if k.startswith(ENDPOINT)}
    blocks = defaultdict(lambda: defaultdict(dict))
    for k in list(sd):
        m = PART.match(k)
        if m:
            blocks[int(m.group(1))][m.group(2)][m.group(3)] = sd.pop(k)
    # The factorized blocks' dense linears are missing from sd; the model builds them empty and they are replaced
    # below, before any forward.
    patcher = comfy.sd.load_diffusion_model_state_dict(sd, disable_dynamic=True)
    if patcher is None:
        raise RuntimeError(f"Could not detect the model type of {path}")
    model = patcher.model.diffusion_model
    if not isinstance(model, patch.PackedModel):
        raise RuntimeError(f"{path} is not a packed video+audio DiT checkpoint")
    ops, dtype = comfy.ops.manual_cast, torch.bfloat16
    for i, parts in blocks.items():
        block = model.blocks[i]
        block.attn.qkv_proj = patch.SplitQKV(*(low_rank(parts, n, dtype, ops, patch) for n in QKV))
        block.attn.out_proj = low_rank(parts, "attn.out_proj", dtype, ops, patch)
        block.mlp.fc1 = low_rank(parts, "mlp.fc1", dtype, ops, patch)
        block.mlp.fc2 = low_rank(parts, "mlp.fc2", dtype, ops, patch)
        block.mlp.__class__ = patch.FactorizedMLP
    hidden, freq_dim = endpoint_sd["proj_in.weight"].shape
    embedder = patch.packed_dit.TimeEmbedder(freq_dim, hidden, endpoint_sd["proj_out.weight"].shape[0],
                                             dtype=torch.float32, device="cpu", operations=ops)
    embedder.load_state_dict({k: v.float() for k, v in endpoint_sd.items()})
    model.endpoint_time_embedder = embedder
    model.time_embedder.__class__ = patch.TwoTimeEmbedder
    model.time_embedder.__dict__["endpoint_state"] = patch.EndpointState(embedder, gate)
    model.__class__ = patch.HyperFlowModel
    model.hyperflow_sigmas = sigmas
    patcher.size = 0  # modules changed; let the patcher measure again
    logging.info("HyperFlow int8: %d factorized blocks, %d blocks, gate %.3g, %d-step schedule",
                 len(blocks), len(model.blocks), gate, sigmas.shape[0] - 1)
    return patcher


class HyperFlowInt8Loader:
    TITLE = "HyperFlow Int8 Loader"
    CATEGORY = "HyperFlow"
    RETURN_TYPES = ("MODEL",)
    FUNCTION = "load"

    @classmethod
    def INPUT_TYPES(cls):
        names = [n for n in folder_paths.get_filename_list("diffusion_models") if "hyperflow" in n and n.endswith(".safetensors")]
        return {"required": {"unet_name": (names,)}}

    def load(self, unet_name):
        return (load(folder_paths.get_full_path_or_raise("diffusion_models", unet_name)),)


NODE_CLASS_MAPPINGS = {"HyperFlowInt8Loader": HyperFlowInt8Loader}
NODE_DISPLAY_NAME_MAPPINGS = {"HyperFlowInt8Loader": HyperFlowInt8Loader.TITLE}
