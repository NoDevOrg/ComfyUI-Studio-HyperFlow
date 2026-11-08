# ComfyUI-Studio-HyperFlow

A ComfyUI loader for [HyperFlow H3 Flash](https://huggingface.co/VideoRebirthAI/hyperflow-h3-flash) in ComfyUI's native int8 ConvRot layout, the files at [nodevorg/studio-models-h3](https://huggingface.co/nodevorg/studio-models-h3). [studio.no.dev](https://studio.no.dev) uses it for its HyperFlow presets.

Video Rebirth's [ComfyUI-HyperFlow-Flash](https://github.com/Video-Rebirth/ComfyUI-HyperFlow-Flash) loads their GGUF files. This pack loads the int8 file instead, so ComfyUI runs the dense blocks through its own int8 matmuls. HyperFlow drops one block from H3, factors 19 blocks into bf16 low-rank pairs and adds a second time embedder for each step's endpoint. The loader builds the dense model with ComfyUI's H3 loader, then swaps in the low-rank blocks and the endpoint embedder.

On a GB10 the int8 file renders the same clips as Video Rebirth's Q8_0 GGUF (21.7 dB PSNR at the same seed) at about the same speed.

## Install

Clone this pack and Video Rebirth's into `ComfyUI/custom_nodes`. The loader reuses ComfyUI-HyperFlow-Flash's model classes, and its `HyperFlowSigmas` node gives the schedule.

```sh
cd ComfyUI/custom_nodes
git clone https://github.com/NoDevOrg/ComfyUI-Studio-HyperFlow
git clone https://github.com/Video-Rebirth/ComfyUI-HyperFlow-Flash
pip install -r ComfyUI-HyperFlow-Flash/requirements.txt
```

Tested with ComfyUI v0.44.0 and ComfyUI-HyperFlow-Flash at `20a22ab`.

## Models

```sh
hf download nodevorg/studio-models-h3 --include "diffusion_models/*" --local-dir ComfyUI/models
```

There are two towers. `minimax_h3_fl2va_...` does text to video and image to video from a first frame, or first and last frames, through `MiniMaxH3ImageToVideo`. It draws animals noticeably better than ref2va in text to video. `minimax_h3_ref2va_...` does video from reference images and audio through `MiniMaxH3ReferenceToVideo`.

The text encoder and VAEs are the stock MiniMax H3 files from [Comfy-Org/MiniMax-H3](https://huggingface.co/Comfy-Org/MiniMax-H3). The example uses `text_encoders/qwen3vl_32b_minimax_h3_nvfp4_awq.safetensors`, `vae/minimax_h3_video_vae_fp16.safetensors` and `vae/minimax_h3_audio_vae_fp32.safetensors`.

## Nodes

- **HyperFlow Int8 Loader** (`HyperFlowInt8Loader`) lists the `diffusion_models` files with `hyperflow` in the name and returns a MODEL.
- **HyperFlowSigmas** comes from ComfyUI-HyperFlow-Flash. Feed it the model and 6 or 8 steps.

Sample with `SamplerCustomAdvanced` and `BasicGuider`, using `euler` or `lcm`. HyperFlow is distilled for few steps, so leave PDD out.

## Example workflow

`example_workflows/hyperflow_h3_t2va_6step_lcm.json` wraps the graph in a "Hyperflow H3" subgraph on the ref2va tower: `lcm` at 6 steps, 544x960 portrait, 362 frames (15 s), a three-shot prompt with two lines of dialogue. ref2va handles plain text to video fine, and the subgraph exposes its reference image and audio sockets, so you can attach references or swap the model file without rewiring. For image to video, load the fl2va file and condition with `MiniMaxH3ImageToVideo` and a `first_frame`, plus an optional `last_frame`.

## Converting it yourself

`tools/hyperflow_convert.py` writes either int8 file from Video Rebirth's release. It checks every tensor against their Q8_0 GGUF of the same weights and stops on any mismatch. It needs `torch`, `safetensors`, `numpy` and `gguf`, which a ComfyUI environment with ComfyUI-HyperFlow-Flash has.

```sh
T=fl2va  # or ref2va
hf download VideoRebirthAI/hyperflow-h3-flash --include "${T}_int8_convrot/*" \
  --include "${T}_gguf/minimax_h3_${T}_hyperflow_flash_14b-Q8_0.gguf" --local-dir src
python tools/hyperflow_convert.py src/${T}_int8_convrot \
  src/${T}_gguf/minimax_h3_${T}_hyperflow_flash_14b-Q8_0.gguf \
  minimax_h3_${T}_hyperflow_flash_int8_convrot_comfy.safetensors
```

The int8 values and scales copy over unchanged, since both layouts use the same Hadamard rotation and per-row scales. The converter renames tensors, concatenates q, k and v, renumbers the blocks after the dropped ones, puts the SwiGLU gate half first, and zero-pads the low-rank ranks to multiples of 256.

## License

The code here is GPL-3.0, the same as ComfyUI-HyperFlow-Flash, whose classes it imports. See [LICENSE](LICENSE).

The weights are a separate matter. They're a Model Derivative of MiniMax H3 under the [MiniMax H3 Community License Agreement](https://huggingface.co/MiniMaxAI/MiniMax-H3/blob/main/LICENSE). Read it before downloading, since it limits where and how the weights can be used. Not affiliated with MiniMax or Video Rebirth. Powered by MiniMax H3.
