from typing import List
import torch
from comfy.model_patcher import ModelPatcher
import comfy.samplers
from nodes import common_ksampler
import random
import torch
import comfy.ldm.modules.attention
from comfy.ldm.modules.attention import (
    optimized_attention,
    optimized_attention_masked,
    default,
    CrossAttention
)
import torch
from torch import nn


import comfy.ops

ops = comfy.ops.disable_weight_init

T = torch.Tensor


def get_attn_windows(
    video_length: int, window_size: int = 16, stride: int = 4
) -> List[tuple[int, int]]:
    num_blocks_time = (video_length - window_size) // stride + 1
    views = []
    for i in range(num_blocks_time):
        t_start = int(i * stride)
        t_end = t_start + window_size
        views.append((t_start, t_end))
    return views


class WindowState:
    attention_windows: List[tuple[int, int]] = []

    _instance = None

    @classmethod
    def instance(cls):
        if cls._instance is None:
            cls._instance = cls()
        return cls._instance


def attn_windowed(q: T, k: T, v: T, extra_options: dict):
    state = WindowState.instance()
    all_out = torch.zeros_like(q)
    transformer_idx, block_idx, block_depth = (
        extra_options["transformer_index"],
        extra_options["block"],
        extra_options["block_index"],
    )
    print(f"transformer {transformer_idx} block {block_idx} depth {block_depth}")
    for t_start, t_end in state.attention_windows:
        print(f"process attn for {t_start} {t_end}")
        heads = extra_options["n_heads"]
        q_t = q[t_start:t_end]
        k_t = k[t_start:t_end]
        v_t = v[t_start:t_end]
        attn_out = optimized_attention(q_t, k_t, v_t, heads=heads)
        all_out[t_start:t_end] = attn_out
    return all_out

def attn_basic(q: T, k: T, v: T, extra_options: dict):
    heads = extra_options["n_heads"]
    return optimized_attention(q, k, v, heads=heads)

def set_model_patch_replace(model, patch_kwargs, key):
    print(f"patching {key} with kwargs {patch_kwargs}")
    to = model.model_options["transformer_options"]
    if "patches_replace" not in to:
        to["patches_replace"] = {}
    if "attn1" not in to["patches_replace"]:
        to["patches_replace"]["attn1"] = {}
    if key not in to["patches_replace"]["attn1"]:
        to["patches_replace"]["attn1"][key] = attn_basic
    if "attn2" not in to["patches_replace"]:
        to["patches_replace"]["attn2"] = {}
    if key not in to["patches_replace"]["attn2"]:
        to["patches_replace"]["attn2"][key] = attn_basic
    else:
        print(f"already patched {key}")


def patch_model(model: ModelPatcher, is_sdxl: bool = False):
    patch_kwargs = {"number": 0}

    if not is_sdxl:
        for id in [1, 2, 4, 5, 7, 8]:  # id of input_blocks that have cross attention
            set_model_patch_replace(model, patch_kwargs, ("input", id))
            patch_kwargs["number"] += 1
        for id in [
            3,
            4,
            5,
            6,
            7,
            8,
            9,
            10,
            11,
        ]:  # id of output_blocks that have cross attention
            set_model_patch_replace(model, patch_kwargs, ("output", id))
            patch_kwargs["number"] += 1
        set_model_patch_replace(model, patch_kwargs, ("middle", 0))
    else:
        for id in [4, 5, 7, 8]:  # id of input_blocks that have cross attention
            block_indices = range(2) if id in [4, 5] else range(10)  # transformer_depth
            for index in block_indices:
                set_model_patch_replace(model, patch_kwargs, ("input", id, index))
                patch_kwargs["number"] += 1
        for id in range(6):  # id of output_blocks that have cross attention
            block_indices = (
                range(2) if id in [3, 4, 5] else range(10)
            )  # transformer_depth
            for index in block_indices:
                set_model_patch_replace(model, patch_kwargs, ("output", id, index))
                patch_kwargs["number"] += 1
        for index in range(10):
            set_model_patch_replace(model, patch_kwargs, ("middle", 0, index))
            patch_kwargs["number"] += 1


class KSamplerExtended:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "model": ("MODEL",),
                "seed": ("INT", {"default": 0, "min": 0, "max": 0xFFFFFFFFFFFFFFFF}),
                "steps": ("INT", {"default": 20, "min": 1, "max": 10000}),
                "cfg": (
                    "FLOAT",
                    {
                        "default": 8.0,
                        "min": 0.0,
                        "max": 100.0,
                        "step": 0.1,
                        "round": 0.01,
                    },
                ),
                "sampler_name": (comfy.samplers.KSampler.SAMPLERS,),
                "scheduler": (comfy.samplers.KSampler.SCHEDULERS,),
                "positive": ("CONDITIONING",),
                "negative": ("CONDITIONING",),
                "latent_image": ("LATENT",),
                "window_size": (
                    "INT",
                    {"default": 16, "min": 0, "max": 128, "step": 1},
                ),
                "window_stride": (
                    "INT",
                    {"default": 4, "min": 0, "max": 128, "step": 1},
                ),
                "denoise": (
                    "FLOAT",
                    {"default": 1.0, "min": 0.0, "max": 1.0, "step": 0.01},
                ),
            }
        }

    RETURN_TYPES = ("LATENT",)
    FUNCTION = "sample"

    CATEGORY = "sampling"

    def sample(
        self,
        model,
        seed: int,
        steps: int,
        cfg: float,
        sampler_name: str,
        scheduler: str,
        positive: dict,
        negative: dict,
        latent_image: dict,
        window_size: int,
        window_stride: int,
        denoise=1.0,
    ):
        random.seed(seed)
        latent_tensor = latent_image["samples"]
        video_num_frames = latent_tensor.shape[0]

        WindowState.attention_windows = get_attn_windows(
            video_num_frames, window_size, window_stride
        )

        m: ModelPatcher = model.clone()
        print(
            f"computing {len(WindowState.attention_windows)} windows: {WindowState.attention_windows}"
        )
        # patch_model(m)

        latents_dict = common_ksampler(
            m,
            seed,
            steps,
            cfg,
            sampler_name,
            scheduler,
            positive,
            negative,
            latent_image,
            denoise=denoise,
        )[0]

        return (latents_dict,)


NODE_CLASS_MAPPINGS = {
    "KSamplerExtended": KSamplerExtended,
}
