import time
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
    CrossAttention,
)
import torch
from torch import nn
from einops import rearrange


import comfy.ops

ops = comfy.ops.disable_weight_init

T = torch.Tensor


def generate_weight_sequence(n):
    if n % 2 == 0:
        max_weight = n // 2
        weight_sequence = list(range(1, max_weight + 1, 1)) + list(
            range(max_weight, 0, -1)
        )
    else:
        max_weight = (n + 1) // 2
        weight_sequence = (
            list(range(1, max_weight, 1))
            + [max_weight]
            + list(range(max_weight - 1, 0, -1))
        )
    return weight_sequence


def get_attn_windows(
    video_length: int, window_size: int = 16, stride: int = 4
) -> List[tuple[int, int]]:
    windows = []
    current = 0
    while current < video_length:
        window_end = min(current + window_size, video_length)
        windows.append((current, window_end))
        current += stride
    return windows


class WindowState:
    attention_windows: List[tuple[int, int]] = []
    num_frames: int = 0

    _instance = None

    @classmethod
    def instance(cls):
        if cls._instance is None:
            cls._instance = cls()
        return cls._instance


def attn_windowed(q: T, k: T, v: T, extra_options: dict):
    state = WindowState.instance()
    heads = extra_options["n_heads"]
    temporal = extra_options.get("temporal", False)

    if not temporal:
        value_out = optimized_attention(q, k, v, heads=heads)
        return value_out

    value_out = torch.zeros_like(q)
    count = torch.zeros_like(q)

    b, _, dim_head = q.shape
    dim_head //= heads
    q, k, v = map(
        lambda t: t.view(b, -1, heads, dim_head).transpose(1, 2),
        (q, k, v),
    )

    for t_start, t_end in state.attention_windows:
        q_t = q[:, :, t_start:t_end]
        k_t = k[:, :, t_start:t_end]
        v_t = v[:, :, t_start:t_end]

        weight_sequence = generate_weight_sequence(t_end - t_start)
        weight_tensor = torch.ones_like(count[:, t_start:t_end])
        weight_tensor = weight_tensor * torch.Tensor(weight_sequence).to(
            q.device
        ).unsqueeze(0).unsqueeze(-1)

        out = torch.nn.functional.scaled_dot_product_attention(
            q_t, k_t, v_t, attn_mask=None, dropout_p=0.0, is_causal=False
        )
        out = out.transpose(1, 2).reshape(b, -1, heads * dim_head)

        value_out[:, t_start:t_end] += out * weight_tensor
        count[:, t_start:t_end] += weight_tensor
    final_out = torch.where(count > 0, value_out / count, value_out)
    return final_out


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
        to["patches_replace"]["attn1"][key] = attn_windowed
    if "attn2" not in to["patches_replace"]:
        to["patches_replace"]["attn2"] = {}
    if key not in to["patches_replace"]["attn2"]:
        to["patches_replace"]["attn2"][key] = attn_windowed
    else:
        print(f"already patched {key}")


def patch_model(model: ModelPatcher):
    patch_kwargs = {}
    for id in range(11):
        set_model_patch_replace(model, patch_kwargs, ("input", id, 0))
    set_model_patch_replace(model, patch_kwargs, ("middle", 0, 0))
    for id in range(12):
        set_model_patch_replace(model, patch_kwargs, ("output", id, 0))


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
                        "default": 2.5,
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
                    {"default": 16, "min": 1, "max": 128, "step": 1},
                ),
                "window_stride": (
                    "INT",
                    {"default": 4, "min": 1, "max": 128, "step": 1},
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
        video_num_frames: int = latent_tensor.shape[0]

        WindowState.num_frames = video_num_frames
        WindowState.attention_windows = get_attn_windows(
            video_num_frames, window_size, window_stride
        )

        m: ModelPatcher = model.clone()
        print(
            f"computing {len(WindowState.attention_windows)} windows: {WindowState.attention_windows}"
        )
        patch_model(m)

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
