from typing import Dict, List, Optional
import torch
from comfy.model_patcher import ModelPatcher
from nodes import common_ksampler
import random
import torch
import comfy.ldm.modules.attention
from comfy.ldm.modules.attention import (
    SpatialVideoTransformer,
)
import torch
import torch
from einops import rearrange, repeat
from typing import Optional, Callable
from functools import partial
from comfy.ldm.modules.diffusionmodules.util import (
    timestep_embedding,
)
from .attention_patch import *

try:
    import xformers  # type: ignore
    import xformers.ops  # type: ignore
except ImportError:
    raise Exception("Install xformers to use attention windowing.")

import comfy.sample as comfy_sample

import comfy.ops


attn_window_options: List[str] = [e.value for e in AttentionWindowOption]


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


ATTN_OPTION_ARGS = {"default": 4, "min": 1, "max": 128, "step": 1}

BASE_INPUT_TYPES = {
    "required": {
        "model": ("MODEL",),
        "latent_image": ("LATENT",),
        "attn_window": (attn_window_options,),
        "attn_window_size": (
            "INT",
            {"default": 16, "min": 1, "max": 128, "step": 1},
        ),
        "attn_window_stride": (
            "INT",
            ATTN_OPTION_ARGS,
        ),
        "pos_emb_scaling": ("BOOLEAN", {"default": False}),
        "pos_emb_frames": (
            "INT",
            ATTN_OPTION_ARGS,
        ),
        "shuffle_windowed_noise": ("BOOLEAN", {"default": False}),
        "temporal_attn_scale": (
            "FLOAT",
            {"default": 1, "min": 0, "max": 10, "step": 0.1},
        ),
    }
}

ADVANCED_INPUT_TYPES = {
    "required": {
        **BASE_INPUT_TYPES["required"],
    }
}


class SVDPatcherBasic:
    @classmethod
    def INPUT_TYPES(cls):
        return BASE_INPUT_TYPES

    RETURN_TYPES = (
        "MODEL",
        "LATENT",
    )
    FUNCTION = "patch"

    CATEGORY = "advanced"

    def patch(
        self,
        model,
        latent_image: dict,
        attn_window: str,
        attn_window_size: int,
        attn_window_stride: int,
        pos_emb_scaling: bool,
        pos_emb_frames: int,
        shuffle_windowed_noise: bool,
        temporal_attn_scale: float,
    ):
        latent_tensor = latent_image["samples"]
        video_num_frames: int = latent_tensor.shape[0]
        attn_window_enum = AttentionWindowOption(attn_window)

        state = WindowState.instance()
        state.video_total_frames = video_num_frames
        state.attn_window_size = attn_window_size
        state.attn_windows = get_attn_windows(
            video_num_frames, attn_window_size, attn_window_stride
        )
        state.attn_window_option = attn_window_enum
        state.pos_emb_frames = pos_emb_frames if pos_emb_scaling else None
        state.temporal_attn_scale = temporal_attn_scale
        state.shuffle_windowed_noise = shuffle_windowed_noise
        state.apply_model_patches = (
            state.pos_emb_frames is not None
            or state.temporal_attn_scale != 1.0
            or attn_window_enum != AttentionWindowOption.DISABLED
        )

        if state.shuffle_windowed_noise:
            for t_start, t_end in state.attn_windows:
                idx_list = list(range(t_start, t_end))
                random.shuffle(idx_list)
                latent_tensor[t_start:t_end] = latent_tensor[idx_list]

        latent_image["samples"] = latent_tensor

        m: ModelPatcher = model.clone()

        patch_model(m)
        comfy_sample.sample = patch_comfy_sample(comfy_sample.sample)
        comfy_sample.sample_custom = patch_comfy_sample(comfy_sample.sample_custom)


        return (m, latent_image)


NODE_CLASS_MAPPINGS = {
    "SVDPatcherBasic": SVDPatcherBasic,
}
