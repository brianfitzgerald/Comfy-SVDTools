from typing import List
from ComfyUI_stability.nodes import NODE_DISPLAY_NAME_MAPPINGS
from comfy.model_patcher import ModelPatcher
import random
import torch
import torch
import torch
from .attention_patch import *

try:
    import xformers  # type: ignore
    import xformers.ops  # type: ignore
except ImportError:
    raise Exception("Install xformers to use attention windowing.")

import comfy.sample as comfy_sample


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
        "scale_position_embedding": ("BOOLEAN", {"default": False}),
        "position_embedding_frames": (
            "INT",
            ATTN_OPTION_ARGS,
        ),
        "attn_k_scale": (
            "FLOAT",
            {"default": 1, "min": 0, "max": 10, "step": 0.1},
        ),
    }
}

EXPERIMENTAL_INPUT_TYPES = {
    "required": {
        **BASE_INPUT_TYPES["required"],
        "attn_q_scale": (
            "FLOAT",
            {"default": 1, "min": 0, "max": 10, "step": 0.1},
        ),
        "attn_v_scale": (
            "FLOAT",
            {"default": 1, "min": 0, "max": 10, "step": 0.1},
        ),
        "attn_window": (attn_window_options,),
        "attn_window_size": (
            "INT",
            {"default": 16, "min": 1, "max": 128, "step": 1},
        ),
        "attn_window_stride": (
            "INT",
            ATTN_OPTION_ARGS,
        ),
        "temporal_attn_scale": (
            "FLOAT",
            {"default": 1, "min": 0, "max": 10, "step": 0.1},
        ),
        "shuffle_windowed_noise": ("BOOLEAN", {"default": False}),
    }
}


def common(
    model: ModelPatcher,
    latent_image: dict,
    scale_position_embedding: bool,
    position_embedding_frames: int,
    attn_k_scale: float,
    attn_q_scale: Optional[float] = None,
    attn_v_scale: Optional[float] = None,
    attn_window: Optional[str] = None,
    attn_window_size: Optional[int] = None,
    attn_window_stride: Optional[int] = None,
    temporal_attn_scale: Optional[float] = None,
    shuffle_windowed_noise: Optional[bool] = None,
):
    latent_tensor = latent_image["samples"]
    video_num_frames: int = latent_tensor.shape[0]
    attn_window_enum = AttentionWindowOption(attn_window)

    state = WindowState.instance()
    state.video_total_frames = video_num_frames
    state.attn_window_option = attn_window_enum
    state.pos_emb_frames = (
        position_embedding_frames if scale_position_embedding else None
    )
    state.attn_k_scale = attn_k_scale
    if attn_q_scale is not None:
        state.attn_q_scale = attn_q_scale
    if attn_v_scale is not None:
        state.attn_v_scale = attn_v_scale

    # experimental node
    if (
        attn_window_size
        and attn_window_stride
        and temporal_attn_scale
        and shuffle_windowed_noise
    ):
        state.attn_window_size = attn_window_size
        state.attn_windows = get_attn_windows(
            video_num_frames, attn_window_size, attn_window_stride
        )
        state.temporal_attn_scale = temporal_attn_scale
        state.shuffle_windowed_noise = shuffle_windowed_noise

        if shuffle_windowed_noise:
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


class SVDToolsPatcher:
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
        model: ModelPatcher,
        latent_image: dict,
        scale_position_embedding: bool,
        position_embedding_frames: int,
        attn_k_scale: float,
        attn_q_scale: float,
        attn_v_scale: float,
        attn_window: str,
        attn_window_size: int,
        attn_window_stride: int,
        temporal_attn_scale: float,
        shuffle_windowed_noise: bool,
    ):
        return common(
            model,
            latent_image,
            scale_position_embedding,
            position_embedding_frames,
            attn_k_scale,
            attn_q_scale,
            attn_v_scale,
            attn_window,
            attn_window_size,
            attn_window_stride,
            temporal_attn_scale,
            shuffle_windowed_noise,
        )


class SVDToolsPatcherExperimental:
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
        model: ModelPatcher,
        latent_image: dict,
        scale_position_embedding: bool,
        position_embedding_frames: int,
        attn_k_scale: float,
    ):
        return common(
            model,
            latent_image,
            scale_position_embedding,
            position_embedding_frames,
            attn_k_scale,
        )


NODE_CLASS_MAPPINGS = {
    "SVDToolsPatcher": SVDToolsPatcher,
    "SVDToolsPatcherExperimental": SVDToolsPatcherExperimental,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "SVDToolsPatcher": "SVD Tools Patcher",
    "SVDToolsPatcherExperimental": "SVD Tools Patcher (Experimental)",
}