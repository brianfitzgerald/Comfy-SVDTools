from typing import Dict, List, Optional
import torch
from comfy.model_patcher import ModelPatcher
import comfy.samplers
from nodes import common_ksampler
import random
import torch
import comfy.ldm.modules.attention
from comfy.ldm.modules.attention import (
    SpatialVideoTransformer,
    optimized_attention,
)
import torch
import torch
from einops import rearrange, repeat
from typing import Optional, Callable
from functools import partial
from comfy.ldm.modules.diffusionmodules.util import (
    timestep_embedding,
)
from enum import Enum


def exists(val):
    return val is not None


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


class AttentionWindowOption(Enum):
    DISABLED = "disabled"
    SCALE_K = "scale_k"
    FULL_ASSIGN = "full_assign"


attn_window_options: List[str] = [e.value for e in AttentionWindowOption]


class WindowState:
    attn_windows: List[tuple[int, int]] = []

    # Total number of frames, i.e. batch size
    video_total_frames: int = 0
    # Size of window to apply attention to
    attn_window_size: int = 0

    # Options
    attn_window_option: AttentionWindowOption = AttentionWindowOption.DISABLED
    time_embedding_frames: Optional[float] = None
    patch_transformer: bool = True
    shuffle_inits: bool = True

    original_forwards: Dict[str, Callable] = {}

    _instance = None

    @classmethod
    def instance(cls):
        if cls._instance is None:
            cls._instance = cls()
        return cls._instance


def attn_windowed(q: T, k: T, v: T, extra_options: dict):
    state = WindowState.instance()
    value_out = torch.zeros_like(q)
    n_heads = extra_options["n_heads"]
    temporal = extra_options.get("temporal", False)
    is_attn1 = k.shape[1] == state.video_total_frames
    if not temporal or is_attn1:
        value_out = optimized_attention(q, k, v, heads=n_heads)
        return value_out

    count = torch.zeros_like(q)
    for t_start, t_end in state.attn_windows:
        q_t = q[:, t_start:t_end]
        k_t = k[:, t_start:t_end]
        v_t = v[:, t_start:t_end]

        weight_sequence = generate_weight_sequence(t_end - t_start)
        weight_tensor = torch.ones_like(count[:, t_start:t_end])
        weight_tensor = weight_tensor * torch.Tensor(weight_sequence).to(
            q.device
        ).unsqueeze(0).unsqueeze(-1)

        attn_out = optimized_attention(q_t, k_t, v_t, heads=n_heads)
        value_out[:, t_start:t_end] += attn_out * weight_tensor
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


def patch_model(model: ModelPatcher, unpatch: bool = False):
    state = WindowState.instance()

    if state.attn_window_option != AttentionWindowOption.DISABLED:
        # patch attention
        patch_kwargs = {}
        for id in range(11):
            set_model_patch_replace(model, patch_kwargs, ("input", id, 0))
        set_model_patch_replace(model, patch_kwargs, ("middle", 0, 0))
        for id in range(12):
            set_model_patch_replace(model, patch_kwargs, ("output", id, 0))

    if state.patch_transformer:
        # patch SpatialVideoTransformer
        for layer_name, module in model.model.named_modules():
            if isinstance(module, SpatialVideoTransformer):
                print(f"update layer={layer_name}, unpatch={unpatch}")
                if unpatch and state.original_forwards:
                    module.forward = state.original_forwards[layer_name]
                else:
                    state.original_forwards[layer_name] = module.forward
                    module.forward = partial(patched_forward, module)


def patched_forward(
    self: SpatialVideoTransformer,
    x: torch.Tensor,
    context: Optional[torch.Tensor] = None,
    time_context: Optional[torch.Tensor] = None,
    timesteps: Optional[int] = None,
    image_only_indicator: Optional[torch.Tensor] = None,
    transformer_options={},
) -> torch.Tensor:
    _, _, h, w = x.shape
    x_in = x
    spatial_context = None
    if exists(context):
        spatial_context = context

    assert spatial_context is not None
    assert timesteps is not None

    if self.use_spatial_context:
        assert context is not None
        assert (
            context.ndim == 3
        ), f"n dims of spatial context should be 3 but are {context.ndim}"

        if time_context is None:
            time_context = context
        time_context_first_timestep = time_context[::timesteps]
        time_context = repeat(
            time_context_first_timestep, "b ... -> (b n) ...", n=h * w
        )
    elif time_context is not None and not self.use_spatial_context:
        time_context = repeat(time_context, "b ... -> (b n) ...", n=h * w)
        if time_context.ndim == 2:
            time_context = rearrange(time_context, "b c -> b 1 c")

    x = self.norm(x)
    if not self.use_linear:
        x = self.proj_in(x)
    x = rearrange(x, "b c h w -> b (h w) c")
    if self.use_linear:
        x = self.proj_in(x)

    state = WindowState.instance()

    num_frames = torch.arange(timesteps // 2, device=x.device)
    num_frames = torch.repeat_interleave(num_frames, 2)
    num_frames = repeat(num_frames, "t -> b t", b=x.shape[0] // timesteps)
    num_frames = rearrange(num_frames, "b t -> (b t)")
    t_emb = timestep_embedding(
        num_frames,
        self.in_channels,
        repeat_only=False,
        max_period=self.max_time_embed_period,
    ).to(x.dtype)
    emb = self.time_pos_embed(t_emb)
    emb = emb[:, None, :]

    for it_, (block, mix_block) in enumerate(
        zip(self.transformer_blocks, self.time_stack)
    ):
        transformer_options["block_index"] = it_
        x = block(
            x,
            context=spatial_context,
            transformer_options=transformer_options,
        )

        x_mix = x
        x_mix = x_mix + emb

        B, S, C = x_mix.shape
        x_mix = rearrange(x_mix, "(b t) s c -> (b s) t c", t=timesteps)
        mix_options = transformer_options.copy()
        mix_options["temporal"] = True
        x_mix = mix_block(x_mix, context=time_context, transformer_options=mix_options)
        x_mix = rearrange(
            x_mix, "(b s) t c -> (b t) s c", s=S, b=B // timesteps, c=C, t=timesteps
        )

        x = self.time_mixer(
            x_spatial=x, x_temporal=x_mix, image_only_indicator=image_only_indicator
        )

    if self.use_linear:
        x = self.proj_out(x)
    x = rearrange(x, "b (h w) c -> b c h w", h=h, w=w)
    if not self.use_linear:
        x = self.proj_out(x)
    out = x + x_in
    return out


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
                "attn_window": (attn_window_options,),
                "attn_window_size": (
                    "INT",
                    {"default": 16, "min": 1, "max": 128, "step": 1},
                ),
                "attn_window_stride": (
                    "INT",
                    {"default": 4, "min": 1, "max": 128, "step": 1},
                ),
                "shuffle_noise": ("BOOLEAN", {"default": False}),
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
        attn_window: str,
        attn_window_size: int,
        attn_window_stride: int,
        shuffle_noise: bool,
        denoise=1.0,
    ):
        random.seed(seed)
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

        if shuffle_noise:
            for t_start, t_end in state.attn_windows:
                idx_list = list(range(t_start, t_end))
                random.shuffle(idx_list)
                print(f"shuffled {idx_list} for window {t_start} to {t_end}")
                latent_tensor[t_start:t_end] = latent_tensor[idx_list]

        latent_image["samples"] = latent_tensor

        m: ModelPatcher = model.clone()
        print(
            f"computing {len(WindowState.attn_windows)} windows: {WindowState.attn_windows}"
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

        patch_model(m, True)

        return (latents_dict,)


NODE_CLASS_MAPPINGS = {
    "KSamplerExtended": KSamplerExtended,
}
