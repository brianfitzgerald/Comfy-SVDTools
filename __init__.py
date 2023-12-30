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
    attention_pytorch,
    BROKEN_XFORMERS,
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
import comfy.model_management as model_management
import math

try:
    import xformers  # type: ignore
    import xformers.ops  # type: ignore
except ImportError:
    raise Exception("Install xformers to use attention windowing.")


def exists(val):
    return val is not None


import comfy.ops

ops = comfy.ops.disable_weight_init

T = torch.Tensor


def attention_xformers_scaling(q, k, v, heads, mask=None, scale: float = 1.0):
    b, _, dim_head = q.shape
    dim_head //= heads
    if BROKEN_XFORMERS:
        if b * heads > 65535:
            return attention_pytorch(q, k, v, heads, mask)

    scale = math.sqrt(scale / dim_head)

    q, k, v = map(
        lambda t: t.unsqueeze(3)
        .reshape(b, -1, heads, dim_head)
        .permute(0, 2, 1, 3)
        .reshape(b * heads, -1, dim_head)
        .contiguous(),
        (q, k, v),
    )

    # actually compute the attention, what we cannot get enough of
    out = xformers.ops.memory_efficient_attention(q, k, v, attn_bias=None, scale=scale)  # type: ignore

    if exists(mask):
        raise NotImplementedError
    out = (
        out.unsqueeze(0)
        .reshape(b, heads, -1, dim_head)
        .permute(0, 2, 1, 3)
        .reshape(b, -1, heads * dim_head)
    )
    return out


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
    SCALE_PER_WINDOW = "scale_per_window"
    INDEPENDENT_WINDOWS = "independent_windows"


attn_window_options: List[str] = [e.value for e in AttentionWindowOption]


class WindowState:
    attn_windows: List[tuple[int, int]] = []

    # Total number of frames, i.e. batch size
    video_total_frames: int = 0
    # Size of window to apply attention to
    attn_window_size: int = 0

    # Options
    attn_window_option: AttentionWindowOption = AttentionWindowOption.DISABLED
    # no. of frames to use for positional embedding
    pos_emb_frames: Optional[float] = None
    apply_model_patches: bool = False
    shuffle_windowed_noise: bool = False
    temporal_attn_scale: float = 1.0

    original_forwards: Dict[str, Callable] = {}

    _instance = None

    @classmethod
    def instance(cls):
        if cls._instance is None:
            cls._instance = cls()
        return cls._instance


def attn_windowed(q: T, k: T, v: T, extra_options: dict) -> T:
    state = WindowState.instance()
    window_option = state.attn_window_option
    n_heads = extra_options["n_heads"]
    temporal = extra_options.get("temporal", False)

    is_attn1 = k.shape[1] == state.video_total_frames
    attn_scale = state.temporal_attn_scale

    if (
        not temporal
        or window_option == AttentionWindowOption.DISABLED
        or (window_option == AttentionWindowOption.INDEPENDENT_WINDOWS and is_attn1)
    ):
        out = attention_xformers_scaling(q, k, v, heads=n_heads, scale=attn_scale)
        return out

    out = torch.zeros_like(q)
    count = torch.zeros_like(q)
    if window_option == AttentionWindowOption.INDEPENDENT_WINDOWS:
        for t_start, t_end in state.attn_windows:
            q_t = q[:, t_start:t_end]
            k_t = k[:, t_start:t_end]
            v_t = v[:, t_start:t_end]

            weight_sequence = generate_weight_sequence(t_end - t_start)
            weight_tensor = torch.ones_like(count[:, t_start:t_end])
            weight_tensor = weight_tensor * torch.Tensor(weight_sequence).to(
                q.device
            ).unsqueeze(0).unsqueeze(-1)

            attn_out = attention_xformers_scaling(
                q_t, k_t, v_t, heads=n_heads, scale=attn_scale
            )
            out[:, t_start:t_end] += attn_out * weight_tensor
            count[:, t_start:t_end] += weight_tensor
        final_out = torch.where(count > 0, out / count, out)
        return final_out
    elif window_option == AttentionWindowOption.SCALE_PER_WINDOW:
        # TODO make this work - currently not any better than independent
        for t_start, t_end in state.attn_windows:
            t_mask = torch.zeros_like(k)
            t_mask[:, t_start:t_end] = 1.0
            v_t = v * t_mask

            attn_out = attention_xformers_scaling(q, k, v, n_heads, scale=attn_scale)
            out += attn_out * t_mask
    return out


def attn_basic(q: T, k: T, v: T, extra_options: dict):
    heads = extra_options["n_heads"]
    return attention_xformers_scaling(q, k, v, heads=heads)


def set_model_patch_replace(model, patch_kwargs, key):
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

    if not unpatch:
        # patch attention
        patch_kwargs = {}
        for id in range(11):
            set_model_patch_replace(model, patch_kwargs, ("input", id, 0))
        set_model_patch_replace(model, patch_kwargs, ("middle", 0, 0))
        for id in range(12):
            set_model_patch_replace(model, patch_kwargs, ("output", id, 0))

    if state.apply_model_patches:
        # patch SpatialVideoTransformer
        for layer_name, module in model.model.named_modules():
            if isinstance(module, SpatialVideoTransformer):
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

    num_frames = torch.linspace(0, state.attn_window_size, timesteps, device=x.device)
    num_frames = num_frames.round().long()
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


ATTN_OPTION_ARGS = {"default": 4, "min": 1, "max": 128, "step": 1}


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
                "denoise": (
                    "FLOAT",
                    {"default": 1, "min": 0, "max": 1, "step": 0.1},
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
        pos_emb_scaling: bool,
        pos_emb_frames: int,
        shuffle_windowed_noise: bool,
        temporal_attn_scale: float,
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
