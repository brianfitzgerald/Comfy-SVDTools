import torch
from comfy.model_patcher import ModelPatcher
import comfy.samplers
import math
import comfy.ops
from enum import Enum
from typing import Dict, List, Optional, Callable
from einops import rearrange, repeat
import comfy.ldm.modules.attention
from comfy.ldm.modules.attention import (
    SpatialVideoTransformer,
    attention_pytorch,
    BROKEN_XFORMERS,
)
from comfy.ldm.modules.diffusionmodules.util import (
    timestep_embedding,
)
from functools import partial

ops = comfy.ops.disable_weight_init

T = torch.Tensor

try:
    import xformers  # type: ignore
    import xformers.ops  # type: ignore
except ImportError:
    raise Exception("Install xformers to use attention windowing.")


class AttentionWindowOption(Enum):
    DISABLED = "disabled"
    SCALE_PER_WINDOW = "scale_per_window"
    INDEPENDENT_WINDOWS = "independent_windows"


class WindowState:
    attn_windows: List[tuple[int, int]] = []

    # Total number of frames, i.e. batch size
    video_total_frames: int = 0
    # Size of window to apply attention to
    attn_window_size: Optional[int] = None

    # Options
    attn_window_option: AttentionWindowOption = AttentionWindowOption.DISABLED
    # no. of frames to use for positional embedding
    timestep_embedding_frames: Optional[float] = None
    shuffle_windowed_noise: bool = False
    temporal_attn_scale: float = 1.0

    attn_k_scale: float = 1.0
    attn_q_scale: float = 1.0
    attn_v_scale: float = 1.0

    original_forwards: Dict[str, Callable] = {}

    _instance = None

    @classmethod
    def instance(cls):
        if cls._instance is None:
            cls._instance = cls()
        return cls._instance


def set_model_patch_replace(model, key):
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
        for id in range(11):
            set_model_patch_replace(model, ("input", id, 0))
        set_model_patch_replace(model, ("middle", 0, 0))
        for id in range(12):
            set_model_patch_replace(model, ("output", id, 0))

    # patch SpatialVideoTransformer
    for layer_name, module in model.model.named_modules():
        if isinstance(module, SpatialVideoTransformer):
            if unpatch and state.original_forwards:
                module.forward = state.original_forwards[layer_name]
            else:
                state.original_forwards[layer_name] = module.forward
                module.forward = partial(patched_forward, module)


def patch_comfy_sample(orig_comfy_sample: Callable) -> Callable:
    def sample_svd(model: ModelPatcher, noise: T, *args, **kwargs):
        out = orig_comfy_sample(model, noise, *args, **kwargs)
        patch_model(model, True)
        return out

    return sample_svd


def exists(val):
    return val is not None


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


def attn_windowed(q: T, k: T, v: T, extra_options: dict) -> T:
    state = WindowState.instance()
    window_option = state.attn_window_option
    n_heads = extra_options["n_heads"]
    temporal = extra_options.get("temporal", False)

    is_attn1 = k.shape[1] == state.video_total_frames
    temporal_scale = state.temporal_attn_scale

    if (
        not temporal
        or window_option == AttentionWindowOption.DISABLED
        or (window_option != AttentionWindowOption.DISABLED and is_attn1)
    ):
        out = attention_xformers_scaling(q, k, v, heads=n_heads)
        return out

    q = q * state.attn_q_scale
    v = v * state.attn_v_scale
    k = k * state.attn_k_scale

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
                q_t, k_t, v_t, heads=n_heads, scale=temporal_scale
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

            attn_out = attention_xformers_scaling(
                q, k, v_t, n_heads, scale=temporal_scale
            )
            out += attn_out * t_mask
    return out


def attn_basic(q: T, k: T, v: T, extra_options: dict):
    heads = extra_options["n_heads"]
    return attention_xformers_scaling(q, k, v, heads=heads)


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

    frames_end = state.timestep_embedding_frames if state.timestep_embedding_frames else timesteps
    num_frames = torch.linspace(0, frames_end, timesteps, device=x.device)
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
