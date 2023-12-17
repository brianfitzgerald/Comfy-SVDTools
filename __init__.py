import torch
from comfy.model_patcher import ModelPatcher
import comfy.samplers
from nodes import common_ksampler

T = torch.Tensor


class WindowedAttentionProcessor:
    def __init__(self, t_start: int, t_end: int):
        self.t_start = t_start
        self.t_end = t_end

    def __call__(self, q, k, v, extra_options):
        print("attn with", self.t_start, self.t_end)
        return q, k, v


def get_attn_windows(video_length: int, window_size: int = 16, stride: int = 4):
    num_blocks_time = (video_length - window_size) // stride + 1
    views = []
    for i in range(num_blocks_time):
        t_start = int(i * stride)
        t_end = t_start + window_size
        views.append((t_start, t_end))
    return views


class KSamplerFreeNoise:
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
        latent_tensor = latent_image["samples"]
        video_num_frames = latent_tensor.shape[0]
        attn_windows = get_attn_windows(video_num_frames, window_size, window_stride)
        m = model.clone()
        latents = latent_tensor
        out_latents = torch.zeros_like(latents)
        out_dict = {}
        print(f"computing {len(attn_windows)} windows")
        for i in range(len(attn_windows)):
            t_start, t_end = attn_windows[i]
            print(f"process window {i}/{len(attn_windows)} ({t_start}, {t_end})")
            m.set_model_attn1_patch(WindowedAttentionProcessor(t_start, t_end))
            latents = common_ksampler(
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
            )
            breakpoint()
            out_latents[t_start:t_end] = latents[t_start:t_end]
            out_dict["samples"] = out_latents
            return (out_dict)

        return (out_dict)


NODE_CLASS_MAPPINGS = {
    "KSamplerFreeNoise": KSamplerFreeNoise,
}
