# Comfy-SVDTools

A collection of techniques that extend the functionality of Stable Video Diffusion in ComfyUI. Most of these were investigated for the purpose of extending context length; though they may be useful for other purposes as well.

I've divided the functionality into two nodes: `SVDToolsPatcher` and `SVDToolsPatcherExperimental`. Techniques in `SVDToolsPatcher` are marked as 'Experimental' below and may change or be removed in the future. Techniques in `SVDToolsPatcher` tend to give good results, and probably won't change.

## Techniques

### Position Embedding Scaling

Similar to [YaRN](https://arxiv.org/abs/2309.00071) for language models, this technique scales the position embeddings in the `SpatialVideoTransformer` layers to match a set embedding length. For example, if `position_embedding_frames` is set to 12, but the batch size is 42, the model will generate video with 42 frames, but the position embeddings will be scaled to 12 frames. This allows the model to generate video with a longer context length than the position embeddings would normally allow.

<table>
  <thead>
		<th>Base (48 frames with SVD)</th>
		<th>With Position Embedding Scaling</th>
		<th>With Key and Pos. Emb. Scaling</th>
	</thead>
	<tr>
		<td><video width="100" controls src="./resources/control_00002.webm" muted="false"></video></td>
	</tr>
</table>

#### Settings

- `scale_timestep_embedding`: Enable / disable position embedding scaling.
- `position_embedding_frames`: The number of frames to scale the position embeddings to. The model will be conditioned as if it were generating video with this many frames, but will actually generate video with the number of frames in the batch.

### Key Scaling

Scales the keys only for temporal attention. Consistently leads to less jittering at higher motion bucket IDs, especially with long context windows.

#### Settings
- `temporal_attn_k_scale`: Higher leads to more movement, lower leads to less movement. A value of 1.0 is the same as the default attention scaling.

### Attention Windowing (Experimental)

Following the [FreeNoise](http://haonanqiu.com/projects/FreeNoise.html) paper, this technique uses a windowed attention mechanism to only compute cross-attention in each temporal layer for a subset of the total latents. 

#### Settings

- `attn_window_size`: The size of the window to use for attention. This is the number of latents to attend to in each layer.
- `attn_window_stride`: The stride of the window. This is the number of latents to skip between each window, i.e. a stride of 6 with a window size of 12 will attend to latents 0-11, 6-17, 12-23, etc.
- `shuffle_windowed_noise`: Shuffles the initial batch of latents. This is a technique mentioned in the FreeNoise paper, and can sometimes help with inter-batch stability.

### Temporal Attention Scale (Experimental)

An implementation of [Jonathan Fischoff's technique](https://jfischoff.github.io/blog/motion_control_with_attention_scaling.html) for scaling the attention in each temporal layer. This scales the self attention values by `sqrt(scale/dim_head)`.

#### Settings
- `temporal_attn_scale`: Higher leads to more movement, lower leads to less movement. A value of 1.0 is the same as the default attention scaling.

## How to Use

Simply download or git clone this repository in `ComfyUI/custom_nodes`. An example pipeline is provided in the `resources` folder in this repo.

## Limitations
- `xformers` must be installed; this is temporary, until the `scale` parameter is added to the self-attention nodes in ComfyUI.
- The `SVDToolsPatcher` nodes override the Comfy `comfy.sample.sample` function, in order to unpatch the `forward` method of `SpatialVideoTransformer`. This may cause issues with other custom sample nodes. This is done as there's no way to patch the `forward` method of `SpatialVideoTransformer` using `ModelPatcher` - if this is added to Comfy in the future, this override will be removed.

## Up Next

Techniques I'm either currently working on implementing or plan to implement in the future:

- [ ] [FreeInit](https://arxiv.org/abs/2312.07537)
- [ ] Motion transfer, following [the FreeNoise implementation](http://haonanqiu.com/projects/FreeNoise.html)
- [ ] Looping mode (overlap the first and last windows)
- [ ] Text conditioning interpolation / blending