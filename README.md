# Video Distortion Removal for VapourSynth
Also known as atmospheric turbulance mitigation, warp stabilization, film shrink or VHS distortion fix, dewobble, dewiggle, detilt, rectification, heat haze removal. Can help with distortions from low bitrate compression or old codecs like MPEG2.

This is a partial implementation of the [Turbulence Mitigation Transformer](https://github.com/xg416/TMT). (only distortion removal, no deblurring)

This does not do general video stabilization for shaky footage, only removes distortions within the frames. It is recommented to stabilize first if needed.

__Check out stinkybread's comparisons [here](https://kaizoku.pw/c/tezc1iieeHympQGQeWKl) and [here](https://kaizoku.pw/c/4HsCLWDxuFMNpOIdSNI8).__ (chrome recommended for smooth playback)

<p align="center">
    <img src="https://github.com/xg416/TMT/blob/main/figs/video_22.gif"/>
</p>

### Requirements
* `pip install numpy` *(optional, only for pytorch backend)*
* [pytorch with cuda](https://pytorch.org/) *(optional, only for pytorch backend)*
* [vs-mlrt with tensorrt](https://github.com/AmusementClub/vs-mlrt) *(optional, only for tensorrt backend)*

### Setup
Put the entire `vs_undistort` folder into your vapoursynth scripts folder.  
Or install via pip: `pip install -U git+https://github.com/pifroggi/vs_undistort.git`

<br />

## PyTorch Backend
The PyTorch backend offers some extra control knobs and supports any CPU and Nvidia GPU, but is slower and requires more VRAM.

```python
import vs_undistort
clip = vs_undistort.pytorch(clip, temp_window=10, tile_width=None, tile_height=None, overlap=None, scales=[True, True, True], interpolation="bilinear", device="cuda")
```

__*`clip`*__  
Distorted clip. Must be in RGB format.

__*`temp_window`*__  
Temporal window length. How many frames are grouped together and processed as a single chunk. Larger means higher VRAM requirements, but better temporal averaging and slower distortions can be removed. If this is too small, some distortions may not get removed, small jumps/hitches may be visible between windows and seams from tile size may become more obvious.

__*`tile_width`*, *`tile_height`* (optional)__  
Size of tiles to split the frames into. Must be a multiple of 16. By default the full frame dimensions are used.  
Larger means higher VRAM requirements, but better spatial averaging and larger distortions can be removed.

__*`overlap`* (optional)__  
Overlap from one tile to the next. Use if seams between tiles are visible.

__*`interpolation`* (optional)__  
Interpolation mode used for warping the frames.  
Mode "bilinear" is a bit faster, but slightly blurry.  
Mode "bicubic" is a bit slower and may oversharpen slightly, but no blur.

__*`scales`* (optional)__  
Sets which distortion scales get fixed via `scales=[True, True, True]`, which stands for `[coarse, middle, fine]`. Set one or more to False to disable them. This is an experimental feature and may get removed if it turns out to be useless.

__*`device`* (optional)__  
Can be "cuda" to use with an Nvidia GPU, or "cpu". This will be extremely slow on CPU.

<br />

## TensorRT Backend
The TensorRT backend is much faster and requires less VRAM, but only supports bilinear interpolation and requires an Nvidia RTX GPU. On the first run, it will automatically build an engine, which takes a few minutes. Changing tile size or temporal window length will trigger rebuilding, but previously build engines are saved for later.

```python
import vs_undistort
clip = vs_undistort.tensorrt(clip, temp_window=10, tile_width=None, tile_height=None, overlap=None, num_streams=1)
```

__*`clip`*__  
Distorted clip. Must be in RGBH format.

__*`temp_window`*__  
Temporal window length. How many frames are grouped together and processed as a single chunk. Larger means higher VRAM requirements, but better temporal averaging and slower distortions can be removed. If this is too small, some distortions may not get removed, small jumps/hitches may be visible between windows and seams from tile size may become more obvious.

__*`tile_width`*, *`tile_height`* (optional)__  
Size of tiles to split the frames into. Must be a multiple of 16. By default the full frame dimensions are used.  
Larger means higher VRAM requirements, but better spatial averaging and larger distortions can be removed.

__*`overlap`* (optional)__  
Overlap from one tile to the next. Use if seams between tiles are visible.

__*`num_streams`* (optional)__  
Number of parallel TensorRT streams. For high end GPUs higher can be faster, but requires more VRAM.

<br />
<br />

> [!TIP]
> * If you see jumps/hitches between temporal windows, you can crossfade them with [vs_tiletools](https://github.com/pifroggi/vs_tiletools?tab=readme-ov-file#fix-jumpshitches-on-chunktemporal-window-based-filters-via-crossfading).
> * If you have an undistorted reference clip, you can also try to align to it with [vs_align](https://github.com/pifroggi/vs_align).

## Benchmarks

| Hardware | Resolution  | PyTorch FPS | TensorRT FPS
| -------- | ----------- | ----------- | ------------
| RTX 4090 | 720x480     | ~6.5 fps    | ~32 fps
| RTX 4090 | 1440x1080   | ~1.5 fps    | ~7 fps
| RTX 4090 | 2880x2160   | ~0.5 fps    | ~2 fps
