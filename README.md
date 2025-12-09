# Video Distortion Removal for VapourSynth
Also known as atmospheric turbulance mitigation, warp stabilization, film shrink or VHS distortion fix, dewobble, dewiggle, detilt, rectification, heat haze removal. Can help with distortions from low bitrate compression or old codecs like MPEG2.

This does not do general video stabilization for shaky footage, only removes distortions within the frames. It is recommented to stabilize first if needed.

This is a partial implementation of the [Turbulence Mitigation Transformer](https://github.com/xg416/TMT). (only distortion removal, no deblurring)

<p align="center">
    <img src="https://github.com/xg416/TMT/blob/main/figs/video_22.gif"/>
</p>

<br />

## Requirements
* `pip install numpy` *(optional, only for pytorch backend)*
* [pytorch with cuda](https://pytorch.org/) *(optional, only for pytorch backend)*
* [vs-mlrt with tensorrt](https://github.com/AmusementClub/vs-mlrt) *(optional, only for tensorrt backend)*

## Setup
Put the entire `vs_undistort` folder into your vapoursynth scripts folder.  
Or install via pip: `pip install -U git+https://github.com/pifroggi/vs_undistort.git`

<br />

## Pytorch Backend
The Pytorch backend offers more control and supports any CPU and Nvidia GPU, but is slower and requires more VRAM.

```python
import vs_undistort
clip = vs_undistort.pytorch(clip, temp_window=10, tile_width=None, tile_height=None, overlap=None, scales=[True, True, True], interpolation="bilinear", device="cuda")
```

__*`clip`*__  
Distorted clip. Must be in RGB format.

__*`temp_window`*__  
Temporal window length. How many frames are grouped together and processed as a single chunk. Larger means higher VRAM requirements, but better temporal averaging and slower distortions can be removed. If this is too small, some distortions may not get removed, small jumps/hitches may be visible between windows and seams from tile size may become more obvious.

__*`tile_width`*, *`tile_height`*__  
Size of tiles to split the frames into. Must be a multiple of 16.  
Larger means higher VRAM requirements, but better spatial averaging and larger distortions can be removed.

__*`overlap`*__  
Overlap from one tile to the next. Increase if seams between tiles are visible.

__*`interpolation`*__  
Interpolation mode used for warping the frames.  
Mode "bilinear" is a bit faster, but slightly blurry.  
Mode "bicubic" is a bit slower and may oversharpen slightly, but no blur.

__*`scales`*__  
Sets which distortion scales should be fixed via `scales=[True, True, True]`, which stands for `[coarse, middle, fine]`. Set one or more to False to disable them. This is an experimental feature and may get removed if it turns out to be useless.

__*`device`*__  
Possible values are "cuda" to use with an Nvidia GPU, or "cpu". This will be extremely slow on CPU.

<br />

## TensorRT Backend
The TensorRT backend is much faster and requires less VRAM, but lacks a few controls and requires an Nvidia RTX GPU. On the first run, it will automatically build an engine, which takes a few minutes. Changing tile size or temporal window length will trigger rebuilding, but engines with old settings are saved.

```python
import vs_undistort
clip = vs_undistort.tensorrt(clip, temp_window=10, tile_width=None, tile_height=None, overlap=None, num_streams=1)
```

__*`clip`*__  
Distorted clip. Must be in RGBH format.

__*`temp_window`*__  
Temporal window length. How many frames are grouped together and processed as a single chunk. Larger means higher VRAM requirements, but better temporal averaging and slower distortions can be removed. If this is too small, some distortions may not get removed, small jumps/hitches may be visible between windows and seams from tile size may become more obvious.

__*`tile_width`*, *`tile_height`*__  
Size of tiles to split the frames into. Must be a multiple of 16.  
Larger means higher VRAM requirements, but better spatial averaging and larger distortions can be removed.

__*`overlap`*__  
Overlap from one tile to the next. Increase if seams between tiles are visible.

__*`num_streams`*__  
How many streams to process in parallel. Higher can be faster, but requires more VRAM.

<br />
<br />
<br />

> [!TIP]
> * If you are getting *`RuntimeError: CUDA error: invalid argument`* you are likely running out of GPU memory. Try lowering the tile size or the temporal window length.
> * If you have an undistorted reference clip, try to align to it with [vs_align](https://github.com/pifroggi/vs_align).
> * If you see jumps/hitches between temporal windows, you can crossfade them with [vs_tiletools](https://github.com/pifroggi/vs_tiletools) like this:
>   ```python
>   clip = vs_tiletools.window(clip, length=10, overlap=4)  # creates a temporal overlap of 4 frames
>   clip = vs_undistort.pytorch(clip, temp_window=10)
>   clip = vs_tiletools.unwindow(clip, fade=True)  # uses the overlap to fade between temporal windows
>   ```

<br />

## Benchmarks

| Hardware | Resolution  | Pytorch FPS | TensorRT FPS
| -------- | ----------- | ----------- | ------------
| RTX 4090 | 720x480     | ~6.5 fps    | ~32 fps
| RTX 4090 | 1440x1080   | ~1.5 fps    | ~7 fps
| RTX 4090 | 2880x2160   | ~0.5 fps    | ~2 fps
