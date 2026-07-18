# Video Distortion Removal for VapourSynth
Also known as dewobble, warp stabilization, film or VHS distortion fix, rectification, atmospheric turbulence mitigation, or heat haze removal.

This is a partial implementation of the [Turbulence Mitigation Transformer](https://github.com/xg416/TMT) (only distortion removal, no deblurring). It does not do general video stabilization for shaky footage, only removes distortions within the frames. It is recommented to stabilize first if needed.

Check out stinkybread's comparisons [here](https://kaizoku.pw/c/tezc1iieeHympQGQeWKl) and [here](https://kaizoku.pw/c/4HsCLWDxuFMNpOIdSNI8).

<p align="center">
    <img src="https://github.com/xg416/TMT/blob/main/figs/video_22.gif"/>
</p>

<br />

## Installation

```
pip install -U vs_undistort
```
* To enable the CPU/CUDA backends, install [PyTorch with CUDA](https://pytorch.org/). *(optional)*
<br />

For older vapoursynth versions below R74, follow the manual installation steps [here](https://github.com/pifroggi/vs_undistort/wiki/Manual-Installation).

<br />

## Usage

```python
from vs_undistort import vs_undistort
clip = vs_undistort(clip, temp_window=10, tiles=1, overlap=8, interpolation="bicubic", backend="tensorrt", num_streams=1, engine_folder=None)
```

__*`clip`*__  
Distorted clip. Must be in RGBH format.

__*`temp_window`*__  
Temporal window length. How many frames are grouped together and processed as a single chunk. Larger means higher VRAM requirements, but better temporal averaging and slower distortions can be removed. If this is too small, some distortions may not get removed, small jumps/hitches may be visible between windows and seams from tiling may become more obvious.

__*`tiles`* (optional)__  
Amount of tiles to split the frames into. A higher amount reduces VRAM requirements, but also worsens spatial averaging. Default tiles=1 uses the full frame.

__*`overlap`* (optional)__  
Overlap from one tile to the next. Increase if seams between tiles are visible.

__*`interpolation`* (optional)__  
The interpolation mode used to warp the frames:
* `bilinear` More blurry.
* `bicubic` No blur, but may oversharpen slightly.

__*`backend`* (optional)__  
The backend used to run the model:
* `cpu` CPU mode using PyTorch *(very slow)*.
* `cuda` GPU mode using PyTorch with CUDA support. Requires any Nvidia GPU *(fast)*.
* `tensorrt` GPU mode using vs-mlrt with TensorRT support. Requires an Nvidia RTX GPU. On the first run, this mode will automatically build an engine, which may take a few minutes. Changing interpolation, temp_window, or input dimensions will trigger rebuilding, but previously build engines are stored *(very fast/low vram)*.

__*`num_streams`* (optional)__  
Number of parallel TensorRT streams. For high end GPUs higher can be a bit faster, but requires more VRAM. Only affects the TensorRT backend.

__*`engine_folder`* (optional)__  
Optional path to the TensorRT engine storage location. By default engines are stored in `vs_undistort/engines`. Only affects the TensorRT backend.

> [!TIP]
> * If you see jumps/hitches between temporal windows, you can crossfade them with [vs_tiletools](https://github.com/pifroggi/vs_tiletools?tab=readme-ov-file#fix-jumpshitches-on-chunktemporal-window-based-filters-via-crossfading).
> * If you have an undistorted reference clip, you can also try to align to it with [vs_align](https://github.com/pifroggi/vs_align).

<br />

## Benchmarks

| Hardware | Resolution | TensorRT | CUDA
|----------|------------|----------|---------
| RTX 4090 | 720x480    | ~35 fps  | ~6.5 fps
| RTX 4090 | 1440x1080  | ~7.5 fps | ~1.5 fps
| RTX 4090 | 2880x2160  | ~2 fps   | ~0.5 fps
