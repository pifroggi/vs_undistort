
























# Video Distortion Removal for Vapoursynth
Also known as atmospheric turbulance mitigation, warp stabilization, film shrink distortion fix, dewobble, dewiggle, detilt, rectification, heat haze removal.

This does not do general video stabilization for shaky footage, only removes distortions within the frames. It is recommented to stabilize first if needed.

This is a partial implementation of the [Turbulence Mitigation Transformer](https://github.com/xg416/TMT). (only distortion removal, no deblurring)

<p align="center">
    <img src="https://github.com/xg416/TMT/blob/main/figs/video_22.gif"/>
</p>

<br />

## Requirements
* pip install numpy
* pip install einops
* [pytorch](https://pytorch.org/)

## Setup
Drop the entire "vs_undistort" folder to where you typically load scripts from.

## Usage

    from vs_undistort import vs_undistort
    clip = vs_undistort(clip, temp_window=10, tile_size=480, device="cuda")

__*clip*__  
Distorted clip. Must be in RGBS format.

__*temp_window*__  
Temporal window. Amount of frames to include in the calculation and size of chunks the clip will be processed in.  
Larger means higher VRAM requirements, but better temporal averaging effect and slower distortions can be removed. If this is too small, some distortions may not get removed and seams from tile_size may become more obvious.  

__*tile_size*__  
Size of tiles to split the frames into. Must be a multiple of 16.  
Larger means higher VRAM requirements, but better spatial averaging effect and larger/lower frequency distortions can be removed. If distortions are larger than tile_size, they can not be removed. If only some distortions are larger than tile_size, visible seams may appear.  

__*device*__  
Possible values are "cuda" to use with an Nvidia GPU, or "cpu". This will be extremely slow on CPU.

## Tips
If you have an undistorted reference clip, try this: https://github.com/pifroggi/vs_align
