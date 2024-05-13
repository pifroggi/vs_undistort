
# Script to run "Turbulence Mitigation Transformer" https://github.com/xg416/TMT

# Vapoursynth Implementation by pifroggi
# or tepete on the "Enhance Everything!" Discord Server

import vapoursynth as vs
import os
import torch
import numpy as np
from vs_undistort.TMT_dynamic_1st_stage import process_images
from vs_undistort.UNet3d_TMT import DetiltUNet3DS

core = vs.core

def stack_frames(clip, temp_window, frame_width, frame_height):
    length = clip.num_frames
    stacked_clips = []
    #stack frames
    for i in range(0, length, temp_window):
        if i + temp_window <= length:
            frames = [clip[j] for j in range(i, i + temp_window)]
            stacked_frame = core.std.StackHorizontal(clips=frames)
        else:
            #if last frames not enough, add border as replacement
            remaining_frames = length % temp_window
            frames = [clip[j] for j in range(i, i + remaining_frames)]
            stacked_frame = core.std.StackHorizontal(clips=frames)
            border_width = (temp_window - remaining_frames) * frame_width
            stacked_frame = core.std.AddBorders(stacked_frame, right=border_width, color=[0.5, 0.5, 0.5])
        stacked_clips.append(stacked_frame)    
    #build clip from stacked frames
    return core.std.Splice(clips=stacked_clips, mismatch=True)

def split_stacked_frames(stacked_clip, frame_width, temp_window, frame_height):
    frames = []
    for i in range(stacked_clip.num_frames):
        #extract original frames from stacked frames
        for j in range(temp_window):
            crop_left = frame_width * j
            frame = core.std.CropAbs(clip=stacked_clip[i], left=crop_left, top=0, width=frame_width, height=frame_height)
            frames.append(frame)
    #rebuild original clip
    return core.std.Splice(clips=frames, mismatch=True)

def frame_to_array(frame: vs.VideoFrame) -> np.ndarray:
    return np.dstack([np.asarray(frame[p]) for p in range(frame.format.num_planes)])

def array_to_frame(img: np.ndarray, frame: vs.VideoFrame):
    for p in range(frame.format.num_planes):
        pls = frame[p]
        frame_arr = np.asarray(pls)
        np.copyto(frame_arr, img[:, :, p])

def load_model(device):
    model_tilt = DetiltUNet3DS(norm='LN', residual='pool', conv_type='dw').to(device)
    path_tilt = os.path.join('vs_undistort', 'dynamic_1st_stage.pth')
    if os.path.exists(path_tilt):
        ckpt_tilt = torch.load(path_tilt, map_location=device)
        model_tilt.load_state_dict(ckpt_tilt['state_dict'] if 'state_dict' in ckpt_tilt else ckpt_tilt)
    model_tilt.eval()
    return model_tilt

#convert frame to numpy array, process, convert back
def process_clip(clip: vs.VideoNode, patch_size: int, temp_window: int, model_tilt, device):
    def process_frame(n, f):
        source_frame = frame_to_array(f)
        processed_frame_np = process_images(source_frame, patch_size, temp_window, model_tilt, device)
        fout = f.copy()
        array_to_frame(processed_frame_np, fout)
        return fout
    processed_clip = core.std.ModifyFrame(clip, clips=[clip], selector=process_frame)
    return processed_clip

def vs_undistort(clip, temp_window=10, tile_size=480, device="cuda"):
    device = torch.device(device)

    #checks
    if clip.format.id not in [vs.RGBS]:
        raise ValueError("Input clip must be in RGBS format.")

    #orginal dimensions
    original_width = clip.width
    original_height = clip.height

    #load model
    model_tilt = load_model(device)

    #stack, process, unstack
    stacked_clip = stack_frames(clip, temp_window, original_width, original_height)
    stacked_clip = process_clip(stacked_clip, tile_size, temp_window, model_tilt, device)
    unstacked_clip = split_stacked_frames(stacked_clip, original_width, temp_window, original_height)
    return core.std.Trim(unstacked_clip, last=clip.num_frames-1)
