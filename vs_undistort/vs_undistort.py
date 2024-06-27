
# Script to run "Turbulence Mitigation Transformer" https://github.com/xg416/TMT

# Vapoursynth Implementation by pifroggi
# or tepete on the "Enhance Everything!" Discord Server

import vapoursynth as vs
import os
import torch
import numpy as np
from .TMT_dynamic_1st_stage import process_images
from .UNet3d_TMT import DetiltUNet3DS

core = vs.core

def stack_frames(clip, temp_window, frame_width, frame_height):
    length = clip.num_frames
    stacked_clips = []
    border_color = [0.5, 0.5, 0.5]

    for i in range(0, length, temp_window):
        remaining_frames = min(temp_window, length - i)
        frames = [clip[j] for j in range(i, i + remaining_frames)]
        
        if remaining_frames < temp_window:
            border_width = (temp_window - remaining_frames) * frame_width
            stacked_frame = core.std.AddBorders(core.std.StackHorizontal(clips=frames), right=border_width, color=border_color)
        else:
            stacked_frame = core.std.StackHorizontal(clips=frames)
        
        stacked_clips.append(stacked_frame)
    return core.std.Splice(clips=stacked_clips, mismatch=True)

def split_stacked_frames(stacked_clip, frame_width, temp_window, frame_height):
    frames = []
    crop_params = [(frame_width * j, frame_width, frame_height) for j in range(temp_window)]

    for i in range(stacked_clip.num_frames):
        stacked_frame = stacked_clip[i]
        for crop_left, width, height in crop_params:
            frame = core.std.CropAbs(clip=stacked_frame, left=crop_left, top=0, width=width, height=height)
            frames.append(frame)

    return core.std.Splice(clips=frames, mismatch=True)

def frame_to_tensor(frame: vs.VideoFrame, device: str) -> torch.Tensor:
    array = np.empty((frame.height, frame.width, 3), dtype=np.float32)
    for p in range(frame.format.num_planes):
        array[..., p] = np.asarray(frame[p], dtype=np.float32)
    tensor = torch.from_numpy(array).to(device)
    return tensor.clamp_(0, 1)

def tensor_to_frame(tensor: torch.Tensor, frame: vs.VideoFrame):
    tensor_np = tensor.cpu().numpy()
    for p in range(frame.format.num_planes):
        frame_arr = np.asarray(frame[p])
        np.copyto(frame_arr, tensor_np[:, :, p])

def load_model(device):
    model_tilt = DetiltUNet3DS(norm='LN', residual='pool', conv_type='dw').to(device)
    current_folder = os.path.dirname(os.path.abspath(__file__))
    path_tilt = os.path.join(current_folder, 'dynamic_1st_stage.pth')
    ckpt_tilt = torch.load(path_tilt, map_location=device)
    model_tilt.load_state_dict(ckpt_tilt['state_dict'] if 'state_dict' in ckpt_tilt else ckpt_tilt)
    model_tilt.eval()
    return model_tilt

def vs_undistort(clip, temp_window=10, tile_size=480, device="cuda"):

    #checks
    if clip.format.id not in [vs.RGBS]:
        raise ValueError("Input clip must be in RGBS format.")
    if tile_size % 16 != 0:
        raise ValueError("tile_size must be a multiple of 16.")

    original_width = clip.width
    original_height = clip.height
    device = torch.device(device)
    model_tilt = load_model(device)

    def process_frame(n, f):
        source_frame = frame_to_tensor(f, device)
        processed_frame_tensor = process_images(source_frame, tile_size, temp_window, model_tilt)
        fout = f.copy()
        tensor_to_frame(processed_frame_tensor, fout)
        return fout

    #stack, process, unstack
    stacked_clip = stack_frames(clip, temp_window, original_width, original_height)
    stacked_clip = core.std.ModifyFrame(stacked_clip, clips=[stacked_clip], selector=process_frame)
    unstacked_clip = split_stacked_frames(stacked_clip, original_width, temp_window, original_height)
    return core.std.Trim(unstacked_clip, last=clip.num_frames - 1)
