# Code from "Turbulence Mitigation Transformer" https://github.com/xg416/TMT

# Vapoursynth Implementation by pifroggi https://github.com/pifroggi/vs_undistort
# or tepete and pifroggi on Discord

import torch
import math


def split_to_patches(h, w, s_h, s_w, min_overlap=0):
    if min_overlap is None:
        min_overlap = 0
    
    def _axis_positions(size, patch, min_ol):
        
        # single patch or smaller image = no tiling / overlap needed
        if patch >= size:
            return [0]

        # clamp min_overlap to [0, patch-1]
        min_ol = max(0, min(min_ol, patch - 1))

        # original behaviour without min_overlap
        if min_ol == 0:
            n = size // patch + (1 if size % patch != 0 else 0)
            if n == 1:
                return [0]
            overlap = int((n * patch - size) / (n - 1))
            step = patch - overlap
            positions = [0]
            for _ in range(1, n):
                positions.append(min(positions[-1] + step, size - patch))
            return positions

        # when minimum overlap > 0:
        max_start = size - patch          # last start index
        step_max = patch - min_ol         # max stride to keep >= min_ol overlap

        if step_max <= 0 or max_start <= 0:
            return [0]

        # smallest n such that max step <= step_max
        n = int(math.ceil(max_start / step_max)) + 1
        if n <= 1:
            return [0]

        positions = []
        for i in range(n):
            pos = (i * max_start) // (n - 1)
            positions.append(pos)

        return positions

    hpos = _axis_positions(h, s_h, min_overlap)
    wpos = _axis_positions(w, s_w, min_overlap)
    return hpos, wpos


def test_spatial_overlap(input_blk, model_tilt, patch_height, patch_width, min_overlap=0, scales=[True, True, True], tile_device=torch.device("cpu")):
    model_device  = next(model_tilt.parameters()).device
    b, l, c, h, w = input_blk.shape
    
    hpos, wpos = split_to_patches(h, w, patch_height, patch_width, min_overlap)
    out_spaces = torch.zeros_like(input_blk,  device=tile_device)                          # (B, L, C, H, W)
    out_counts = torch.zeros((b, 1, 1, h, w), device=tile_device, dtype=torch.int16)       # (B, 1, 1, H, W)
    ones_count = None

    for hi in hpos:
        for wi in wpos:
            inp = input_blk[..., hi:hi + patch_height, wi:wi + patch_width]                # (B, L, C, ph, pw)
            
            # move only the tile to the model device
            if inp.device != model_device:
                non_blocking = (inp.device.type == "cpu" and model_device.type == "cuda" and inp.is_pinned())
                inp = inp.to(model_device, non_blocking=non_blocking)
            
            rectified = model_tilt(inp, scales=scales)
            hs, ws = rectified.shape[-2:]
            
            # move tile result back to tile device
            if rectified.device != tile_device:
                rectified = rectified.to(tile_device)
            
            out_spaces[..., hi:hi + hs, wi:wi + ws].add_(rectified)
            
            if ones_count is None or ones_count.shape[-2:] != (hs, ws):
                ones_count = torch.ones((1, 1, 1, hs, ws), device=tile_device, dtype=out_counts.dtype)

            out_counts[..., hi:hi + hs, wi:wi + ws].add_(ones_count)

    return out_spaces / out_counts.to(out_spaces.dtype)


def process_images(frames_tensor, patch_height, patch_width, model_tilt, min_overlap=0, scales=[True, True, True], tile_device=torch.device("cpu")):
    T, C, H, W = frames_tensor.shape
    input_blk  = frames_tensor.unsqueeze(0)                                                # (1, T, C, H, W)

    with torch.inference_mode():
        recovered = test_spatial_overlap(input_blk, model_tilt, patch_height, patch_width, min_overlap=min_overlap, scales=scales, tile_device=tile_device)

    recovered = recovered[0]
    stacked   = recovered.permute(2, 0, 3, 1).contiguous().reshape(H, T * W, C)  # output one large frame to let vs handle the caching
    return stacked
