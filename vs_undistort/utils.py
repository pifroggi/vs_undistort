
# Script by pifroggi https://github.com/pifroggi/vs_undistort
# or tepete and pifroggi on Discord

import math
import vapoursynth as vs

core = vs.core


def expression(clips, expr, format=None):
    # backend for basic exprs supported by std.Expr
    if hasattr(core, "akarin"):
        return core.akarin.Expr(clips, expr, format=format)
    else:
        return core.std.Expr(clips, expr, format=format)


def get_window(clip, temp_window):
    # how many pad frames to reach a multiple of temp_window
    num_frames = clip.num_frames
    pad = (-num_frames) % temp_window

    if pad:
        # pad black frames if too short
        pad_clip = core.std.BlankClip(clip=clip, length=pad)
        clip     = core.std.Splice([clip, pad_clip])

    # offset_clips[i] contains frames i, i+temp_window, i+2*temp_window, ...
    return [core.std.SelectEvery(clip[i:], cycle=temp_window, offsets=[0]) for i in range(temp_window)]


def trim_overlaps(clip, full_length, temp_window, window_overlap):
    if window_overlap == 0 or clip.num_frames <= temp_window:
        return clip[:full_length]
    remaining = core.std.SelectEvery(clip[temp_window:], cycle=temp_window, offsets=list(range(window_overlap, temp_window)), modify_duration=False)
    return core.std.Splice([clip[:temp_window], remaining])[:full_length]


def insert_overlaps(clip, temp_window, window_overlap):
    stride = temp_window - window_overlap
    length = clip.num_frames

    if window_overlap == 0:
        return clip
    if length <= temp_window:
        return clip

    window_count = 1 + (length - window_overlap - 1) // stride
    phases = []

    if stride == 1:  # selectevery requires cycle > 1
        phases = [clip[offset:] for offset in range(temp_window)]
    else:
        for offset in range(temp_window):
            phase = core.std.SelectEvery(clip, cycle=stride, offsets=offset % stride, modify_duration=False)
            skip = offset // stride
            if skip:
                phase = phase[skip:]
            phases.append(phase)

    overlapped = core.std.Interleave(phases, modify_duration=False)
    last_start = (window_count - 1) * stride
    output_length = (window_count - 1) * temp_window + min(temp_window, length - last_start)
    return overlapped[:output_length]


def get_tiles(clip_w, clip_h, tiles, overlap=0):
    # calculate tile size and choose the most square layout
    if tiles not in (1, 2, 4, 6, 8, 12, 16, 24, 32):
        raise ValueError("vs_undistort: Tiles must be 1, 2, 4, 6, 8, 12, 16, 24, or 32.")

    layouts = {
        1: [(1, 1)],
        2: [(2, 1), (1, 2)],
        4: [(4, 1), (2, 2), (1, 4)],
        6: [(6, 1), (3, 2), (2, 3), (1, 6)],
        8: [(8, 1), (4, 2), (2, 4), (1, 8)],
        12: [(12, 1), (6, 2), (4, 3), (3, 4), (2, 6), (1, 12)],
        16: [(16, 1), (8, 2), (4, 4), (2, 8), (1, 16)],
        24: [(24, 1), (12, 2), (8, 3), (6, 4), (4, 6), (3, 8), (2, 12), (1, 24)],
        32: [(32, 1), (16, 2), (8, 4), (4, 8), (2, 16), (1, 32)],
    }[tiles]

    def _tile_size(layout):
        cols, rows = layout
        tile_w = math.ceil(math.ceil((clip_w + 2 * overlap * (cols - 1)) / cols) / 16) * 16
        tile_h = math.ceil(math.ceil((clip_h + 2 * overlap * (rows - 1)) / rows) / 16) * 16
        return tile_w, tile_h

    def _layout_valid(layout):
        # tiles must have a positive non overlapped stride
        cols, rows = layout
        tile_w, tile_h = _tile_size(layout)
        if cols > 1 and tile_w <= 2 * overlap:
            return False
        if rows > 1 and tile_h <= 2 * overlap:
            return False
        return True

    def _score(layout):
        tile_w, tile_h = _tile_size(layout)
        cols, rows = layout
        tile_aspect = tile_w / tile_h
        square_error = abs(math.log(tile_aspect))
        orientation_penalty = rows if clip_w >= clip_h else cols
        balance_penalty = abs(cols - rows)
        return (square_error, orientation_penalty, balance_penalty)

    valid_layouts = [layout for layout in layouts if _layout_valid(layout)]
    if not valid_layouts:
        raise ValueError("vs_undistort: Clip dimensions are too small for current tile amount. Reduce tiles or overlap.")

    cols, rows = min(valid_layouts, key=_score)
    tile_w, tile_h = _tile_size((cols, rows))
    if tile_w > 2 * tile_h or tile_h > 2 * tile_w:
        raise ValueError("vs_undistort: The current tile amount produces tiles that are too elongated. Try a different tile amount.")

    return tile_w, tile_h


def get_tile_positions(h, w, s_h, s_w, overlap=0):
    if overlap is None:
        overlap = 0
    
    def _axis_positions(size, patch, overlap):
        
        # single patch or smaller image = no tiling / overlap needed
        if patch >= size:
            return [0]

        # overlap is removed from both sides of the tile stride to match vsmlrt
        step      = patch - 2 * overlap
        max_start = size - patch

        if step <= 0 or max_start <= 0:
            return [0]

        positions = [0]
        while positions[-1] < max_start:
            positions.append(min(positions[-1] + step, max_start))

        return positions

    hpos = _axis_positions(h, s_h, overlap)
    wpos = _axis_positions(w, s_w, overlap)
    return hpos, wpos


def inference_tiled(input_blk, model_tilt, patch_height, patch_width, overlap=0, scales=[True, True, True], tile_device=None):
    import torch
    tile_device   = torch.device("cpu") if tile_device is None else tile_device
    model_device  = next(model_tilt.parameters()).device
    b, l, c, h, w = input_blk.shape
    
    hpos, wpos = get_tile_positions(h, w, patch_height, patch_width, overlap)
    out_spaces = torch.empty_like(input_blk, device=tile_device)                           # (B, L, C, H, W)

    for hi in hpos:
        y_crop_start = 0 if hi == 0 else overlap
        y_crop_end   = 0 if hi == h - patch_height else overlap
        for wi in wpos:
            x_crop_start = 0 if wi == 0 else overlap
            x_crop_end   = 0 if wi == w - patch_width else overlap
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
            
            out_spaces[..., hi + y_crop_start:hi + hs - y_crop_end, wi + x_crop_start:wi + ws - x_crop_end].copy_(rectified[..., y_crop_start:hs - y_crop_end, x_crop_start:ws - x_crop_end])

    return out_spaces
