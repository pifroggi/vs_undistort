
# Code from "Turbulence Mitigation Transformer" https://github.com/xg416/TMT

# Vapoursynth Implementation by pifroggi
# or tepete on the "Enhance Everything!" Discord Server

import torch
import torch.nn.functional as F

def fetch_images(image_tensor, temp_window, patch_unit=16):
    h, w, c = image_tensor.shape
    frame_width = w // temp_window
    frames = torch.empty((temp_window, c, h, frame_width), dtype=image_tensor.dtype, device=image_tensor.device)

    for i in range(temp_window):
        frames[i] = image_tensor[:, i * frame_width:(i + 1) * frame_width, :].permute(2, 0, 1)

    #padding
    padh = patch_unit - h % patch_unit if h % patch_unit != 0 else 0
    padw = patch_unit - frame_width % patch_unit if frame_width % patch_unit != 0 else 0
    frames = F.pad(frames, (0, padw, 0, padh), mode='reflect')

    return frames, h, frame_width, temp_window

def split_to_patches(h, w, s):
    nh = h // s + (1 if h % s != 0 else 0)
    nw = w // s + (1 if w % s != 0 else 0)
    ol_h = int((nh * s - h) / (nh - 1)) if nh > 1 else 0
    ol_w = int((nw * s - w) / (nw - 1)) if nw > 1 else 0
    hpos, wpos = [0], [0]
    for i in range(1, nh):
        hpos.append(min(hpos[-1] + s - ol_h, h - s))
    for i in range(1, nw):
        wpos.append(min(wpos[-1] + s - ol_w, w - s))
    return hpos, wpos

def test_spatial_overlap(input_blk, model_tilt, patch_size):
    b, c, l, h, w = input_blk.shape
    hpos, wpos = split_to_patches(h, w, patch_size)
    out_spaces = torch.zeros_like(input_blk)
    out_masks = torch.zeros((b, c, l, h, w), device=input_blk.device, dtype=input_blk.dtype)
    ones_patch = torch.ones((b, c, l, patch_size, patch_size), device=input_blk.device, dtype=input_blk.dtype)

    for hi in hpos:
        for wi in wpos:
            input_ = input_blk[..., hi:hi + patch_size, wi:wi + patch_size]
            _, _, rectified = model_tilt(input_.permute(0, 2, 1, 3, 4))
            rectified = rectified.permute(0, 2, 1, 3, 4)
            out_spaces[..., hi:hi + patch_size, wi:wi + patch_size] += rectified
            out_masks[..., hi:hi + patch_size, wi:wi + patch_size] += ones_patch[..., :rectified.shape[-2], :rectified.shape[-1]]

    out_masks[out_masks == 0] = 1
    return out_spaces / out_masks

def process_images(image_tensor, patch_size, temp_window, model_tilt):
    turb_imgs, h, frame_width, _ = fetch_images(image_tensor, temp_window, patch_size)
    turb_imgs = turb_imgs.unsqueeze(0).permute(0, 2, 1, 3, 4)

    with torch.no_grad():
        with torch.cuda.amp.autocast():
            recovered = test_spatial_overlap(turb_imgs, model_tilt, patch_size)
        recovered = recovered[..., :h, :frame_width].permute(0, 2, 1, 3, 4)

        stacked_image = torch.cat([recovered[0, i].permute(1, 2, 0) for i in range(recovered.shape[1])], dim=1)
        return stacked_image
