import torch
import torch.nn.functional as F
import numpy as np

def fetch_images(image_array, temp_window, patch_unit=16):
    h, w, c = image_array.shape
    frame_width = w // temp_window
    frames = [image_array[:, i * frame_width:(i + 1) * frame_width, :] for i in range(temp_window)]
    
    #padding
    padh = patch_unit - h % patch_unit if h % patch_unit != 0 else 0
    padw = patch_unit - frame_width % patch_unit if frame_width % patch_unit != 0 else 0

    #convert numpy arrays to tensors and apply padding
    frames = [torch.tensor(frame, dtype=torch.float32) for frame in frames]
    frames = [F.pad(frame.permute(2, 0, 1), (0, padw, 0, padh), mode='reflect') for frame in frames]
    
    #stack frames along a new dimension
    stacked_frames = torch.stack(frames, dim=0)

    return stacked_frames, h, frame_width, temp_window

def tensor_to_array(tensor, b, fidx):
    img_tensor = tensor[b, fidx, ...].data.unsqueeze(0).clamp_(0, 1)
    img = img_tensor.squeeze().cpu().numpy()
    if img.ndim == 3:
        img = np.transpose(img, (1, 2, 0))  # CHW to HWC
    return img

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
    _, c, l, h, w = input_blk.shape
    hpos, wpos = split_to_patches(h, w, patch_size)
    out_spaces = torch.zeros_like(input_blk)
    out_masks = torch.zeros_like(input_blk)
    for hi in hpos:
        for wi in wpos:
            input_ = input_blk[..., hi:hi+patch_size, wi:wi+patch_size]
            _, _, rectified = model_tilt(input_.permute(0,2,1,3,4))
            out_spaces[..., hi:hi+patch_size, wi:wi+patch_size].add_(rectified.permute(0,2,1,3,4))
            out_masks[..., hi:hi+patch_size, wi:wi+patch_size].add_(torch.ones_like(input_))
    return out_spaces / out_masks

def process_images(image_array, patch_size, temp_window, model_tilt, device):
    turb_imgs, h, frame_width, _ = fetch_images(image_array, temp_window, patch_size)
    turb_imgs = turb_imgs.unsqueeze(0).to(device).permute(0,2,1,3,4)

    with torch.no_grad():
        recovered = test_spatial_overlap(turb_imgs, model_tilt, patch_size)
        recovered = recovered[..., :h, :frame_width].permute(0,2,1,3,4)

        stacked_image = np.hstack([tensor_to_array(recovered, 0, i) for i in range(recovered.shape[1])])
        return stacked_image