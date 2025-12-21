# Script to run "Turbulence Mitigation Transformer" https://github.com/xg416/TMT

# Vapoursynth Implementation by pifroggi https://github.com/pifroggi/vs_undistort
# or tepete and pifroggi on Discord

import os
import math
import shutil
import logging
import subprocess
import vapoursynth as vs

core = vs.core

def _get_window(clip, temp_window):
    # how many pad frames to reach a multiple of temp_window
    num_frames = clip.num_frames
    pad = (-num_frames) % temp_window

    if pad:
        # pad black frames if too short
        pad_clip = core.std.BlankClip(clip=clip, length=pad)
        clip     = core.std.Splice([clip, pad_clip])

    # offset_clips[i] contains frames i, i+temp_window, i+2*temp_window, ...
    return [core.std.SelectEvery(clip[i:], cycle=temp_window, offsets=[0]) for i in range(temp_window)]



# tensorrt code

def _get_trtexec() -> str:
    # first search for tensorrt plugins, then check for trtexec
    candidates = ("trt", "trt_rtx")
    plugins_path = None
    for name in candidates:
        try:
            mod = getattr(core, name)
        except AttributeError:
            continue
        try:
            path = mod.Version()["path"]
        except Exception:
            continue
        if isinstance(path, bytes):
            path = path.decode(errors="ignore")
        if path:
            plugins_path = os.path.dirname(path)
            break
    
    if plugins_path is not None:
        exe_name = "trtexec.exe" if os.name == "nt" else "trtexec"
        local_trtexec = os.path.join(plugins_path, "vsmlrt-cuda", exe_name)

        # try vsmlrt trtexec first, then check for system trtexec
        if os.path.isfile(local_trtexec) and os.access(local_trtexec, os.X_OK):
            return local_trtexec

    system_trtexec = shutil.which("trtexec")
    if system_trtexec is not None:
        return system_trtexec

    raise FileNotFoundError("vs_undistort.tensorrt: trtexec not found. Please install a version of vs-mlrt with TensorRT support. Make sure to follow the installation instructions.")


def _get_engine(onnx_path, engine_dir, temp_window=10, width=720, height=480, force_rebuild=False) -> str:
    # build or reuse tensorrt engine
    os.makedirs(engine_dir, exist_ok=True)  # create engine folder if needed
    engine_name = f"undistort_T{temp_window}_H{height}_W{width}_fp16.engine"
    engine_path = os.path.join(engine_dir, engine_name)

    # if engine file exist, return it
    if not force_rebuild and os.path.isfile(engine_path) and os.path.getsize(engine_path) >= 1024:
        return engine_path

    # else build new engine
    logging.warning("vs_undistort.tensorrt: Building new TensorRT engine for temp_window=%d, tile_width=%d, tile_height=%d. This may take a few minutes.", temp_window, width, height)
    
    # engine in/out shape: (1, temp_window * 3, height, width)
    trtexec    = _get_trtexec()
    opt_shapes = f"input:1x{temp_window * 3}x{height}x{width}"
    opt_level  = 1
    cmd = [
        trtexec,
        "--fp16",
        "--inputIOFormats=fp16:chw",
        "--outputIOFormats=fp16:chw",
        "--skipInference",
        "--avgTiming=3",
        "--precisionConstraints=obey",  # some layers kept in fp32 due to low precisions issues on 30 series and possibly others
        "--layerPrecisions=node_layer_norm:fp32,node_Div_11:fp32,node_Erf_12:fp32,node_gelu:fp32,node_Mul_16:fp32,node_Add_14:fp32,node_permute_2:fp32,node_conv3d_1:fp32,node_conv3d_2:fp32,node_layer_norm_1:fp32,node_Div_18:fp32,node_Erf_19:fp32,node_gelu_1:fp32,node_add_73:fp32,node_Mul_23:fp32,node_Add_21:fp32,node_permute_4:fp32,node_conv3d_4:fp32,node_conv3d_5:fp32,node_permute_5:fp32,node_layer_norm_2:fp32,node_Div_31:fp32,node_Erf_32:fp32,node_Mul_36:fp32,node_Add_34:fp32,node_gelu_2:fp32,node_permute_6:fp32,node_conv3d_6:fp32,node_layer_norm_3:fp32,node_Div_38:fp32,node_Erf_39:fp32,node_gelu_3:fp32,node_add_148:fp32,node_Mul_43:fp32,node_Add_41:fp32,node_permute_8:fp32,node_conv3d_10:fp32,node_conv3d_11:fp32,node_permute_9:fp32,node_layer_norm_4:fp32,node_Div_50:fp32,node_Erf_51:fp32,node_Mul_55:fp32,node_Add_53:fp32,node_gelu_4:fp32,node_permute_10:fp32,node_conv3d_12:fp32,node_layer_norm_5:fp32,node_Div_57:fp32,node_Erf_58:fp32,node_gelu_5:fp32,node_add_223:fp32,node_Mul_62:fp32,node_Add_60:fp32,node_permute_12:fp32,node_layer_norm_6:fp32,node_Div_69:fp32,node_Erf_70:fp32,node_gelu_6:fp32,node_Mul_74:fp32,node_Add_72:fp32,node_permute_14:fp32,node_layer_norm_7:fp32,node_permute_17:fp32,node_Div_76:fp32,node_Erf_77:fp32,node_gelu_7:fp32,node_add_298:fp32,node_Mul_81:fp32,node_Add_79:fp32,node_permute_16:fp32,node_layer_norm_8:fp32,node_permute_18:fp32,node_layer_norm_9:fp32,node_Div_95:fp32,node_Erf_96:fp32,node_gelu_9:fp32,node_Mul_100:fp32,node_Add_98:fp32,node_permute_20:fp32,node_layer_norm_10:fp32,node_permute_23:fp32,node_Div_102:fp32,node_Erf_103:fp32,node_gelu_10:fp32,node_add_403:fp32,node_Mul_107:fp32,node_Add_105:fp32,node_permute_22:fp32,node_layer_norm_11:fp32,node_permute_24:fp32,node_layer_norm_12:fp32,node_Div_121:fp32,node_Erf_122:fp32,node_gelu_12:fp32,node_Mul_126:fp32,node_Add_124:fp32,node_permute_26:fp32,node_layer_norm_13:fp32,node_permute_29:fp32,node_Div_128:fp32,node_Erf_129:fp32,node_gelu_13:fp32,node_add_508:fp32,node_Mul_133:fp32,node_Add_131:fp32,node_permute_28:fp32,node_layer_norm_14:fp32,node_permute_30:fp32,node_layer_norm_15:fp32,node_Div_147:fp32,node_Erf_148:fp32,node_gelu_15:fp32,node_Mul_152:fp32,node_Add_150:fp32,node_permute_32:fp32,node_layer_norm_16:fp32,node_Div_154:fp32,node_Erf_155:fp32,node_gelu_16:fp32,node_add_613:fp32,node_Mul_159:fp32,node_Add_157:fp32,node_permute_34:fp32,node_upsample_trilinear3d:fp32,node_upsample_trilinear3d_1:fp32,node_permute_35:fp32,node_view_2:fp32,node_permute_36:fp32,node_view_7:fp32,node_permute_37:fp32,node_view_12:fp32,node_arange:fp32,node_view_3:fp32,node_expand_1:fp32,node_Unsqueeze_204:fp32,node_stack:fp32,node_unsqueeze:fp32,node_select_4:fp32,node_mul_1235:fp32,node_sub_555:fp32,node_div_4:fp32,node_scalar_tensor_default:fp32,node_Unsqueeze_312:fp32,node_stack_5:fp32,node_Unsqueeze_310:fp32,node_div_5:fp32,node_scalar_tensor_default_1:fp32,node_sub_568:fp32,node_mul_1249:fp32,node_select_5:fp32,node_add_860:fp32,node_select_2:fp32,node_mul_1128:fp32,node_sub_496:fp32,node_div_2:fp32,node_Unsqueeze_263:fp32,node_stack_3:fp32,node_Unsqueeze_261:fp32,node_div_3:fp32,node_sub_509:fp32,node_mul_1142:fp32,node_select_3:fp32,node_add_770:fp32,node_select:fp32,node_mul_1017:fp32,node_sub_433:fp32,node_div:fp32,node_Unsqueeze_213:fp32,node_stack_1:fp32,node_Unsqueeze_211:fp32,node_div_1:fp32,node_sub_447:fp32,node_mul_1031:fp32,node_select_1:fp32,node_add_675:fp32,node_Unsqueeze_206:fp32,node_expand:fp32,node_arange_1:fp32,node_view_4:fp32,node_grid_sampler:fp32,node_grid_sampler_2:fp32,node_grid_sampler_1:fp32,node_view_15:fp32,node_view_16:fp32",
        f"--onnx={onnx_path}",
        f"--saveEngine={engine_path}",
        f"--optShapes={opt_shapes}",
        f"--builderOptimizationLevel={opt_level}",
    ]

    try:
        result = subprocess.run(cmd, check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True, encoding="locale", errors="replace")
    except subprocess.CalledProcessError as e:
        msg = (
            "vs_undistort.tensorrt: trtexec failed while building the TensorRT engine.\n"
            f"  Command: {' '.join(cmd)}\n"
            f"  Return code: {e.returncode}\n"
        )
        if e.stdout:
            msg += f"\n=== trtexec stdout ===\n{e.stdout}"
        if e.stderr:
            msg += f"\n=== trtexec stderr ===\n{e.stderr}"
        raise RuntimeError(msg) from e

    logging.warning("vs_undistort.tensorrt: Engine building complete.")
    return engine_path


def tensorrt(clip, temp_window=10, tile_width=None, tile_height=None, overlap=None, num_streams=1):
    
    # checks
    if not isinstance(clip, vs.VideoNode):
        raise TypeError("vs_undistort.tensorrt: Clip must be a vapoursynth clip.")
    if clip.format.id not in [vs.RGBH]:
        raise ValueError("vs_undistort.tensorrt: Clip must be in RGBH format.")
    if temp_window < 2:
        raise ValueError("vs_undistort.tensorrt: Temporal window length must be at least 2.")
    if (tile_width and tile_width % 16) or (tile_height and tile_height % 16):
        closest_w = math.ceil(tile_width  / 16) * 16
        closest_h = math.ceil(tile_height / 16) * 16
        raise ValueError(f"vs_undistort.tensorrt: Tile size must be a multiple of 16. Closest upper values: {closest_w} x {closest_h}.")

    # pad to multiple of 16
    orig_w, orig_h = clip.width, clip.height
    pad_w = (16 - (orig_w % 16)) % 16
    pad_h = (16 - (orig_h % 16)) % 16
    if pad_w or pad_h:
        clip = core.std.AddBorders(clip, right=pad_w, bottom=pad_h)

    # decide which dimensions to build the engine for
    tile_w = clip.width  if tile_width  is None else int(tile_width)
    tile_h = clip.height if tile_height is None else int(tile_height)
    tile_w = min(tile_w, clip.width)
    tile_h = min(tile_h, clip.height)
    if tile_w < 128 or tile_h < 128:
        raise ValueError("Tile size must be at least 128 x 128.")

    # make sure the extremely slow engine isn't build
    cur_pixels = temp_window * tile_w * tile_h
    max_pixels = 4194304  #2^22
    if cur_pixels >= max_pixels:
        raise ValueError(f"vs_undistort.tensorrt: temp_window * tile_width * tile_height must be smaller than {max_pixels} (currently {cur_pixels}). Reduce temp_window, tile_width, or tile_height. If no tile size is set, the input dimensions are used.")

    # decide which backend to use
    backend = getattr(core, "trt", None)
    if backend is None:
        raise RuntimeError("vs_undistort.tensorrt: Please install a version of vs-mlrt with TensorRT support.")
    
    current_dir    = os.path.dirname(os.path.abspath(__file__))
    onnx_path      = os.path.join(current_dir, "dynamic_1st_stage_op19_fp32_dynamic.onnx")
    engine_dir     = os.path.join(current_dir, "engines")
    flex_out_prop  = "vs_undistort_output"
    force_rebuild  = False

    # get engine and inference window
    engine_path    = _get_engine(temp_window=temp_window, width=tile_w, height=tile_h, force_rebuild=force_rebuild, onnx_path=onnx_path, engine_dir=engine_dir)
    offset_clips   = _get_window(clip, temp_window)

    # try inference, rebuild engine if it fails
    try:
        stacked_clips  = backend.Model(offset_clips, engine_path=engine_path, num_streams=num_streams, tilesize=[tile_w, tile_h], overlap=overlap, flexible_output_prop=flex_out_prop)
    except vs.Error as e:
        err_msg = str(e).lower()
        serialization_keywords = ("serialize", "serialization", "deserialize", "deserialization")
        if any(k in err_msg for k in serialization_keywords) and not force_rebuild:
            logging.warning("vs_undistort.tensorrt: Engine loading failed. This may be due to a TensorRT or driver update. Rebuilding...")
            engine_path = _get_engine(temp_window=temp_window, width=tile_w, height=tile_h, force_rebuild=True, onnx_path=onnx_path, engine_dir=engine_dir)
            stacked_clips = backend.Model(offset_clips, engine_path=engine_path, num_streams=num_streams, tilesize=[tile_w, tile_h], overlap=overlap, flexible_output_prop=flex_out_prop)
        else:
            raise
    
    # turn output into normal clip
    carrier_clip   = stacked_clips["clip"]
    num_planes     = stacked_clips["num_planes"]
    planes         = [carrier_clip.std.PropToClip(prop=f"{flex_out_prop}{i}")  for i in range(num_planes)]       # turn planes back into normal clips
    grouped_clips  = [core.std.ShufflePlanes(planes[i:i+3], [0, 0, 0], vs.RGB) for i in range(0, num_planes, 3)] # group every 3 planes back into RGB clips
    unstacked_clip = core.std.Interleave(grouped_clips)                                                          # interleave to restore chronological order
    
    if pad_w or pad_h:
        unstacked_clip = core.std.Crop(unstacked_clip, right=pad_w, bottom=pad_h)
    if unstacked_clip.num_frames != clip.num_frames:
        unstacked_clip = core.std.Trim(unstacked_clip, last=clip.num_frames - 1)
    
    return core.std.CopyFrameProps(unstacked_clip, clip)



# pytorch code

def pytorch(clip, temp_window=10, tile_width=None, tile_height=None, overlap=None, scales=[True, True, True], interpolation="bilinear", device="cuda"):
    import torch
    import numpy as np
    from .TMT_dynamic_1st_stage import process_images
    from .UNet3d_TMT import DetiltUNet3DS
    
    def _frames_to_tensor(frames, device, fp16=False):
        temp_window = len(frames)
        h, w = frames[0].height, frames[0].width
        num_planes = frames[0].format.num_planes
        dtype = np.float16 if fp16 == True else np.float32
        arr = np.empty((temp_window, num_planes, h, w), dtype=dtype)
        for i, fr in enumerate(frames):
            for p in range(num_planes):
                arr[i, p, :, :] = np.asarray(fr[p])
        return torch.from_numpy(arr).to(device).clamp_(0, 1)
    
    def _tensor_to_frame(tensor, frame):
        tensor_np = tensor.detach().cpu().numpy()
        for p in range(frame.format.num_planes):
            frame_arr = np.asarray(frame[p])
            np.copyto(frame_arr, tensor_np[:, :, p])
    
    def _load_model(device, interpolation, fp16=False):
        model_tilt = DetiltUNet3DS(norm="LN", residual="pool", conv_type="dw", interpolation=interpolation)
        current_folder = os.path.dirname(os.path.abspath(__file__))
        path_tilt = os.path.join(current_folder, "dynamic_1st_stage.pth")
        ckpt_tilt = torch.load(path_tilt, map_location=device)
        model_tilt.load_state_dict(ckpt_tilt["state_dict"] if "state_dict" in ckpt_tilt else ckpt_tilt)
        model_tilt.to(device)
        if fp16:
            model_tilt.half()
        model_tilt.eval()
        return model_tilt
    
    def _process_window(n, f):
        fout = f[0].copy()
        window_frames  = f[1:]
        frames_tensor  = _frames_to_tensor(window_frames, device, fp16=fp16)
        stacked_tensor = process_images(frames_tensor, tile_h, tile_w, model_tilt, min_overlap=overlap, scales=scales)
        _tensor_to_frame(stacked_tensor, fout)
        return fout
    
    # checks
    if not isinstance(clip, vs.VideoNode):
        raise TypeError("vs_undistort.pytorch: Clip must be a vapoursynth clip.")
    if clip.format.color_family != vs.RGB:
        raise ValueError("vs_undistort.pytorch: Clip must be in RGB format.")
    
    # normalize inputs
    temp_window     = int(temp_window)
    if isinstance(overlap, float):
        overlap     = int(overlap)
    if isinstance(tile_width, float):
        tile_width  = int(tile_width)
    if isinstance(tile_height, float):
        tile_height = int(tile_height)
    
    # more checks
    if temp_window < 2:
        raise ValueError("vs_undistort.pytorch: Temporal window must be at least 2.")
    if overlap and overlap < 0:
        raise ValueError("vs_undistort.pytorch: Overlap can not be negative.")
    if interpolation not in ["bilinear", "bicubic"]:
        raise ValueError("vs_undistort.pytorch: Interpolation mode must be 'bilinear' or 'bicubic'.")
    if (tile_width and tile_width % 16) or (tile_height and tile_height % 16):
        closest_w = math.ceil(tile_width  / 16) * 16
        closest_h = math.ceil(tile_height / 16) * 16
        raise ValueError(f"vs_undistort.pytorch: Tile size must be a multiple of 16. Closest upper values: {closest_w} x {closest_h}.")

    # pad to multiple of 16
    orig_w, orig_h = clip.width, clip.height
    pad_w = (16 - (orig_w % 16)) % 16
    pad_h = (16 - (orig_h % 16)) % 16
    if pad_w or pad_h:
        clip = core.std.AddBorders(clip, right=pad_w, bottom=pad_h)

    width       = clip.width
    height      = clip.height
    orig_format = clip.format.id
    device      = torch.device(device)
    fp16        = device.type == "cuda" and torch.cuda.get_device_capability()[0] >= 7
    req_format  = vs.RGBH if fp16 else vs.RGBS
    model_tilt  = _load_model(device, interpolation, fp16=fp16)

    # convert inputs to needed precision
    if clip.format.id != req_format:
        clip = core.resize.Point(clip, format=req_format)

    # decide tile size
    tile_w = width  if tile_width  is None else int(tile_width)
    tile_h = height if tile_height is None else int(tile_height)
    tile_w = min(tile_w, width)
    tile_h = min(tile_h, height)
    if tile_w < 128 or tile_h < 128:
        raise ValueError("Tile size must be at least 128 x 128.")

    # inference
    offset_clips   = _get_window(clip, temp_window)
    out_shape      = core.std.BlankClip(clip=offset_clips[0], width=width * temp_window, height=height, keep=True)
    stacked_clip   = core.std.ModifyFrame(out_shape, clips=[out_shape, *offset_clips], selector=_process_window)
    offset_clips   = [core.std.Crop(stacked_clip, left=i * width, right=(temp_window - 1 - i) * width) for i in range(temp_window)]
    unstacked_clip = core.std.Interleave(offset_clips)
    
    if unstacked_clip.format.id != orig_format:
        unstacked_clip = core.resize.Point(unstacked_clip, format=orig_format)
    if pad_w or pad_h:
        unstacked_clip = core.std.Crop(unstacked_clip, right=pad_w, bottom=pad_h)
    if unstacked_clip.num_frames != clip.num_frames:
        unstacked_clip = core.std.Trim(unstacked_clip, last=clip.num_frames - 1)
    
    return core.std.CopyFrameProps(unstacked_clip, clip)
