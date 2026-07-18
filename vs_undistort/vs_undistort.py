# Script to run "Turbulence Mitigation Transformer" https://github.com/xg416/TMT

# Vapoursynth Implementation by pifroggi https://github.com/pifroggi/vs_undistort
# or tepete and pifroggi on Discord

import os
import re
import shutil
import logging
import subprocess
import vapoursynth as vs
from pathlib import Path
from .utils import get_window, get_tiles, expression

core = vs.core


def _pytorch(clip, temp_window=10, tiles=1, overlap=8, interpolation="bicubic", device="cuda"):
    import threading
    import numpy as np
    from collections import OrderedDict

    if device == "cpu":
        try:
            import torch
        except ImportError:
            raise RuntimeError("vs_undistort: The CPU/CUDA backends require PyTorch. Please install it from https://pytorch.org/ or choose a different backend. For the CUDA backend specifically, install a version of PyTorch with CUDA support.") from None

    if device == "cuda":
        try:
            import torch
        except ImportError:
            raise RuntimeError("vs_undistort: The CUDA backend requires PyTorch with CUDA. Please install a version of PyTorch with CUDA support from https://pytorch.org/ or choose a different backend.") from None
        if not torch.cuda.is_available():
            raise RuntimeError("vs_undistort: The CUDA backend requires PyTorch with CUDA, but the installed version has no CUDA support. Please upgrade to a version with CUDA support from https://pytorch.org/ or choose a different backend.")

    from .utils import inference_tiled
    from .models.UNet3d_TMT_arch import DetiltUNet3DS
    os.environ["CUDA_MODULE_LOADING"] = "LAZY"

    def _frames_to_tensor(frames, device, tile_device, fp16=False):
        temp_window = len(frames)
        h, w = frames[0].height, frames[0].width
        num_planes = frames[0].format.num_planes
        dtype = np.float16 if fp16 == True else np.float32
        arr = np.empty((temp_window, num_planes, h, w), dtype=dtype)
        for i, fr in enumerate(frames):
            for p in range(num_planes):
                arr[i, p, :, :] = np.asarray(fr[p])
        tensor = torch.from_numpy(arr)
        if tile_device.type != "cpu":
            return tensor.to(tile_device).clamp_(0, 1).unsqueeze(0)
        if device.type == "cuda":
            tensor = tensor.pin_memory()
        return tensor.clamp_(0, 1).unsqueeze(0)          
    
    def _tensor_to_frame(tensor, frame):
        tensor_np = tensor.detach().cpu().numpy()
        for p in range(frame.format.num_planes):
            frame_arr = np.asarray(frame[p])
            np.copyto(frame_arr, tensor_np[:, :, p])
    
    def _load_model(device, interpolation, fp16=False):
        model_tilt = DetiltUNet3DS(norm="LN", residual="pool", conv_type="dw", interpolation=interpolation)
        current_folder = os.path.dirname(os.path.abspath(__file__))
        path_tilt = os.path.join(current_folder, "models", "UNet3d_TMT.pth")
        ckpt_tilt = torch.load(path_tilt, map_location=device, weights_only=True)
        model_tilt.load_state_dict(ckpt_tilt["state_dict"] if "state_dict" in ckpt_tilt else ckpt_tilt)
        model_tilt.to(device)
        if fp16:
            model_tilt.half()
        model_tilt.eval()
        return model_tilt
    
    def _pytorch_inference(n, f):
        with torch.inference_mode():
            out = f[0].copy()
            tensor = _frames_to_tensor(f[1:], device, tile_device, fp16=fp16)          # (1, T, C, H, W)
            tensor = inference_tiled(tensor, model_tilt, tile_h, tile_w, overlap=overlap, scales=[True, True, True], tile_device=tile_device)
            stacked_tensor = tensor[0].permute(2, 0, 3, 1).contiguous().flatten(1, 2)  # (H, T * W, C), output one large frame to let vs handle the caching
            _tensor_to_frame(stacked_tensor, out)
            return out
    
    # checks
    if not isinstance(clip, vs.VideoNode):
        raise TypeError("vs_undistort: Clip must be a vapoursynth clip.")
    if clip.format.id == vs.PresetVideoFormat.NONE or clip.width == 0 or clip.height == 0:
        raise TypeError("vs_undistort: Clip must have constant format and dimensions.")
    if clip.format.color_family != vs.RGB:
        raise ValueError("vs_undistort: Clip must be in RGB format.")
    if not isinstance(temp_window, int) or isinstance(temp_window, bool):
        raise TypeError("vs_undistort: Temporal window length must be an integer.")
    if temp_window < 2:
        raise ValueError("vs_undistort: Temporal window length must be at least 2.")
    if not isinstance(tiles, int) or isinstance(tiles, bool):
        raise TypeError("vs_undistort: Tiles must be an integer.")
    if not isinstance(overlap, int) or isinstance(overlap, bool):
        raise TypeError("vs_undistort: Overlap must be an integer.")
    if overlap < 0:
        raise ValueError("vs_undistort: Overlap can not be negative.")
    if interpolation not in ["bilinear", "bicubic"]:
        raise ValueError("vs_undistort: Warp interpolation mode must be 'bilinear' or 'bicubic'.")
    
    orig_format = clip.format.id
    device      = torch.device(device)
    fp16        = device.type == "cuda" and torch.cuda.get_device_capability()[0] >= 7
    req_format  = vs.RGBH if fp16 else vs.RGBS
    model_tilt  = _load_model(device, interpolation, fp16=fp16)
    
    # decide tile size
    tile_w, tile_h = get_tiles(clip_w=clip.width, clip_h=clip.height, tiles=tiles, overlap=overlap)

    # pad if tile is larger than clip
    pad_w = max(0, tile_w - clip.width)
    pad_h = max(0, tile_h - clip.height)
    if pad_w or pad_h:
        clip = core.std.AddBorders(clip, right=pad_w, bottom=pad_h)
    
    width       = clip.width
    height      = clip.height
    tile_device = device if tiles == 1 else torch.device("cpu")

    # convert inputs to needed precision
    if clip.format.id != req_format:
        clip = core.resize.Point(clip, format=req_format)
    if tile_w < 128 or tile_h < 128:
        raise ValueError("vs_undistort: Tile size must be at least 128 x 128. Reduce tiles.")
    if tiles > 1 and overlap > min(tile_w, tile_h) // 2:
        raise ValueError("vs_undistort: Overlap can not be larger than half of tile size. Reduce overlap.")
    
    # inference
    offset_clips   = get_window(clip, temp_window)
    out_shape      = core.std.BlankClip(clip=offset_clips[0], width=width * temp_window, height=height, keep=True)
    stacked_clip   = core.std.ModifyFrame(out_shape, clips=[out_shape, *offset_clips], selector=_pytorch_inference)
    offset_clips   = [core.std.Crop(stacked_clip, left=i * width, right=(temp_window - 1 - i) * width) for i in range(temp_window)]
    unstacked_clip = core.std.Interleave(offset_clips)
    
    if unstacked_clip.format.id != orig_format:
        unstacked_clip = core.resize.Point(unstacked_clip, format=orig_format)
    if pad_w or pad_h:
        unstacked_clip = core.std.Crop(unstacked_clip, right=pad_w, bottom=pad_h)
    if unstacked_clip.num_frames != clip.num_frames:
        unstacked_clip = core.std.Trim(unstacked_clip, last=clip.num_frames - 1)
    
    return core.std.CopyFrameProps(unstacked_clip, clip)


def _get_builder(plugin_path, trt_version, cuda_major):
    # finds compatible tensorrt engine builders
    exe_name = "trtexec.exe" if os.name == "nt" else "trtexec"
    builders = []
    errors   = []
    
    # check for python tensorrt
    try:
        import tensorrt
        package_version = list(map(int, tensorrt.__version__.split(".")[:3]))
        if package_version == trt_version:
            builders.append(["python", tensorrt])
        else:
            errors.append(f"Python TensorRT: Wrong version {'.'.join(map(str, package_version))}")
    except ImportError:
        errors.append("Python TensorRT: Not found.")
    except Exception:
        errors.append("Python TensorRT: Found but failed to check version.")
    
    # check for bundled trtexec
    bundled_trtexec = Path(plugin_path) / "vsmlrt-cuda" / exe_name
    if bundled_trtexec.is_file() and os.access(str(bundled_trtexec), os.X_OK):
        builders.append(["trtexec", bundled_trtexec])
    else:
        errors.append(f"Bundled trtexec: Not found.")

    # check for system trtexec
    system_trtexec = shutil.which("trtexec")
    if system_trtexec is not None:
        try:
            trtexec_path = Path(system_trtexec)
            help_output  = subprocess.run([str(trtexec_path), "--help"], check=False, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True, encoding="locale", errors="replace")
            help_output  = f"{help_output.stdout}\n{help_output.stderr}"
            
            trtexec_version = None
            trtexec_version = re.search(r"\[TensorRT v(\d+)\]", help_output)
            if trtexec_version is None:
                raise RuntimeError("vs_undistort: Internal Error: Regex failed to find the version.")

            trtexec_version = int(trtexec_version.group(1))
            trtexec_version = [trtexec_version // 10000, (trtexec_version % 10000) // 100, trtexec_version % 100]
            if trtexec_version == trt_version:
                builders.append(["trtexec", trtexec_path])
            else:
                errors.append(f"System trtexec: Wrong version {'.'.join(map(str, trtexec_version))}")
        except Exception:
            errors.append("System trtexec: Found but failed to check version.")
    else:
        errors.append("System trtexec: Not found.")
    
    # return first compatible builder
    if builders:
        return builders[0]
    
    errors = "\n".join(f"{builder}" for builder in errors)
    raise FileNotFoundError(f"vs_undistort: No compatible TensorRT engine builder found. Please install the python package 'tensorrt' or install trtexec. The required TensorRT version is {'.'.join(map(str, trt_version))}. The required CUDA version is {cuda_major}.\n{errors}")


def _build_engine_trtexec(onnx_path, engine_path, temp_window, engine_w, engine_h, interpolation, trt_version, trtexec_path):
    # build engine using trtexec, supports trt 10 and 11

    # settings
    opt_shapes = f"input:1x{temp_window * 3}x{engine_h}x{engine_w}"
    io_formats = f"fp16:chw" if trt_version[0] < 11 else "chw"
    cmd = [
        str(trtexec_path),
        *(["--stronglyTyped"] if trt_version[0] < 11 else []),
        *(["--markDebug=grid_sampler,grid_sampler_1"] if interpolation == "bicubic" else []),  # part of gridsample bicubic workaround
        "--skipInference",
        "--memPoolSize=workspace:6144",
        "--builderOptimizationLevel=3",
        f"--inputIOFormats={io_formats}",
        f"--outputIOFormats={io_formats}",
        f"--onnx={onnx_path}",
        f"--saveEngine={engine_path}",
        f"--optShapes={opt_shapes}",
    ]

    # build
    try:
        result = subprocess.run(cmd, check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True, encoding="locale", errors="replace")
    except subprocess.CalledProcessError as e:
        msg = (
            "vs_undistort: Internal Error: trtexec failed while building the TensorRT engine.\n"
            f"  Command: {' '.join(cmd)}\n"
            f"  Return code: {e.returncode}\n"
        )
        if e.stdout:
            msg += f"\n=== trtexec stdout ===\n{e.stdout}"
        if e.stderr:
            msg += f"\n=== trtexec stderr ===\n{e.stderr}"
        raise RuntimeError(msg) from e


def _build_engine_python(onnx_path, engine_path, temp_window, engine_w, engine_h, interpolation, trt_package):
    # build engine using tensorrt python package, supports only trt 11 because of vapoursynth-mlrt-trt
    trt = trt_package

    # custom logger for errors
    class _TrtLogger(trt.ILogger):
        def __init__(self):
            trt.ILogger.__init__(self)
            self.messages = []
            self.fatal    = False
        def log(self, severity, msg):
            if severity <= trt.Logger.WARNING:
                self.messages.append((severity, msg))
                if self.fatal:
                    logging.critical(f"  [{severity}] {msg}")
                elif severity == trt.Logger.INTERNAL_ERROR:  # print fatal errors immediately because python may not get control back
                    self.fatal = True
                    log = "\n".join(f"  [{log_severity}] {log_msg}" for log_severity, log_msg in self.messages)
                    logging.critical(f"vs_undistort: Internal Error: TensorRT failed while building the TensorRT engine.\n=== TensorRT log ===\n{log}")
        def get_log(self):
            return "\n".join(f"  [{severity}] {msg}" for severity, msg in self.messages)

    # initialize trt and load model
    logger  = _TrtLogger()
    builder = trt.Builder(logger)
    network = builder.create_network()
    config  = builder.create_builder_config()
    parser  = trt.OnnxParser(network, logger)
    if not parser.parse_from_file(str(onnx_path)):
        errors = "\n".join(f"  {parser.get_error(i)}" for i in range(parser.num_errors))
        raise RuntimeError(f"vs_undistort: Internal Error: TensorRT failed while parsing the ONNX model.\n{errors}")
    
    # mark debug
    if interpolation == "bicubic":
        for layer_index in range(network.num_layers):
            layer = network.get_layer(layer_index)
            for output_index in range(layer.num_outputs):
                tensor = layer.get_output(output_index)
                if tensor is not None and tensor.name in ("grid_sampler", "grid_sampler_1"):  # part of gridsample bicubic workaround
                    network.mark_debug(tensor)                         
    
    # settings
    opt_shapes = (1, temp_window * 3, engine_h, engine_w)                                                             # optShapes
    network.get_input(0).allowed_formats = network.get_output(0).allowed_formats = 1 << int(trt.TensorFormat.LINEAR)  # IOFormats:chw
    config.set_memory_pool_limit(trt.MemoryPoolType.WORKSPACE, 6144 << 20)                                            # workspace
    config.builder_optimization_level = 3                                                                             # builderOptimizationLevel

    # build
    profile = builder.create_optimization_profile()
    profile.set_shape(network.get_input(0).name, opt_shapes, opt_shapes, opt_shapes)
    config.add_optimization_profile(profile)
    engine  = builder.build_serialized_network(network, config)
    if engine is None:
        log = logger.get_log()
        msg = "vs_undistort: Internal Error: TensorRT failed while building the TensorRT engine."
        if log:
            msg += f"\n=== TensorRT log ===\n{log}"
        raise RuntimeError(msg)
    
    # save engine
    with open(engine_path, "wb") as f:
        f.write(engine)


def _get_engine(onnx_path, engine_dir, temp_window, engine_w, engine_h, interpolation, force_rebuild=False) -> str:
    # get path to tensorrt engine
    os.makedirs(engine_dir, exist_ok=True)  # create engine folder if needed
    engine_name = f"undistort_{interpolation}_t{temp_window}_h{engine_h}_w{engine_w}_fp16.engine"
    engine_path = os.path.join(engine_dir, engine_name)
    
    # check plugin version
    try:
        info = core.trt.Version()
    except Exception as e:
        raise RuntimeError("vs_undistort: Please install a version of vs-mlrt with TensorRT support or choose a different backend.") from e
    
    # if engine file exist, return it
    if not force_rebuild and os.path.isfile(engine_path) and os.path.getsize(engine_path) >= 512:
        return engine_path
    
    # get plugin info
    plugin_path = os.path.dirname(info["path"].decode(errors="ignore"))
    trt_version = int(info["tensorrt_version"].decode(errors="ignore"))
    trt_version = [trt_version // 10000, (trt_version % 10000) // 100, trt_version % 100]
    cuda_major  = int(info["cuda_runtime_version"].decode(errors="ignore")) // 1000
    
    # build new engine
    logging.warning("vs_undistort: Building new TensorRT engine for interpolation='%s' with temp_window=%d, width=%d, and height=%d. This may take a few minutes.", interpolation, temp_window, engine_w, engine_h)
    builder_info = _get_builder(plugin_path=plugin_path, trt_version=trt_version, cuda_major=cuda_major)
    if builder_info[0] == "python":
        _build_engine_python(onnx_path=onnx_path, engine_path=engine_path, temp_window=temp_window, engine_w=engine_w, engine_h=engine_h, interpolation=interpolation, trt_package=builder_info[1])
    elif builder_info[0] == "trtexec":
        _build_engine_trtexec(onnx_path=onnx_path, engine_path=engine_path, temp_window=temp_window, engine_w=engine_w, engine_h=engine_h, interpolation=interpolation, trt_version=trt_version, trtexec_path=builder_info[1])
    else:
        raise RuntimeError(f"vs_undistort: Internal Error: Unknown TensorRT engine builder: {builder_info[0]}")
    logging.warning("vs_undistort: Engine building complete.")
    return engine_path


def _tensorrt_inference(input_clips, onnx_path, engine_dir, temp_window, tile_w, tile_h, overlap, tiles, interpolation, num_streams, flex_out_prop, force_rebuild=False):
    engine_path = _get_engine(onnx_path=onnx_path, engine_dir=engine_dir, temp_window=temp_window, engine_w=tile_w, engine_h=tile_h, interpolation=interpolation, force_rebuild=force_rebuild)
    model_args  = dict(engine_path=engine_path, num_streams=num_streams, flexible_output_prop=flex_out_prop, **(dict(tilesize=(tile_w, tile_h), overlap=(overlap, overlap)) if tiles > 1 else {}))

    # try inference, rebuild engine if it fails
    try:
        out = core.trt.Model(input_clips, **model_args)
    except vs.Error as e:
        err_msg = str(e).lower()
        serialization_keywords = ("serialize", "serialization", "deserialize", "deserialization")
        if any(k in err_msg for k in serialization_keywords) and not force_rebuild:
            logging.warning("vs_undistort: Engine loading failed. This may be due to a TensorRT or driver update. Rebuilding...")
            model_args["engine_path"] = _get_engine(onnx_path=onnx_path, engine_dir=engine_dir, temp_window=temp_window, engine_w=tile_w, engine_h=tile_h, interpolation=interpolation, force_rebuild=True)
            out = core.trt.Model(input_clips, **model_args)
        else:
            raise
    return out


def _tensorrt(clip, temp_window=10, tiles=1, overlap=8, interpolation="bicubic", num_streams=1, engine_folder=None):
    
    # checks
    if not isinstance(clip, vs.VideoNode):
        raise TypeError("vs_undistort: Clip must be a vapoursynth clip.")
    if clip.format.id == vs.PresetVideoFormat.NONE or clip.width == 0 or clip.height == 0:
        raise TypeError("vs_undistort: Clip must have constant format and dimensions.")
    if clip.format.id not in [vs.RGBH]:
        raise ValueError("vs_undistort: Clip must be in RGBH format for the TensorRT backend.")
    if not isinstance(temp_window, int) or isinstance(temp_window, bool):
        raise TypeError("vs_undistort: Temporal window length must be an integer.")
    if temp_window < 2:
        raise ValueError("vs_undistort: Temporal window length must be at least 2.")
    if not isinstance(tiles, int) or isinstance(tiles, bool):
        raise TypeError("vs_undistort: Tiles must be an integer.")
    if not isinstance(overlap, int) or isinstance(overlap, bool):
        raise TypeError("vs_undistort: Overlap must be an integer.")
    if overlap < 0:
        raise ValueError("vs_undistort: Overlap can not be negative.")
    if not isinstance(num_streams, int) or isinstance(num_streams, bool):
        raise TypeError("vs_undistort: Number of parallel TensorRT streams (num_streams) must be an integer.")
    if num_streams < 1:
        raise ValueError("vs_undistort: Number of parallel TensorRT streams (num_streams) must be at least 1.")
    
    # clamp
    clip = expression(clip, expr=["x 0 max 1 min"])
    
    # decide which dimensions to build the engine for
    tile_w, tile_h = get_tiles(clip_w=clip.width, clip_h=clip.height, tiles=tiles, overlap=overlap)

    # pad if tile is larger than clip
    pad_w = max(0, tile_w - clip.width)
    pad_h = max(0, tile_h - clip.height)
    if pad_w or pad_h:
        clip = core.std.AddBorders(clip, right=pad_w, bottom=pad_h)

    if tile_w < 128 or tile_h < 128:
        raise ValueError("vs_undistort: Tile size must be at least 128 x 128. Reduce tiles.")
    if tiles > 1 and overlap > min(tile_w, tile_h) // 2:
        raise ValueError("vs_undistort: Overlap can not be larger than half of tile size. Reduce overlap.")

    # make sure the extremely slow engine isn't build due to some tensorrt tactic limitations
    cur_pixels = temp_window * tile_w * tile_h
    max_pixels = 4194304  #2^22
    if cur_pixels >= max_pixels:
        raise ValueError(f"vs_undistort: temp_window * tile width * tile height must be smaller than {max_pixels} (currently {cur_pixels}). Increase tiles or reduce temp_window or overlap.")
    
    if interpolation == "bilinear":
        model_name = "UNet3d_TMT_lin_op19_fp16.onnx"
    elif interpolation == "bicubic":
        model_name = "UNet3d_TMT_cub_op19_fp16.onnx"
    else:
        raise ValueError("vs_undistort: Warp interpolation mode must be 'bilinear' or 'bicubic'.")
    
    current_dir   = os.path.dirname(os.path.abspath(__file__))
    engine_dir    = os.path.join(current_dir, "engines") if engine_folder is None else os.path.abspath(engine_folder)
    onnx_path     = os.path.join(current_dir, "models", model_name)
    flex_out_prop = "vs_undistort_output"
    force_rebuild = False

    # get inference window and do inference
    input_clips   = get_window(clip, temp_window)
    stacked_clips = _tensorrt_inference(
        input_clips=input_clips,
        onnx_path=onnx_path,
        engine_dir=engine_dir,
        temp_window=temp_window,
        tile_w=tile_w,
        tile_h=tile_h,
        overlap=overlap,
        tiles=tiles,
        interpolation=interpolation,
        num_streams=num_streams,
        flex_out_prop=flex_out_prop,
        force_rebuild=force_rebuild,
    )
    
    # turn stacked output into normal clip
    carrier_clip   = stacked_clips["clip"]
    num_planes     = stacked_clips["num_planes"]
    planes         = [carrier_clip.std.PropToClip(prop=f"{flex_out_prop}{i}")  for i in range(num_planes)]        # turn planes back into normal clips
    grouped_clips  = [core.std.ShufflePlanes(planes[i:i+3], [0, 0, 0], vs.RGB) for i in range(0, num_planes, 3)]  # group every 3 planes back into RGB clips
    unstacked_clip = core.std.Interleave(grouped_clips)                                                           # interleave to restore chronological order
    
    if pad_w or pad_h:
        unstacked_clip = core.std.Crop(unstacked_clip, right=pad_w, bottom=pad_h)
    if unstacked_clip.num_frames != clip.num_frames:
        unstacked_clip = core.std.Trim(unstacked_clip, last=clip.num_frames - 1)
    
    return core.std.CopyFrameProps(unstacked_clip, clip)


def vs_undistort(clip, temp_window=10, tiles=1, overlap=8, interpolation="bicubic", backend="tensorrt", num_streams=1, engine_folder=None):
    """Removes distortions. Also known as atmospheric turbulence mitigation, warp stabilization, film shrink or VHS distortion fix, heat haze removal.

    Args:
        clip: Distorted clip. Must be in RGB format.
        temp_window: Temporal window length. This is how many frames are grouped together and processed as a single chunk. Larger means
            higher VRAM requirements, but better temporal averaging and slower distortions can be removed. If this is too small,
            some distortions may not get removed, small jumps/hitches may be visible between windows and seams from tiling
            may become more obvious.
        tiles: Amount of tiles to split the frames into. Must be 1, 2, 4, 6, 8, 12, 16, 24, or 32. A higher amount reduces VRAM requirements,
            but also reduces spatial averaging and the size of distortions that can be removed.
        overlap: Overlap from one tile to the next. Use if seams between tiles are visible.
        interpolation: Interpolation mode used to warp the frames.  
            - `bilinear` = More blurry.
            - `bicubic` = No blur, but may oversharpen slightly.
        backend: The backend used to run the model.
            - `cpu` = CPU mode using PyTorch (very slow).
            - `cuda` = GPU mode using PyTorch with CUDA support. Requires any Nvidia GPU (fast).
            - `tensorrt` = GPU mode using vs-mlrt with TensorRT support. Requires an Nvidia RTX GPU (very fast and lower vram usage).
        num_streams: Number of parallel TensorRT streams. For high end GPUs higher can be a bit faster, but requires more VRAM. Only affects the TensorRT backend.
        engine_folder: Optional path to the TensorRT engine storage location. By default engines are stored in `vs_undistort/engines`. Only affects the TensorRT backend.
    """
    
    if backend in ["cpu", "cuda"]:
        return _pytorch(clip, temp_window=temp_window, tiles=tiles, overlap=overlap, interpolation=interpolation, device=backend)
    if backend in ["tensorrt", "trt"]:
        return _tensorrt(clip, temp_window=temp_window, tiles=tiles, overlap=overlap, interpolation=interpolation, num_streams=num_streams, engine_folder=engine_folder)
    raise ValueError("vs_undistort: Backend must be 'cpu', 'cuda', or 'tensorrt'.")
