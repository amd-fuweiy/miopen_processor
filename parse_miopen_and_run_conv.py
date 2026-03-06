import re
import argparse

import torch
import torch.nn as nn
from torch.profiler import profile, record_function, ProfilerActivity


def extract(cmd, flags, default=None, cast=int):
    """Extract value after flags in a cmd string. Only supports integer values."""
    if not isinstance(flags, list):
        flags = [flags]
    for f in flags:
        m = re.search(rf"{re.escape(f)}\s+(\d+)", cmd)
        if m:
            return cast(m.group(1))
    return default


def normalize_dtype_name(name: str) -> str:
    """Normalize user-friendly dtype name to canonical tokens."""
    s = (name or "").strip().lower()
    mapping = {
        "fp32": "fp32",
        "float32": "fp32",
        "f32": "fp32",
        "fp16": "fp16",
        "float16": "fp16",
        "f16": "fp16",
        "half": "fp16",
        "bf16": "bf16",
        "bfloat16": "bf16",
    }
    if s in mapping:
        return mapping[s]
    raise ValueError(f"Unknown dtype name: {name}. Supported: fp32, fp16, bf16")


def dtype_from_name(name: str) -> torch.dtype:
    name = normalize_dtype_name(name)
    if name == "fp32":
        return torch.float32
    if name == "fp16":
        return torch.float16
    if name == "bf16":
        return torch.bfloat16
    # unreachable
    raise ValueError(name)


def parse_miopen_subcmd(cmd: str) -> str:
    """
    Extract MIOpenDriver subcommand:
      MIOpenDriver conv... <flags>
    """
    m = re.search(r"\bMIOpenDriver(?:\.exe)?\s+(\S+)", cmd)
    return m.group(1) if m else ""


def parse_miopen_precision(cmd: str, force_dtype: str | None = None):
    """
    Map MIOpenDriver subcommand to torch dtype.

    Common MIOpenDriver subcommands seen in the wild:
      - conv          (fp32)
      - convfp32      (fp32)
      - convfp16      (fp16)
      - convhfp16     (fp16)   # some builds use "hfp16"
      - convbfp16     (bf16)

    If force_dtype is provided (fp32/fp16/bf16), it overrides subcommand mapping.

    Returns:
      (precision_name, torch_dtype, subcmd)
    """
    subcmd = parse_miopen_subcmd(cmd)
    subcmd_l = subcmd.lower()

    if force_dtype is not None:
        prec = normalize_dtype_name(force_dtype)
        return prec, dtype_from_name(prec), subcmd

    # Heuristic mapping by substring (more robust than exact equality)
    # Order matters: bfp16 must be checked before fp16
    if "bfp16" in subcmd_l or "bf16" in subcmd_l:
        return "bf16", torch.bfloat16, subcmd

    if "fp16" in subcmd_l or "hfp16" in subcmd_l or subcmd_l.endswith("half"):
        return "fp16", torch.float16, subcmd

    # Optional: fp32 explicit
    if "fp32" in subcmd_l:
        return "fp32", torch.float32, subcmd

    # default: plain "conv" is fp32 in your convention
    if subcmd_l.startswith("conv"):
        return "fp32", torch.float32, subcmd

    # fallback
    return "fp32", torch.float32, subcmd


def parse_miopen_conv(cmd: str, force_dtype: str | None = None):
    args = {}

    # precision / conv type
    prec_name, torch_dtype, subcmd = parse_miopen_precision(cmd, force_dtype=force_dtype)
    args["miopen_subcmd"] = subcmd          # e.g., "conv", "convfp16", "convbfp16"
    args["precision"] = prec_name           # "fp32" / "fp16" / "bf16"
    args["torch_dtype"] = torch_dtype       # torch.float32 / torch.float16 / torch.bfloat16

    # 2D or 3D
    spatial_dim = extract(cmd, "--spatial_dim", default=2)
    args["dim"] = spatial_dim

    # Common
    args["n"] = extract(cmd, "-n")
    args["c"] = extract(cmd, "-c")
    args["k"] = extract(cmd, "-k")
    args["groups"] = extract(cmd, "-g", default=1)

    if spatial_dim == 2:
        # Input
        args["H"] = extract(cmd, "-H")
        args["W"] = extract(cmd, "-W")
        # Kernel
        args["y"] = extract(cmd, "-y")
        args["x"] = extract(cmd, "-x")
        # Padding
        args["pad_h"] = extract(cmd, "-p", default=0)
        args["pad_w"] = extract(cmd, "-q", default=0)
        # Stride
        args["stride_h"] = extract(cmd, "-u", default=1)
        args["stride_w"] = extract(cmd, "-v", default=1)
        # Dilation
        args["dil_h"] = extract(cmd, "-l", default=1)
        args["dil_w"] = extract(cmd, "-j", default=1)
    else:
        # 3D Conv
        args["D"] = extract(cmd, "--in_d")
        args["H"] = extract(cmd, "-H")
        args["W"] = extract(cmd, "-W")
        args["fil_d"] = extract(cmd, "--fil_d")
        args["y"] = extract(cmd, "-y")
        args["x"] = extract(cmd, "-x")

        args["pad_d"] = extract(cmd, "--pad_d", default=0)
        args["pad_h"] = extract(cmd, "-p", default=0)
        args["pad_w"] = extract(cmd, "-q", default=0)

        args["stride_d"] = extract(cmd, "--conv_stride_d", default=1)
        args["stride_h"] = extract(cmd, "-u", default=1)
        args["stride_w"] = extract(cmd, "-v", default=1)

        args["dil_d"] = extract(cmd, "--dilation_d", default=1)
        args["dil_h"] = extract(cmd, "-l", default=1)
        args["dil_w"] = extract(cmd, "-j", default=1)

    return args


def validate_dtype_for_torch_conv(dtype: torch.dtype):
    """
    PyTorch eager Conv supports fp32/fp16/bf16 on CUDA (depending on hardware/backends).
    int8 etc. is not a drop-in replacement here; that would be quantized conv / QAT path.
    """
    if dtype in (torch.float32, torch.float16, torch.bfloat16):
        return
    raise ValueError(
        f"Unsupported dtype for this script: {dtype}. "
        "This script supports fp32/fp16/bf16 only. "
        "If you need int8, you'd need a quantized conv path (not nn.Conv eager)."
    )


def build_torch_conv(args, device):
    dtype = args.get("torch_dtype", torch.float32)
    validate_dtype_for_torch_conv(dtype)

    if args["dim"] == 2:
        conv = nn.Conv2d(
            in_channels=args["c"],
            out_channels=args["k"],
            kernel_size=(args["y"], args["x"]),
            stride=(args["stride_h"], args["stride_w"]),
            padding=(args["pad_h"], args["pad_w"]),
            dilation=(args["dil_h"], args["dil_w"]),
            groups=args["groups"],
            bias=False,
        ).to(device=device, dtype=dtype)

        input_shape = (args["n"], args["c"], args["H"], args["W"])
    else:
        conv = nn.Conv3d(
            in_channels=args["c"],
            out_channels=args["k"],
            kernel_size=(args["fil_d"], args["y"], args["x"]),
            stride=(args["stride_d"], args["stride_h"], args["stride_w"]),
            padding=(args["pad_d"], args["pad_h"], args["pad_w"]),
            dilation=(args["dil_d"], args["dil_h"], args["dil_w"]),
            groups=args["groups"],
            bias=False,
        ).to(device=device, dtype=dtype)

        input_shape = (args["n"], args["c"], args["D"], args["H"], args["W"])

    return conv, input_shape


def measure_forward_time(conv, input_shape, dtype, device="cuda", repeats=10):
    x = torch.randn(*input_shape, device=device, dtype=dtype)

    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)

    torch.cuda.synchronize()
    times = []

    warmup = 3
    for _ in range(warmup):
        _ = conv(x)

    for _ in range(repeats):
        start.record()
        _ = conv(x)
        end.record()
        torch.cuda.synchronize()
        times.append(start.elapsed_time(end))

    return sum(times) / len(times)


def measure_backward_time(conv, input_shape, dtype, device="cuda", repeats=10):
    x = torch.randn(*input_shape, device=device, dtype=dtype, requires_grad=True)

    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)

    torch.cuda.synchronize()
    times = []

    warmup = 3
    for _ in range(warmup):
        y = conv(x)
        loss = y.sum()
        conv.zero_grad(set_to_none=True)
        if x.grad is not None:
            x.grad.zero_()
        loss.backward()

    for _ in range(repeats):
        y = conv(x)
        loss = y.sum()

        conv.zero_grad(set_to_none=True)
        if x.grad is not None:
            x.grad.zero_()

        start.record()
        loss.backward()
        end.record()

        torch.cuda.synchronize()
        times.append(start.elapsed_time(end))

    return sum(times) / len(times)


def profile_conv(conv, input_shape, dtype, device="cuda"):
    x = torch.randn(*input_shape, device=device, dtype=dtype, requires_grad=True)

    with profile(
        activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA],
        record_shapes=True,
        with_stack=False,
        profile_memory=False,
        with_flops=True,
    ) as prof:
        with record_function("forward"):
            y = conv(x)

        with record_function("backward"):
            y.sum().backward()

    print(prof.key_averages().table(sort_by="cuda_time_total", row_limit=50))
    # prof.export_chrome_trace("conv_trace.json")
    return prof


def main():
    parser = argparse.ArgumentParser(description="Parse MIOpenDriver conv command to PyTorch code")
    parser.add_argument("--cmd", type=str, required=True, help="Full MIOpenDriver command line string")
    parser.add_argument(
        "--force_dtype",
        type=str,
        default=None,
        help="Override dtype mapping. One of: fp32, fp16, bf16",
    )
    parser.add_argument("--repeats", type=int, default=10, help="Timing repeats")
    parser.add_argument("--do_profile", action="store_true", help="Run torch profiler once")
    args_cli = parser.parse_args()

    cmd = args_cli.cmd
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    print("\n=== Parsing MIOpenDriver command ===")
    parsed = parse_miopen_conv(cmd, force_dtype=args_cli.force_dtype)
    print(parsed)

    conv, shape = build_torch_conv(parsed, device)
    dtype = parsed["torch_dtype"]

    print(f"device: {device}")
    print(f"miopen_subcmd: {parsed.get('miopen_subcmd')}, precision: {parsed.get('precision')}, torch_dtype: {dtype}")
    print(f"input_shape: {shape}")

    fwd_ms = measure_forward_time(conv, shape, dtype=dtype, device=device, repeats=args_cli.repeats)
    bwd_ms = measure_backward_time(conv, shape, dtype=dtype, device=device, repeats=args_cli.repeats)

    print(f"Forward time: {fwd_ms:.4f} ms")
    print(f"Backward time: {bwd_ms:.4f} ms")

    if args_cli.do_profile:
        profile_conv(conv, shape, dtype=dtype, device=device)


if __name__ == "__main__":
    main()
