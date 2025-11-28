import re
import torch
import torch.nn as nn
import argparse
from torch.profiler import profile, record_function, ProfilerActivity


def extract(cmd, flags, default=None, cast=int):
    """Extract value after flags in a cmd string."""
    if not isinstance(flags, list):
        flags = [flags]
    for f in flags:
        m = re.search(rf"{f}\s+(\d+)", cmd)
        if m:
            return cast(m.group(1))
    return default


def parse_miopen_conv(cmd: str):
    args = {}

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


def build_torch_conv(args, device):
    if args["dim"] == 2:
        conv = nn.Conv2d(
            in_channels=args["c"],
            out_channels=args["k"],
            kernel_size=(args["y"], args["x"]),
            stride=(args["stride_h"], args["stride_w"]),
            padding=(args["pad_h"], args["pad_w"]),
            dilation=(args["dil_h"], args["dil_w"]),
            groups=args["groups"],
            bias=False
        ).to(device)
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
            bias=False
        ).to(device)
        input_shape = (args["n"], args["c"], args["D"], args["H"], args["W"])

    return conv, input_shape


def run_fake(conv, input_shape, device):
    x = torch.randn(*input_shape, device=device)
    y = conv(x)
    return y


def measure_forward_time(conv, input_shape, device="cuda", repeats=10):
    x = torch.randn(*input_shape, device=device)

    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)

    torch.cuda.synchronize()
    times = []

    warmup = 3
    for _ in range(warmup):
        y = conv(x)

    for _ in range(repeats):
        start.record()
        y = conv(x)
        end.record()
        torch.cuda.synchronize()
        times.append(start.elapsed_time(end))

    return sum(times) / len(times)


def measure_backward_time(conv, input_shape, device="cuda", repeats=10):
    x = torch.randn(*input_shape, device=device, requires_grad=True)

    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)

    torch.cuda.synchronize()
    times = []

    warmup = 3
    for _ in range(warmup):
        y = conv(x)
        loss = y.sum()
        conv.zero_grad()
        if x.grad is not None:
            x.grad.zero_()
        loss.backward()


    for _ in range(repeats):
        # forward (not timed)
        y = conv(x)
        loss = y.sum()

        conv.zero_grad()
        if x.grad is not None:
            x.grad.zero_()

        # measure backward only
        start.record()
        loss.backward()
        end.record()

        torch.cuda.synchronize()
        times.append(start.elapsed_time(end))

    return sum(times) / len(times)


def profile_conv(conv, input_shape, device="cuda"):
    x = torch.randn(*input_shape, device=device, requires_grad=True)

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

    args_cli = parser.parse_args()
    cmd = args_cli.cmd

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("\n=== Parsing MIOpenDriver command ===")
    parsed = parse_miopen_conv(cmd)
    print(parsed)

    #print("\n=== Creating PyTorch Conv Layer ===")
    conv, shape = build_torch_conv(parsed, device)
    #print(conv)
    #print("Input shape:", shape)

    #print("\n=== Running fake forward ===")
    #out = run_fake(conv, shape, device)
    #print("Output shape:", tuple(out.shape))
    print(f"device: {device}")
    fwd_ms = measure_forward_time(conv, shape, device)
    bwd_ms = measure_backward_time(conv, shape, device)
    print(f"Forward time:  {fwd_ms:.4f} ms")
    print(f"Backward time: {bwd_ms:.4f} ms")
    #profile_conv(conv, shape, device)




if __name__ == "__main__":
    main()

