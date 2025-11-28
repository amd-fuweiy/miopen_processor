import argparse
import subprocess
import csv
import shlex
import os

def run_single_cmd(cmd, parser_script="parse_cmd_and_run_conv.py"):
    """
    Run the single parse_cmd_and_run_conv.py command
    and return forward/backward time in ms.
    """
    full_cmd = f'python {parser_script} --cmd "{cmd}"'
    proc = subprocess.Popen(
        full_cmd,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        shell=True,
        text=True,
    )
    out, _ = proc.communicate()

    forward = None
    backward = None

    for line in out.splitlines():
        line = line.strip()
        if line.startswith("Forward time:"):
            # Forward time:  0.6211 ms
            forward = float(line.split()[-2])
        elif line.startswith("Backward time:"):
            backward = float(line.split()[-2])

    return forward, backward, out


def batched_run(cmd_list_file, output_csv, parser_script):
    with open(cmd_list_file, "r") as f:
        cmds = [line.strip() for line in f if line.strip()]

    rows = []
    for i, cmd in enumerate(cmds):
        print(f"[{i+1}/{len(cmds)}] Running command:")
        print(cmd)

        fwd, bwd, raw_output = run_single_cmd(cmd, parser_script)

        print(f"fwd: {fwd}, bwd: {bwd}")

        if fwd is None or bwd is None:
            print("Warning: Failed to parse time from output:")
            print(raw_output)

        rows.append([cmd, fwd, bwd])

    # Write CSV
    with open(output_csv, "w", newline="false") as f:
        writer = csv.writer(f)
        writer.writerow(["command", "forward_ms", "backward_ms"])
        writer.writerows(rows)

    print(f"\n=== Done. Results written to: {output_csv} ===")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--parser_script", type=str, default="parse_cmd_and_run_conv.py")
    parser.add_argument("--cmd_list", type=str, required=True)
    parser.add_argument("--output_csv", type=str, default="batched_results.csv")
    args = parser.parse_args()

    batched_run(args.cmd_list, args.output_csv, args.parser_script)
