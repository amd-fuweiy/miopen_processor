import argparse
import subprocess
import re
import csv


def parse_miopen_time(output: str):
    if not output:
        return None

    # 1) split by 'stats:' (case-insensitive), iterate from last to first
    parts = re.split(r'stats:\s*', output, flags=re.IGNORECASE)
    # skip the zeroth part which is text before the first 'stats:'
    for part in reversed(parts[1:]):
        # use only the first physical line after this 'stats:'
        line = part.splitlines()[0].strip()
        if not line:
            continue
        # split by commas and take last token
        toks = [t.strip() for t in line.split(',') if t.strip() != ""]
        if len(toks) >= 1:
            last = toks[-1]
            # extract a float from the last token
            m = re.search(r'([0-9]*\.[0-9]+|[0-9]+)', last)
            if m:
                try:
                    return float(m.group(1))
                except ValueError:
                    continue

    # 2) fallback: look for "Elapsed: <num> ms"
    m = re.search(r'Elapsed:\s*([0-9]*\.[0-9]+|[0-9]+)\s*ms', output, flags=re.IGNORECASE)
    if m:
        return float(m.group(1))

    # 3) fallback: generic timeMs pattern (older code)
    m = re.search(r'timeMs\s*.*?([0-9]*\.[0-9]+|[0-9]+)', output, flags=re.IGNORECASE)
    if m:
        return float(m.group(1))

    return None


def run_and_get_time(cmd: str):
    print(f"Executing: {cmd}")
    p = subprocess.Popen(
        cmd.split(),
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True
    )
    out, _ = p.communicate()

    time = parse_miopen_time(out)
    return time


def replace_F(cmd: str, F: int):
    if re.search(r"-F\s+\d+", cmd):
        return re.sub(r"-F\s+\d+", f"-F {F}", cmd)
    else:
        return cmd + f" -F {F}"


def process_cmd_list(file_path, output_csv="results.csv"):
    with open(file_path, "r") as f:
        command_list = [line.strip() for line in f if line.strip()]

    results = []

    for base_cmd in command_list:
        print(f"\n=== Processing Command ===\n{base_cmd}")

        cmd_f1 = replace_F(base_cmd, 1)
        cmd_f2 = replace_F(base_cmd, 2)
        cmd_f4 = replace_F(base_cmd, 4)

        t1 = run_and_get_time(cmd_f1)
        t2 = run_and_get_time(cmd_f2)
        t4 = run_and_get_time(cmd_f4)

        # backward = F2 + F4
        if t2 is not None and t4 is not None:
            tb = t2 + t4
        else:
            tb = None

        print(f"  Forward (F1): {t1} ms")
        print(f"  Backward (F2+F4): {tb} ms  (F2={t2}, F4={t4})")

        results.append({
            "command": base_cmd,
            "fwd_ms": t1,
            "bwd_ms": tb,
            "f2_ms": t2,
            "f4_ms": t4,
        })

    with open(output_csv, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["command", "fwd_ms", "bwd_ms", "f2_ms", "f4_ms"])
        writer.writeheader()
        writer.writerows(results)

    print(f"\n=== Done. Results written to {output_csv} ===")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--cmd_list", required=True, help="Path to command_list.txt")
    parser.add_argument("--out_csv", default="results.csv", help="Output CSV filename")
    args = parser.parse_args()

    process_cmd_list(args.cmd_list, args.out_csv)

