import subprocess
import argparse
import re
import tempfile
import os

def extract_miopendriver_commands(log_file_path, op_type="conv"):
    """
    Extract MIOpenDriver commands from the log file
    """
    #pattern = re.compile(r"MIOpen\(HIP\): Command \[.*\]\s+(MIOpenDriver .*%s.*)" % op_type)
    pattern = re.compile(r"MIOpenDriver\s+conv\b.*")
    commands = []
    with open(log_file_path, "r") as f:
        for line in f:
            m = pattern.search(line)
            if m:
                commands.append(m.group(0).strip())
    #print(f"found commands: {commands}")
    return commands


def test_command(parse_script_path, cmd):
    """
    Call parse_miopen.py with a command and capture the MIOpen log
    """
    # Temporary file for log
    with tempfile.NamedTemporaryFile(delete=False) as tmpfile:
        log_file = tmpfile.name

    # Set environment variables
    env = os.environ.copy()
    env["MIOPEN_ENABLE_LOGGING_CMD"] = "1"
    env["MIOPEN_DEBUG_FORCE_IMMED_MODE_FALLBACK"] = "1"
    
    # Run parse_miopen.py and redirect stdout to log
    subprocess.run(
        ["python", parse_script_path, "--cmd", cmd],
        stdout=open(log_file, "w"),
        stderr=subprocess.STDOUT,
        env=env
    )

    # Extract conv commands from log
    # debug and print the exact output
    #with open(log_file, "r") as f:
    #    content = f.read()
    #    print(content)
    logged_cmds = extract_miopendriver_commands(log_file, op_type="conv")

    # Remove temporary log file
    os.remove(log_file)

    return logged_cmds


def compare_cmds(original_cmd, logged_cmds):
    """
    Return True if the original command is found in the log
    """
    original_cmd = original_cmd.strip()
    for idx, cmd in enumerate(logged_cmds):
        if original_cmd == cmd:
            return True, idx
    return False, -1


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--parse_script", type=str, required=True, help="Path to parse_miopen.py")
    parser.add_argument("--cmd_list", type=str, required=True,
                        help="Path to a text file with one MIOpenDriver command per line")
    args = parser.parse_args()

    # Read command list
    with open(args.cmd_list, "r") as f:
        cmd_list = [line.strip() for line in f if line.strip()]

    for i, cmd in enumerate(cmd_list):
        if "-F 4" in cmd or "-F 2" in cmd:
            continue
        print(f"\n=== Testing command #{i} ===")
        print("Original command:", cmd)

        logged_cmds = test_command(args.parse_script, cmd)
        print(f"Found {len(logged_cmds)} conv commands in log")

        match, idx = compare_cmds(cmd, logged_cmds)
        if match:
            print(f"✅ Match found at log entry #{idx}")
        else:
            print("❌ Original command NOT found in log")


if __name__ == "__main__":
    main()

