#!/usr/bin/env python3
import argparse
import os
import shlex
import subprocess
import sys
from pathlib import Path
from typing import List, Tuple


MODE_FLAGS = {
    "--no_cot",
    "--candidates_only",
    "--no_candidates_only",
    "--cot_only",
    "--no_cot_only",
}


def _strip_leading_dashdash(args: List[str]) -> List[str]:
    if args and args[0] == "--":
        return args[1:]
    return args


def _pop_output_dir(args: List[str]) -> Tuple[List[str], str]:
    cleaned: List[str] = []
    found = ""
    i = 0
    while i < len(args):
        token = args[i]
        if token == "--output_dir":
            if i + 1 < len(args):
                found = args[i + 1]
                i += 2
                continue
            i += 1
            continue
        if token.startswith("--output_dir="):
            found = token.split("=", 1)[1]
            i += 1
            continue
        cleaned.append(token)
        i += 1
    return cleaned, found


def _strip_mode_flags(args: List[str]) -> List[str]:
    return [x for x in args if x not in MODE_FLAGS]


def _run_one(command: List[str], dry_run: bool) -> int:
    print(f"[GRPO-MULTI] {'DRY-RUN' if dry_run else 'RUN'}: {shlex.join(command)}")
    if dry_run:
        return 0
    proc = subprocess.run(command)
    return int(proc.returncode)


def main() -> int:
    parser = argparse.ArgumentParser(
        description=(
            "Run Experiment_RationaleCompare/04_run_grpo.py twice "
            "(C/R/J mode and J-only mode) with separate output directories."
        )
    )
    parser.add_argument(
        "--base_script",
        type=str,
        default=str(Path(__file__).with_name("04_run_grpo.py")),
        help="Path to the base GRPO script.",
    )
    parser.add_argument(
        "--python",
        type=str,
        default=sys.executable,
        help="Python executable used to launch child runs.",
    )
    parser.add_argument(
        "--output_root",
        type=str,
        default="",
        help=(
            "Root output directory for multitask runs. "
            "If empty, uses --output_dir from passthrough args; otherwise outputs/grpo_multitask."
        ),
    )
    parser.add_argument(
        "--cot_subdir",
        type=str,
        default="cot",
        help="Subdirectory name for C/R/J run.",
    )
    parser.add_argument(
        "--label_subdir",
        type=str,
        default="label",
        help="Subdirectory name for J-only run.",
    )
    parser.add_argument(
        "--skip_cot",
        action="store_true",
        help="Skip C/R/J run.",
    )
    parser.add_argument(
        "--skip_label",
        action="store_true",
        help="Skip J-only run.",
    )
    parser.add_argument(
        "--continue_on_error",
        action="store_true",
        help="Continue with the next mode even if one run fails.",
    )
    parser.add_argument(
        "--dry_run",
        action="store_true",
        help="Print commands without executing child processes.",
    )
    parser.add_argument(
        "passthrough",
        nargs=argparse.REMAINDER,
        help="Arguments passed to 04_run_grpo.py (prepend '--').",
    )
    args = parser.parse_args()

    run_cot = not args.skip_cot
    run_label = not args.skip_label
    if not run_cot and not run_label:
        raise ValueError("Both runs are disabled. Use at least one of C/R/J or label mode.")

    base_script = os.path.abspath(args.base_script)
    if not os.path.exists(base_script):
        raise FileNotFoundError(f"Base script not found: {base_script}")

    passthrough = _strip_leading_dashdash(list(args.passthrough))
    passthrough = _strip_mode_flags(passthrough)
    passthrough, passthrough_output_dir = _pop_output_dir(passthrough)

    output_root = str(args.output_root or "").strip()
    if not output_root:
        output_root = passthrough_output_dir.strip() if passthrough_output_dir else "outputs/grpo_multitask"
    output_root = os.path.abspath(output_root)

    jobs: List[Tuple[str, List[str]]] = []
    if run_cot:
        cot_output = os.path.join(output_root, args.cot_subdir)
        cot_cmd = [args.python, base_script] + passthrough + ["--output_dir", cot_output]
        jobs.append(("cot", cot_cmd))
    if run_label:
        label_output = os.path.join(output_root, args.label_subdir)
        label_cmd = [args.python, base_script] + passthrough + ["--output_dir", label_output, "--no_cot"]
        jobs.append(("label", label_cmd))

    final_rc = 0
    for mode, cmd in jobs:
        print(f"[GRPO-MULTI] mode={mode} output={cmd[cmd.index('--output_dir') + 1]}")
        rc = _run_one(cmd, dry_run=args.dry_run)
        if rc != 0:
            final_rc = rc
            print(f"[GRPO-MULTI] mode={mode} failed with exit_code={rc}")
            if not args.continue_on_error:
                return rc
        else:
            print(f"[GRPO-MULTI] mode={mode} completed")

    return final_rc


if __name__ == "__main__":
    raise SystemExit(main())

