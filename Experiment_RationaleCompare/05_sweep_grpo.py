#!/usr/bin/env python3
import argparse
import itertools
import json
import os
import re
import subprocess
import sys
import time
from typing import Any, Dict, List, Tuple


def _slug(text: str) -> str:
    value = re.sub(r"[^a-zA-Z0-9._-]+", "-", str(text or "").strip())
    value = value.strip("-")
    return value or "run"


def _cartesian_from_grid(grid: Dict[str, List[Any]]) -> List[Dict[str, Any]]:
    if not grid:
        return [{}]
    keys = list(grid.keys())
    values = []
    for k in keys:
        arr = grid.get(k, [])
        if not isinstance(arr, list) or len(arr) == 0:
            raise ValueError(f"grid key '{k}' must be a non-empty list")
        values.append(arr)
    runs = []
    for combo in itertools.product(*values):
        runs.append({k: v for k, v in zip(keys, combo)})
    return runs


def _as_flag(key: str) -> str:
    return "--" + key.replace("_", "-")


def _build_cli_args(params: Dict[str, Any]) -> List[str]:
    args: List[str] = []
    for key, value in params.items():
        if key == "name":
            continue
        if value is None:
            continue

        if key == "include_text":
            args.append("--include_text" if bool(value) else "--no-include-text")
            continue

        if isinstance(value, bool):
            if value:
                args.append(_as_flag(key))
            continue

        args.extend([_as_flag(key), str(value)])
    return args


def _parse_metric_line(line: str, prefix: str) -> Dict[str, float]:
    if prefix not in line:
        return {}
    pairs = re.findall(r"([a-zA-Z0-9_]+)=([0-9.]+)", line)
    out: Dict[str, float] = {}
    for k, v in pairs:
        try:
            out[k] = float(v)
        except Exception:
            continue
    return out


def _score_from_summary(run: Dict[str, Any], rank_by: str) -> float:
    eval_final = run.get("eval_final", {}) or {}
    test_final = run.get("test_final", {}) or {}
    if rank_by.startswith("eval_"):
        key = rank_by[len("eval_") :]
        return float(eval_final.get(key, float("-inf")))
    if rank_by.startswith("test_"):
        key = rank_by[len("test_") :]
        return float(test_final.get(key, float("-inf")))
    return float(run.get(rank_by, float("-inf")))


def _load_config(path: str) -> Dict[str, Any]:
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    if not isinstance(data, dict):
        raise ValueError("config must be a JSON object")
    return data


def _tail_lines(path: str, n: int = 80) -> List[str]:
    try:
        with open(path, "r", encoding="utf-8") as f:
            lines = f.readlines()
        return lines[-n:]
    except Exception:
        return []


def main() -> None:
    parser = argparse.ArgumentParser(description="Sweep runner for 04_run_grpo.py")
    parser.add_argument("--config", type=str, required=True, help="Path to sweep JSON config.")
    parser.add_argument("--output_root", type=str, default="outputs/grpo_sweeps")
    parser.add_argument("--grpo_script", type=str, default="Experiment_RationaleCompare/04_run_grpo.py")
    parser.add_argument("--python_bin", type=str, default=sys.executable or "python3")
    parser.add_argument("--nproc_per_node", type=int, default=1)
    parser.add_argument("--max_runs", type=int, default=None)
    parser.add_argument("--dry_run", action="store_true")
    parser.add_argument("--stop_on_error", action="store_true")
    args = parser.parse_args()

    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    config_path = args.config if os.path.isabs(args.config) else os.path.join(base_dir, args.config)
    output_root = args.output_root if os.path.isabs(args.output_root) else os.path.join(base_dir, args.output_root)
    grpo_script = args.grpo_script if os.path.isabs(args.grpo_script) else os.path.join(base_dir, args.grpo_script)
    os.makedirs(output_root, exist_ok=True)

    config = _load_config(config_path)
    sweep_name = _slug(config.get("name", "grpo_sweep"))
    rank_by = str(config.get("rank_by", "test_intent_acc")).strip()
    common = config.get("common", {})
    if not isinstance(common, dict):
        raise ValueError("config.common must be an object")

    if "runs" in config:
        raw_runs = config["runs"]
        if not isinstance(raw_runs, list):
            raise ValueError("config.runs must be a list")
        run_params = []
        for row in raw_runs:
            if not isinstance(row, dict):
                raise ValueError("each item in config.runs must be an object")
            run_params.append(row)
    else:
        grid = config.get("grid", {})
        if not isinstance(grid, dict):
            raise ValueError("config.grid must be an object")
        run_params = _cartesian_from_grid(grid)

    if args.max_runs is not None:
        run_params = run_params[: max(0, args.max_runs)]

    stamp = time.strftime("%Y%m%d_%H%M%S")
    sweep_dir = os.path.join(output_root, f"{sweep_name}_{stamp}")
    os.makedirs(sweep_dir, exist_ok=True)

    summary: List[Dict[str, Any]] = []
    print(f"[SWEEP] config={config_path}")
    print(f"[SWEEP] grpo_script={grpo_script}")
    print(f"[SWEEP] output_dir={sweep_dir}")
    print(f"[SWEEP] runs={len(run_params)} rank_by={rank_by}")

    for idx, run in enumerate(run_params):
        params = dict(common)
        params.update(run)

        run_name = _slug(params.get("name", f"run_{idx:03d}"))
        run_dir = os.path.join(sweep_dir, f"{idx:03d}_{run_name}")
        os.makedirs(run_dir, exist_ok=True)
        if not params.get("output_dir"):
            params["output_dir"] = run_dir

        cmd: List[str]
        if args.nproc_per_node > 1:
            cmd = [
                "torchrun",
                f"--nproc_per_node={args.nproc_per_node}",
                grpo_script,
            ]
        else:
            cmd = [args.python_bin, grpo_script]
        cmd.extend(_build_cli_args(params))

        cmd_text = " ".join(cmd)
        print(f"\n[SWEEP][{idx + 1}/{len(run_params)}] {run_name}")
        print(f"[SWEEP] cmd={cmd_text}")

        eval_final: Dict[str, float] = {}
        test_final: Dict[str, float] = {}
        status = "ok"
        return_code = 0
        elapsed = 0.0

        log_path = os.path.join(run_dir, "run.log")
        with open(log_path, "w", encoding="utf-8") as log_f:
            log_f.write(cmd_text + "\n\n")
            if args.dry_run:
                log_f.write("[DRY RUN]\n")
            else:
                t0 = time.time()
                proc = subprocess.Popen(
                    cmd,
                    stdout=subprocess.PIPE,
                    stderr=subprocess.STDOUT,
                    text=True,
                    bufsize=1,
                )
                assert proc.stdout is not None
                for line in proc.stdout:
                    log_f.write(line)
                    # Stream child process output so errors are visible immediately.
                    sys.stdout.write(line)
                    sys.stdout.flush()
                    if "[GRPO-EVAL-FINAL]" in line:
                        parsed = _parse_metric_line(line, "[GRPO-EVAL-FINAL]")
                        if parsed:
                            eval_final = parsed
                    if "[GRPO-TEST-FINAL]" in line:
                        parsed = _parse_metric_line(line, "[GRPO-TEST-FINAL]")
                        if parsed:
                            test_final = parsed
                proc.wait()
                elapsed = time.time() - t0
                return_code = int(proc.returncode or 0)
                if return_code != 0:
                    status = "failed"

        row = {
            "index": idx,
            "name": run_name,
            "status": status,
            "return_code": return_code,
            "elapsed_sec": round(elapsed, 3),
            "params": params,
            "eval_final": eval_final,
            "test_final": test_final,
            "log_path": log_path,
        }
        summary.append(row)

        if status == "ok":
            score = _score_from_summary(row, rank_by)
            print(f"[SWEEP] done elapsed={elapsed:.1f}s score({rank_by})={score}")
        else:
            print(f"[SWEEP] failed return_code={return_code} log={log_path}")
            tail = _tail_lines(log_path, n=80)
            if tail:
                print("[SWEEP] ---- tail(run.log) ----")
                for line in tail:
                    sys.stdout.write(line)
                if not tail[-1].endswith("\n"):
                    print("")
                print("[SWEEP] ---- end tail ----")
            if args.stop_on_error:
                break

    # Save summary files.
    summary_path = os.path.join(sweep_dir, "summary.json")
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump({"rank_by": rank_by, "runs": summary}, f, ensure_ascii=False, indent=2)

    jsonl_path = os.path.join(sweep_dir, "summary.jsonl")
    with open(jsonl_path, "w", encoding="utf-8") as f:
        for row in summary:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")

    ranked = [r for r in summary if r.get("status") == "ok"]
    ranked.sort(key=lambda r: _score_from_summary(r, rank_by), reverse=True)
    print("\n[SWEEP] ranking")
    for i, r in enumerate(ranked[:10], start=1):
        print(
            f"{i:02d}. {r['name']} score({rank_by})={_score_from_summary(r, rank_by)} "
            f"elapsed={r['elapsed_sec']}s"
        )
    print(f"[SWEEP] summary={summary_path}")
    print(f"[SWEEP] summary_jsonl={jsonl_path}")


if __name__ == "__main__":
    main()
