#!/usr/bin/env python3
"""Run all teaching diagnostics into a NEW output directory. Standard library only."""
import argparse
import csv
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import platform
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]


def sha256(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def git_info():
    def get(*args):
        result = subprocess.run(["git", *args], cwd=ROOT, text=True,
                                capture_output=True, check=False)
        return result.stdout.strip() if result.returncode == 0 else None
    try:
        status = get("status", "--porcelain")
        return {"commit": get("rev-parse", "HEAD"),
                "dirty": bool(status) if status is not None else None}
    except FileNotFoundError:
        return {"commit": None, "dirty": None}


def experiment_commands(quick=False):
    seeds, episodes = (2, 80) if quick else (5, 800)
    steps = 400 if quick else 4000
    commands = []
    for name in ["prediction", "control", "policy", "dyna", "consolidation"]:
        args = [name, "--seeds", "1" if name == "consolidation" else str(seeds),
                "--episodes", str(episodes if name != "prediction" else (100 if quick else 1000))]
        if name in ["control", "policy", "dyna"]:
            args += ["--switch", str(episodes//2)]
        if name == "policy":
            args += ["--alpha", "0.05"]
        if name == "dyna":
            args += ["--planning", "10"]
        args += ["--out", name]
        commands.append((name, "rl_foundations.py", args))
    commands += [
        ("bandit", "crl_labs.py", ["bandit", "--seeds", "2" if quick else "20",
          "--steps", str(steps), "--alpha", "0.1", "--epsilon", "0.1", "--out", "bandit.csv"]),
        ("bandit-report", "analyze_crl.py", ["bandit.csv", "--out", "bandit.html"]),
        ("memory", "crl_labs.py", ["memory", "--seeds", "1", "--steps", str(steps),
          "--out", "memory.csv"]),
        ("credit", "crl_labs.py", ["credit", "--seeds", str(seeds), "--steps", str(steps),
          "--out", "credit.csv"]),
        ("retention", "crl_labs.py", ["retention", "--seeds", "1",
          "--steps", "100" if quick else "1000", "--out", "retention.csv"]),
    ]
    return commands


def run_suite(out, quick=False):
    out = Path(out).resolve()
    # Never clean or overwrite a reader's previous experiment directory.
    out.mkdir(parents=True, exist_ok=False)
    sources = sorted((ROOT/"examples").glob("*.py")) + [Path(__file__).resolve()]
    manifest = {
        "status": "running",
        "started_utc": datetime.now(timezone.utc).isoformat(),
        "mode": "smoke" if quick else "teaching",
        "python": platform.python_version(),
        "platform": {"system": platform.system(), "machine": platform.machine()},
        "git": git_info(),
        "source_sha256": {p.relative_to(ROOT).as_posix(): sha256(p) for p in sources},
        "reproduce": "python3 scripts/run_all.py " + ("--quick " if quick else "") +
                     "--out NEW_DIRECTORY",
        "command_cwd": "output directory; resolve script paths from the repository",
        "commands": [],
        "scope": "Teaching diagnostics, not a reproduction of deep-RL papers.",
    }
    manifest_path = out/"manifest.json"
    def save():
        manifest_path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2)+"\n",
                                 encoding="utf-8")
    save()
    try:
        for name, script, args in experiment_commands(quick):
            command = [sys.executable, str(ROOT/"examples"/script), *args]
            process = subprocess.run(command, cwd=out, text=True, capture_output=True,
                                     timeout=120, check=False)
            record = {"name": name, "argv": ["python3", "examples/"+script, *args],
                      "returncode": process.returncode}
            manifest["commands"].append(record)
            save()
            if process.returncode:
                raise RuntimeError(name+" failed:\n"+process.stderr)
            print("[ok] "+name)
        artifacts = {}
        for path in sorted(out.iterdir()):
            if path.suffix not in (".csv", ".html"):
                continue
            details = {"sha256": sha256(path), "bytes": path.stat().st_size}
            if path.suffix == ".csv":
                with path.open(encoding="utf-8", newline="") as handle:
                    details["data_rows"] = sum(1 for _ in csv.DictReader(handle))
            artifacts[path.name] = details
        manifest["artifacts"] = artifacts
        manifest["status"] = "complete"
        manifest["finished_utc"] = datetime.now(timezone.utc).isoformat()
        save()
    except BaseException as exc:
        manifest["status"] = "failed"
        manifest["error"] = str(exc)
        save()
        raise
    return manifest


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, required=True, help="A new output directory.")
    parser.add_argument("--quick", action="store_true",
                        help="Small smoke check, not the full teaching experiment.")
    args = parser.parse_args()
    if args.out.exists():
        parser.error("Output directory already exists; choose a new --out directory.")
    manifest = run_suite(args.out, args.quick)
    print(str(len(manifest["artifacts"]))+" outputs; open bandit.html and the five algorithm reports.")
    print("Configuration, source hashes and output hashes: "+str(args.out/"manifest.json"))


if __name__ == "__main__":
    main()
