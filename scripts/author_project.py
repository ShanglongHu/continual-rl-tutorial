"""固定作者工程源码；不安装依赖、不下载外部权重、不自动训练。"""
from __future__ import annotations

import argparse
import json
import re
import subprocess
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def projects():
    data = json.loads((ROOT / "integrations/author_projects.json").read_text(encoding="utf-8"))
    result = {}
    for item in data["projects"]:
        if not re.fullmatch(r"[a-z0-9_-]+", item["id"]) or not re.fullmatch(r"[0-9a-f]{40}", item["commit"]):
            raise ValueError("invalid project identifier or commit")
        if not re.fullmatch(r"https://github.com/[A-Za-z0-9_.-]+/[A-Za-z0-9_.-]+", item["repository"]):
            raise ValueError("only registered HTTPS GitHub repositories are supported")
        for path in item["reading_order"]:
            if Path(path).is_absolute() or ".." in Path(path).parts:
                raise ValueError("unsafe reading path")
        if item["id"] in result:
            raise ValueError("duplicate project identifier")
        result[item["id"]] = item
    return result


def git(directory, *args):
    return subprocess.run(["git", "-C", str(directory), *args], check=True,
                          text=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE).stdout.strip()


def verify(project, directory):
    """核对源码身份、上游remote、工作树清洁与真实阅读入口。"""
    directory = Path(directory).resolve()
    if not (directory / ".git").exists():
        raise ValueError("checkout missing .git")
    # Parent repositories must not accidentally pass as the requested checkout.
    if Path(git(directory, "rev-parse", "--show-toplevel")).resolve() != directory:
        raise ValueError("not the exact checkout root")
    sha = git(directory, "rev-parse", "HEAD")
    dirty = git(directory, "status", "--porcelain", "--untracked-files=all")
    remote = git(directory, "remote", "get-url", "origin").removesuffix(".git")
    missing = [path for path in project["reading_order"] if not (directory / path).is_file()]
    if sha != project["commit"] or dirty or remote != project["repository"] or missing:
        raise ValueError(f"checkout mismatch: sha={sha}, dirty={bool(dirty)}, remote={remote}, missing={missing}")
    return {"id": project["id"], "checkout": str(directory), "commit": sha, "clean": True,
            "reading_paths_verified": project["reading_order"], "claim": "pinned source checkout verified; dependencies and training not executed"}


def prepare(project, directory):
    """仅建立新目录，失败保留目录，不重试覆盖旧尝试。"""
    directory = Path(directory).resolve()
    directory.mkdir(parents=True, exist_ok=False)
    git(directory, "init")
    git(directory, "remote", "add", "origin", project["repository"] + ".git")
    # 固定提交浅抓取，blob按需获取；不拉外部dataset/checkpoint或子模块。
    git(directory, "fetch", "--depth", "1", "--filter=blob:none", "origin", project["commit"])
    git(directory, "checkout", "--detach", project["commit"])
    return verify(project, directory)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    sub.add_parser("list")
    inspect = sub.add_parser("inspect")
    inspect.add_argument("id")
    for command in ("prepare", "verify"):
        entry = sub.add_parser(command)
        entry.add_argument("id")
        entry.add_argument("--directory", type=Path, required=True)
    args = parser.parse_args(argv)
    try:
        registry = projects()
        if args.command == "list":
            result = [{"id": p["id"], "name": p["name"], "commit": p["commit"], "status": p["status"]} for p in registry.values()]
        elif args.id not in registry:
            raise ValueError("unknown author project: " + args.id)
        elif args.command == "inspect":
            result = registry[args.id]
        elif args.command == "prepare":
            result = prepare(registry[args.id], args.directory)
        else:
            result = verify(registry[args.id], args.directory)
        print(json.dumps(result, ensure_ascii=False, indent=2))
        return 0
    except (ValueError, OSError, subprocess.CalledProcessError) as exc:
        detail = exc.stderr if isinstance(exc, subprocess.CalledProcessError) else str(exc)
        parser.exit(1, detail + "\n")


if __name__ == "__main__":
    raise SystemExit(main())
