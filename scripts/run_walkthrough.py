"""按教材目录运行一个独立算例；不启动训练套件或下载依赖。"""
import argparse
import hashlib
import json
import os
from pathlib import Path
import re
import subprocess
import sys
import tempfile

ROOT = Path(__file__).resolve().parents[1]
NAME = re.compile(r"[A-Za-z0-9_-]+\.py\Z")


def catalog(root):
    """目录决定读者入口，导出清单决定来源；两份索引必须相符。"""
    manifest = json.loads((root / "data/site-export.json").read_text(encoding="utf-8"))
    groups = json.loads((root / "data/walkthroughs.json").read_text(encoding="utf-8"))["groups"]
    exports = {}
    for row in manifest["files"]:
        if not row["file"].startswith("walkthrough--"):
            continue
        name = row["file"][len("walkthrough--"):]
        if (not NAME.fullmatch(name) or row["target"] != "tutorials/" + name
                or not re.fullmatch(r"[0-9a-f]{64}", row["sha256"])):
            raise ValueError("不合法的算例导出条目：" + row["file"])
        if name in exports:
            raise ValueError("重复的算例导出：" + name)
        exports[name] = row
    entries = {}
    for group in groups:
        for chapter in group["chapters"]:
            for script in chapter["scripts"]:
                name = script["file"]
                if name not in exports or script["path"] != "/crl-code/tutorials/" + name:
                    raise ValueError("章节脚本与导出清单不符：" + name)
                entry = entries.setdefault(name, {"export": exports[name], "chapters": []})
                entry["chapters"].append((group["title"], chapter["title"], script["sectionTitle"]))
    if set(entries) != set(exports):
        raise ValueError("导出清单有未出现在教材目录中的算例")
    return entries


def source_bytes(root, name, entry):
    directory = (root / "tutorials").resolve(strict=True)
    source = (directory / name).resolve(strict=True)
    if source.parent != directory:
        raise ValueError("脚本路径超出 tutorials 目录：" + name)
    body = source.read_bytes()
    if hashlib.sha256(body).hexdigest() != entry["export"]["sha256"]:
        raise ValueError("脚本已不同于教材导出版本：" + name
                         + "。如在做改写练习，请直接用 python3 tutorials/" + name + " 运行。")
    return body


def run_one(root, name, entry, script_args, timeout):
    body = source_bytes(root, name, entry)
    # 空工作目录检验单文件约定；它不是不可信代码的安全沙箱。
    with tempfile.TemporaryDirectory(prefix="crl-walkthrough-") as temporary:
        script = Path(temporary) / name
        script.write_bytes(body)
        command = [sys.executable, "-I", "-B", str(script), *script_args]
        # Keep the script's stdout intact so its JSON can be redirected as data.
        print("运行 " + name + "（独立临时目录；输出在终端）", file=sys.stderr, flush=True)
        try:
            result = subprocess.run(command, cwd=temporary,
                                    env={**os.environ, "PYTHONDONTWRITEBYTECODE": "1",
                                         "PYTHONIOENCODING": "utf-8"},
                                    timeout=timeout, check=False)
        except subprocess.TimeoutExpired:
            print(f"超出 {timeout} 秒，已停止本次算例。", file=sys.stderr)
            return 124
        return result.returncode


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--list", action="store_true", help="按教材章节列出脚本，不运行")
    parser.add_argument("--timeout", type=int, default=30, help="单次运行秒数，1–300（默认30）")
    parser.add_argument("name", nargs="?", help="完整脚本文件名，见 --list")
    parser.add_argument("script_args", nargs=argparse.REMAINDER,
                        help="文件名之后的参数传给该脚本；用法见对应章节")
    args = parser.parse_args(argv)
    if not 1 <= args.timeout <= 300:
        parser.error("--timeout 必须在1–300之间")
    if args.list and args.name:
        parser.error("--list 不与脚本文件名一起使用")
    if not args.list and not args.name:
        parser.error("请指定文件名，或先用 --list 查看目录")
    try:
        entries = catalog(ROOT)
        if args.list:
            for name, entry in entries.items():
                print(name)
                for group, chapter, section in entry["chapters"]:
                    print("  " + group + " / " + chapter + " / " + section)
            return 0
        if args.name not in entries:
            raise ValueError("未登记的算例：" + args.name + "；请用 --list 查找完整文件名")
        return run_one(ROOT, args.name, entries[args.name], args.script_args, args.timeout)
    except (ValueError, KeyError, OSError) as error:
        print(str(error), file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
