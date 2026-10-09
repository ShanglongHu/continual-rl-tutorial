"""Synchronize public site exports; verify byte hashes before generating repo files.

No training is started. Only paths declared in the public export manifest are read.
Existing local edits to managed output are rejected. Use --check in a read-only audit.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import posixpath
from pathlib import Path, PurePosixPath
import re
import subprocess
from urllib.parse import urljoin, urlsplit, unquote, quote

ROOT = Path(__file__).resolve().parents[1]
SITE = "https://yingwen.io/zh/continual-rl/"
PREFIX = "/zh/continual-rl/"
LEGACY = ["value", "control", "deep-value", "policy", "soft-control", "dyna", "options", "retention", "plasticity", "streaming", "state", "meta", "exploration"]
LEGACY_SLUGS = ["value", "control", "deep-value", "policy", "soft-control", "dyna", "skills", "retention", "plasticity", "streaming", "state", "meta", "exploration"]


def digest(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def foundation_groups(lessons: list[dict], track: str) -> list[tuple[str, list[dict]]]:
    """Use exported editorial roles, never infer roles or numbers from sort keys."""
    rows = sorted((lesson for lesson in lessons if lesson["track"] == track), key=lambda lesson: lesson["order"])
    for lesson in rows:
        if lesson.get("kind") not in {"core", "branch"} or "displayOrdinal" not in lesson:
            raise ValueError("Stale lesson metadata; rebuild site export with kind/displayOrdinal: " + lesson["id"])
        ordinal = lesson["displayOrdinal"]
        if lesson["kind"] == "branch" and ordinal is not None:
            raise ValueError("Research branches must not have chapter numbers: " + lesson["id"])
        if lesson["kind"] == "core" and (type(ordinal) is not int or ordinal < 1):
            raise ValueError("Core chapter numbers must be positive integers: " + lesson["id"])
    core = [lesson for lesson in rows if lesson["kind"] == "core"]
    branches = [lesson for lesson in rows if lesson["kind"] == "branch"]
    if [lesson["displayOrdinal"] for lesson in core] != list(range(1, len(core) + 1)):
        raise ValueError("Core chapter numbers must be consecutive within track: " + track)
    return [("核心算法与训练系统" if track == "deep" else "建议的基础阅读顺序", core), ("并列研究分支", branches)]


def foundation_links(lessons: list[dict], prefix: str = "") -> str:
    def entry(lesson):
        label = (f"第 {lesson['displayOrdinal']} 章 · " if lesson["kind"] == "core" else "") + lesson["title"]
        return f"- [{label}]({prefix}{lesson['path'].rstrip('/').split('/')[-1]}.md)"
    return "\n".join(entry(lesson) for lesson in lessons)


def safe_target(value: str) -> str:
    path = PurePosixPath(value)
    if not path.parts or path.is_absolute() or ".." in path.parts or path.parts[0] not in {"examples", "docs", "data", "textbook", "tutorials", "foundations", "algorithms", "assets"}:
        raise ValueError("Unsafe generated target: " + value)
    if path.parts[0] == "assets" and not re.fullmatch(r"assets/crl-figures/[A-Za-z0-9_-]+(?:\.mobile)?\.svg", value):
        raise ValueError("Unsafe figure target: " + value)
    if not (ROOT/path).resolve().is_relative_to(ROOT.resolve()):
        raise ValueError("Generated target escapes repository: " + value)
    return path.as_posix()


def rewrite_markdown(text: str, source: str, target: str, links: dict[str, str], programs: set[str], anchors: dict[str, set[str]] | None = None, published_tutorials: set[str] | None = None, published_figures: dict[str, str] | None = None) -> str:
    # Manifest-declared sources may not exist until the first sync completes.
    # Confinement still applies if an existing ancestor is an external symlink.
    published_tutorials = published_tutorials or set()
    published_figures = published_figures or {}
    def rewrite(match):
        href = match.group(1)
        absolute = urljoin(SITE + "download/" + source, href)
        original = urlsplit(href)
        # Chapter prose can carry links relative to its HTML page rather than
        # the download endpoint (including inside combined chapter exports).
        # Recover only a page explicitly declared by the export manifest.
        if not original.scheme and not original.netloc and original.path.startswith("../"):
            page = re.sub(r"^(?:\.\./)+", "", unquote(original.path))
            if page in links and not page.startswith("download/"):
                absolute = urljoin(SITE, page)
                absolute = urlsplit(absolute)._replace(query=original.query, fragment=original.fragment).geturl()
        parts = urlsplit(absolute)
        if parts.netloc != "yingwen.io" or parts.scheme not in {"http", "https"}:
            return match.group(0)
        if parts.path.startswith("/crl-figures/"):
            # Only manifest-declared original SVGs become offline repo assets.
            # Third-party, old unbundled figures and data keep online semantics.
            try:
                original_path = unquote(urlsplit(href).path, errors="strict")
                destination = published_figures.get(original_path)
                if destination and not parts.query and re.fullmatch(r"/crl-figures/[A-Za-z0-9_-]+(?:\.mobile)?\.svg", original_path):
                    destination = safe_target(destination)
                    relative = posixpath.relpath(destination, posixpath.dirname(target))
                    return "](" + relative + ("#" + parts.fragment if parts.fragment else "") + ")"
            except (OSError, RuntimeError, ValueError, UnicodeError):
                pass
            return "](" + parts._replace(scheme="https").geturl() + ")"
        if parts.path.startswith("/crl-code/"):
            # Only inspect published Python source roots. Download bundles,
            # results and query-bearing links keep their website semantics.
            online = parts._replace(scheme="https").geturl()
            if not parts.query:
                try:
                    original_path = unquote(urlsplit(href).path, errors="strict")
                    decoded = unquote(parts.path[len("/crl-code/"):], errors="strict")
                    asset = PurePosixPath(decoded)
                    safe = (not asset.is_absolute() and bool(asset.parts)
                            and asset.parts[0] in {"implementations", "tests", "tutorials"}
                            and asset.suffix == ".py"
                            and not any(piece in {".", ".."} for piece in original_path.split("/"))
                            and ".." not in asset.parts
                            and not any(char in decoded for char in ("\\", "\x00")))
                    if safe:
                        resolved = (ROOT / asset).resolve()
                        if resolved.is_relative_to(ROOT.resolve()) and (resolved.is_file() or asset.as_posix() in published_tutorials):
                            relative = posixpath.relpath(asset.as_posix(), posixpath.dirname(target))
                            return "](" + quote(relative, safe="/.-_~") + ("#" + parts.fragment if parts.fragment else "") + ")"
                except (OSError, RuntimeError, ValueError, UnicodeError):
                    # Invalid encodings and escaping/broken symlinks must never
                    # become repository-relative file references.
                    pass
            return "](" + online + ")"
        if not parts.path.startswith(PREFIX):
            return match.group(0)
        key = unquote(parts.path[len(PREFIX):])
        destination = links.get(key)
        # Browser aliases need not exist in Markdown. Keep such links online.
        fragment_exists = not parts.fragment or anchors is None or unquote(parts.fragment) in anchors.get(destination, set())
        if destination and not parts.query and fragment_exists:
            relative = posixpath.relpath(destination, posixpath.dirname(target))
            return "](" + relative + ("#" + parts.fragment if parts.fragment else "") + ")"
        return "](" + absolute + ")"

    text = re.sub(r"\]\(([^\s)]+)\)", rewrite, text)
    rewritten_tutorials = set()
    def command(match):
        script = match.group(2)
        name = script.removeprefix("public/crl-code/tutorials/")
        if script == name and name in programs:
            return match.group(1) + "examples/" + name
        # Website readers download these single files; repository readers run
        # from ROOT. Rewrite an existing or manifest-declared tutorial, never
        # an unknown filename or a symlink escaping the checkout.
        try:
            tutorial = (ROOT / "tutorials" / name).resolve()
            if tutorial.is_relative_to(ROOT.resolve()) and (tutorial.is_file() or 'tutorials/'+name in published_tutorials):
                rewritten_tutorials.add(name)
                return match.group(1) + "tutorials/" + name
        except (OSError, RuntimeError, ValueError):
            pass
        return match.group(0)
    text = re.sub(r"\b(python(?:3)?\s+)((?:public/crl-code/tutorials/)?[A-Za-z0-9_-]+\.py)\b", command, text)
    # A website reader downloads one file; a checkout reader runs from ROOT.
    # Adapt only the prose attached to a rewritten tutorial command or its
    # repository-relative link, not unrelated author-project instructions.
    paragraphs = text.split('\n\n')
    for index, paragraph in enumerate(paragraphs):
        next_paragraph = paragraphs[index + 1] if index + 1 < len(paragraphs) else ''
        for name in rewritten_tutorials:
            local_link = re.search(r'\]\((?:\.\./)*tutorials/' + re.escape(name) + r'\)', paragraph)
            has_command = 'tutorials/' + name in paragraph and re.search(r'\bpython3?\s', paragraph)
            next_command = next_paragraph.startswith('```') and 'tutorials/' + name in next_paragraph
            if local_link or has_command or next_command:
                paragraph = paragraph.replace('在文件所在目录', '在仓库根目录').replace('在保存文件的目录', '在仓库根目录')
                paragraph = paragraph.replace('保存到空目录后运行', '在仓库根目录运行')
                paragraph = paragraph.replace('将这个单文件保存为 ' + name, '仓库中的文件为 `tutorials/' + name + '`')
        paragraphs[index] = paragraph
    text = '\n\n'.join(paragraphs)
    # This parity check needs a website-only JSON, not a bundled repo asset.
    # Keep that source boundary explicit while making the checkout command run.
    comparison_block = (
        '```bash\npython3 tutorials/offline_support_walkthrough.py --test\n'
        'python3 tutorials/offline_support_walkthrough.py\n'
        '# 在网站仓库根目录核对公开图中的数据：\n'
        'python3 tutorials/offline_support_walkthrough.py \\\n'
        '  --compare public/crl-figures/offline-support-data.json\n```')
    if 'offline_support_walkthrough.py' in rewritten_tutorials and comparison_block in text:
        text = text.replace(
            '第一、二行在保存教程文件的目录运行；第三个命令在仓库根目录运行。',
            '以下命令在教程仓库根目录运行。')
        text = text.replace(comparison_block,
            '```bash\npython3 tutorials/offline_support_walkthrough.py --test\n'
            'python3 tutorials/offline_support_walkthrough.py\n```\n\n'
            '如需核对公开图中的数据，先从[网站公开图数据]'
            '(https://yingwen.io/crl-figures/offline-support-data.json)下载 JSON，'
            '保存为教程仓库根目录的 `offline-support-data.json`，再运行下面的命令。'
            '该 JSON 由网站提供，不随教程仓库分发。\n\n'
            '```bash\npython3 tutorials/offline_support_walkthrough.py '
            '--compare offline-support-data.json\n```')
    return text.replace('-r deep_requirements.txt','-r examples/deep_requirements.txt')


def expected_files(directory: Path) -> tuple[dict[str, bytes], dict]:
    manifest_bytes = (directory / "repository-export.json").read_bytes()
    manifest = json.loads(manifest_bytes)
    if manifest.get("schema_version") != 1:
        raise ValueError("Unsupported site export schema")
    entries = manifest["files"]
    if len({e["file"] for e in entries}) != len(entries) or len({e["target"] for e in entries}) != len(entries):
        raise ValueError("Duplicate source or target in export")
    links = {"download/" + e["file"]: safe_target(e["target"]) for e in entries}
    links.update({e["page"]: e["target"] for e in entries if e.get("page")})
    links.update({"": "README.md", "foundations/": "foundations/README.md", "algorithms/": "textbook/README.md", "labs/": "docs/experiments.md", "research/": "docs/research-atlas.md", "frontier/": "docs/research-atlas.md", "experiments/handbook/": "docs/experiment-handbook.md", "start/": "docs/learning-route.md"})
    for track in ("tabular", "approximation", "deep"):
        links["foundations/" + track + "/"] = "foundations/" + track + "/README.md"
    links["overview/"] = "docs/field-framework.md"
    for stage in ("classic-rl", "deep-rl", "continual-rl"):
        links["start/" + stage + "/"] = "docs/learning-route-" + stage + ".md"
    for entry in entries:
        if entry["file"].startswith("intro-"):
            links["start/" + entry["file"][6:-3] + "/"] = entry["target"]
    programs = {e["file"] for e in entries if e["target"] == 'examples/'+e["file"] and e["file"].endswith(".py")}
    published_tutorials = {safe_target(e["target"]) for e in entries if re.fullmatch(r'tutorials/[A-Za-z0-9_-]+\.py',e["target"])}
    published_figures = {}
    for entry in entries:
        if entry['file'].startswith('figure--'):
            name = entry['file'][len('figure--'):]
            if not re.fullmatch(r'[A-Za-z0-9_-]+(?:\.mobile)?\.svg', name) or entry['target'] != 'assets/crl-figures/'+name:
                raise ValueError('Invalid figure declaration: '+entry['file'])
            published_figures['/crl-figures/'+name] = safe_target(entry['target'])
    anchors = {}
    for entry in entries:
        if entry['file'].endswith('.md') and re.fullmatch(r'[A-Za-z0-9_.-]+', entry['file']):
            body = (directory / entry['file']).read_text(encoding='utf-8')
            anchors[entry['target']] = set(re.findall(r'<a id="([^"]+)"', body))
    output: dict[str, bytes] = {}
    source_hashes = {}
    for entry in entries:
        filename = entry["file"]
        if Path(filename).name != filename or not re.fullmatch(r"[A-Za-z0-9_.-]+", filename):
            raise ValueError("Invalid export filename")
        raw = (directory / filename).read_bytes()
        if digest(raw) != entry["sha256"]:
            raise ValueError("Source checksum mismatch: " + filename)
        target = safe_target(entry["target"])
        content = rewrite_markdown(raw.decode("utf-8"), filename, target, links, programs, anchors, published_tutorials, published_figures).encode("utf-8") if filename.endswith(".md") else raw
        output[target] = content
        source_hashes[target] = entry["sha256"]

    # Preserve every old algorithm URL while updating its full tutorial content.
    for index, (chapter, slug) in enumerate(zip(LEGACY, LEGACY_SLUGS), 1):
        target = f"algorithms/{index:02d}-{slug}.md"
        raw = (directory / f"chapter-{chapter}.md").read_text(encoding="utf-8")
        output[target] = rewrite_markdown(raw, f"chapter-{chapter}.md", target, links, programs, anchors, published_tutorials, published_figures).encode()

    def put(target, body):
        output[target] = (body.strip() + "\n").encode("utf-8")
    chapters, lessons = manifest["chapters"], manifest["lessons"]
    put("textbook/README.md", "# CRL 教材与算法\n\n[基础分册](../foundations/README.md) · [实验手册](../docs/experiment-handbook.md) · [研究问题](../docs/research-atlas.md)\n\n" + "\n".join(f"- [{c['title']}]({c['id']}.md)" for c in chapters))
    foundation_parts=[]
    for track, title in [("tabular", "表格强化学习 · Sutton Part I"), ("approximation", "函数逼近与经典进阶 · Sutton Part II"), ("deep", "现代深度强化学习 · 核心算法与并列研究分支")]:
        groups=foundation_groups(lessons, track)
        rows=[lesson for _, group in groups for lesson in group]
        lines=foundation_links(rows)
        if track == "deep":
            sections=[]
            for label, group in groups:
                sections.append("## "+label+"\n\n"+foundation_links(group))
            lines="先掌握核心更新，再按信息、数据和目标条件选择分支。分支可以交叉组合，不是必须依次完成的关卡。\n\n"+"\n\n".join(sections)
        put(f"foundations/{track}/README.md", f"# {title}\n\n{lines}\n\n[全部基础分册](../README.md) · [进入 CRL](../../textbook/README.md)")
        foundation_parts.append(f"## [{title}]({track}/README.md)\n\n" + foundation_links(rows, track+"/"))
    put("foundations/README.md", "# 强化学习基础：表格方法、函数逼近与深度学习\n\n每章给出设定、推导、执行顺序、手算、代码、边界与练习。Part II 不是可跳过的附录：共享参数、半梯度、离策略稳定性、资格迹和策略梯度是后续方法的共同基础。\n\n" + "\n\n".join(foundation_parts) + "\n\n[CRL 教材](../textbook/README.md)")
    put("algorithms/README.md", "# 算法阅读入口\n\n[完整 CRL 教材](../textbook/README.md) · [经典与深度 RL 基础](../foundations/README.md)\n\n下面保留原有十三个链接，其正文与当前教材同步：\n\n" + "\n".join(f"- [{next(c['title'] for c in chapters if c['id']==chapter)}]({i:02d}-{slug}.md)" for i,(chapter,slug) in enumerate(zip(LEGACY,LEGACY_SLUGS),1)))
    all_rows=[]
    for c in lessons+chapters:
        doc=c["path"].rstrip('/')+'.md' if c in lessons else 'textbook/'+c['id']+'.md'
        command=rewrite_markdown(c['command'], '', '', {}, programs)
        all_rows.append(f"## [{c['title']}](../{doc})\n\n{c['scope']}\n\n```bash\n{command}\n```\n\n[源码](../examples/{c['file']})")
    put("docs/experiments.md", "# 实验与实现\n\n在仓库根执行。标准库机制脚本无需额外安装。可选神经训练需要 PyTorch，安装与测试见对应章节。单元测试、短程训练和原论文效能复现是不同验收层。\n\n```bash\npython3 scripts/test_examples.py\npython3 scripts/run_all.py --quick --out results/first-run\n```\n\n[实验设计与测试手册](experiment-handbook.md) · [研究协议](protocol.md) · [研究 Workbench](https://github.com/ying-wen/rl-research-workbench)\n\n"+"\n\n".join(all_rows))
    output["data/site-export.json"] = manifest_bytes
    receipt={"schema_version":1,"source":SITE,"source_export_sha256":digest(manifest_bytes),"files":{name:{"sha256":digest(data),"source_sha256":source_hashes.get(name)} for name,data in sorted(output.items())}}
    return output, receipt


def sync(directory: Path, check: bool = False) -> list[str]:
    output, receipt = expected_files(directory)
    output["data/site-sync.json"]=(json.dumps(receipt,ensure_ascii=False,indent=2)+"\n").encode()
    differences=[name for name,data in output.items() if not (ROOT/name).is_file() or (ROOT/name).read_bytes()!=data]
    if check:
        return differences
    # Protect edits made after the last generated version, and all unstaged tracked edits.
    old_receipt=ROOT/"data/site-sync.json"
    previous = {}
    if old_receipt.is_file():
        previous=json.loads(old_receipt.read_text())["files"]
        for name in differences:
            if name in previous and (ROOT/name).is_file() and digest((ROOT/name).read_bytes())!=previous[name]["sha256"]:
                raise RuntimeError("Modified generated file; preserve or reconcile first: "+name)
    changed=subprocess.check_output(["git","diff","HEAD","--name-only"],cwd=ROOT,text=True).splitlines()
    changed+=subprocess.check_output(["git","ls-files","--others","--exclude-standard"],cwd=ROOT,text=True).splitlines()
    conflicts=(set(changed)&set(differences))-set(previous)-{'data/site-sync.json'}
    if conflicts:
        raise RuntimeError("Uncommitted managed files: "+", ".join(sorted(conflicts)))
    for name in differences:
        path=ROOT/name
        path.parent.mkdir(parents=True,exist_ok=True)
        path.write_bytes(output[name])
    return differences


if __name__ == "__main__":
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-directory",type=Path,required=True,help="built site's public download directory")
    parser.add_argument("--check",action="store_true",help="compare without writing")
    args=parser.parse_args()
    changed=sync(args.source_directory,args.check)
    print(json.dumps({"mode":"check" if args.check else "write","differences":changed,"count":len(changed)},ensure_ascii=False,indent=2))
    raise SystemExit(1 if args.check and changed else 0)
