#!/usr/bin/env python3
"""Run every independent algorithm once per seed; export inspectable comparisons.

This is a teaching suite, not a benchmark leaderboard. Each algorithm is trained
once and its stored observations are reused for its named baseline comparison.
Outputs are immutable: select a new --out for a new run. --site only copies an
already verified snapshot into an explicitly named website checkout.
"""
import argparse
import csv
import hashlib
import json
import math
from pathlib import Path
import shutil
import sys
import zipfile

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from implementations import runtime


def run_gallery(out, seeds, steps):
    out = Path(out)
    out.mkdir(parents=True, exist_ok=False)
    groups = {}
    for algorithm, path in runtime.discover().items():
        meta = runtime.metadata(path)
        groups.setdefault(meta['task'], []).append(algorithm)
    records = []
    for task, ids in groups.items():
        # A task is compared with its own units, not every plot in one global rank.
        result = runtime.experiment(ids, seeds, steps, out/task)
        records.append({'task': task, 'status': result['status']})
    (out/'gallery-run.json').write_text(json.dumps({'seeds': seeds, 'steps': steps, 'tasks': records}, indent=2)+'\n')
    if any(r['status'] != 'complete' for r in records):
        raise RuntimeError('Failed runs retained; gallery export is not allowed')
    return out


def verified_gallery(out):
    """Preflight the complete current population and recompute plotted evidence."""
    out = Path(out)
    sources = runtime.source_hashes()
    paths = runtime.discover()
    metas = {key: runtime.metadata(path) for key,path in paths.items()}
    all_curves, manifests, diagnostics = {}, {}, {}
    expected_population = set(paths)
    seen = set()
    for path in sorted(out.glob('*/manifest.json')):
        m = json.loads(path.read_text())
        if m['status'] != 'complete' or m['source_sha256'] != sources:
            raise ValueError('Incomplete population or source drift: '+str(path))
        selected_ids = [meta['id'] for meta in m['algorithms']]
        if len(selected_ids) != len(set(selected_ids)) or seen.intersection(selected_ids):
            raise ValueError('Duplicate algorithms in gallery manifests')
        if any(key not in metas or meta != metas[key] for key,meta in zip(selected_ids,m['algorithms'])):
            raise ValueError('Manifest metadata differs from current implementation')
        runtime.comparable(m['algorithms'])
        seeds, steps = m['seeds'], m['steps']
        if not seeds or len(seeds)!=len(set(seeds)) or any(type(seed) is not int or seed<0 for seed in seeds) or type(steps) is not int or steps<1:
            raise ValueError('Invalid gallery seed population or clock')
        expected = {(key,seed) for key in selected_ids for seed in seeds}
        actual = [(record['algorithm'],record['seed']) for record in m['runs']]
        if len(actual)!=len(set(actual)) or set(actual)!=expected or any(r['status']!='complete' for r in m['runs']):
            raise ValueError('Missing, duplicate or failed planned algorithm/seed')
        files = [p for p in path.parent.rglob('*') if p.is_file() and p.name!='manifest.json']
        if any(p.is_symlink() for p in path.parent.rglob('*')):
            raise ValueError('Result symlinks are unsupported')
        actual_hashes = {p.relative_to(path.parent).as_posix(): hashlib.sha256(p.read_bytes()).hexdigest() for p in files}
        if m['artifacts'] != actual_hashes:
            raise ValueError('Result artifact list/content differs from manifest')
        records = []
        for record in m['runs']:
            directory = path.parent/record['algorithm']/('seed-'+str(record['seed']))
            rows = [json.loads(line) for line in (directory/'events.jsonl').read_text().splitlines() if line.strip()]
            runtime.validate_rows(rows,steps)
            with (directory/'metrics.csv').open(newline='',encoding='utf-8') as f:
                csv_rows = list(csv.DictReader(f))
            if len(csv_rows)!=len(rows) or any((int(c['step']),float(c['value']),c['phase']) != (r['step'],r['value'],r['phase']) for c,r in zip(csv_rows,rows)):
                raise ValueError('CSV measurements differ from raw event evidence')
            if record['final']!=rows[-1]['value'] or record['observations']!=len(rows):
                raise ValueError('Manifest measurements differ from raw event evidence')
            if metas[record['algorithm']]['task']=='retention_regression':
                for c,r in zip(csv_rows,rows):
                    old=r.get('old_task_mse')
                    if type(old) not in (int,float) or not math.isfinite(old) or float(c.get('old_task_mse','nan'))!=old:
                        raise ValueError('Retention diagnostics missing or CSV/raw evidence differs')
            records.append({**record,'rows':rows})
        curves = json.loads((path.parent/'curves.json').read_text())
        if curves != runtime.summarize(records):
            raise ValueError('Plot data differs from complete raw population')
        if m['algorithms'][0]['task']=='retention_regression':
            diagnostic_records=[{**record,'rows':[{**row,'value':row['old_task_mse']} for row in record['rows']]} for record in records]
            diagnostics.update(runtime.summarize(diagnostic_records))
        seen.update(selected_ids)
        for key,points in curves.items():
            all_curves[key] = points
            manifests[key] = (path.parent,m)
    if seen != expected_population:
        raise ValueError('Gallery algorithm population mismatch; missing='+repr(sorted(expected_population-seen)))
    for key,meta in metas.items():
        baseline = meta['baseline']
        if baseline not in metas:
            raise ValueError('Unknown baseline: '+key)
        runtime.comparable([meta,metas[baseline]])
        left,right = manifests[key][1],manifests[baseline][1]
        if left['seeds']!=right['seeds'] or left['steps']!=right['steps']:
            raise ValueError('Baseline seed population or registered horizon differs')
    return sources, metas, all_curves, manifests, diagnostics


def export_gallery(out, site):
    out, site = Path(out), Path(site)
    if not (site/'src/components/crl').is_dir():
        raise ValueError('--site must be the existing CRL website checkout')
    # Validate all jobs, raw records, metadata and baselines before touching the site.
    sources, metas, all_curves, manifests, diagnostics = verified_gallery(out)
    public = site/'public/crl-code'
    public.mkdir(parents=True, exist_ok=True)
    code_paths = sorted((ROOT/'implementations').rglob('*.py'))
    # Bundle all runtime dependencies, not an unusable lone algorithm file.
    bundle_paths = code_paths+[ROOT/name for name in ('LICENSE','LICENSE-CODE','LICENSE-DOCS.md')]+[ROOT/name for name in (
        'docs/learning-code.md', 'examples/deep_requirements.txt',
        'scripts/author_project.py', 'integrations/author_projects.json',
        'integrations/algorithm_coverage.json', 'docs/author-projects.md',
        'docs/implementation-coverage.md')]
    bundle_readme = '''# 强化学习独立教学实现

解压后在本目录运行。Python 3.10+；classic/continual标准库例子无需GPU。

```bash
python3 implementations/runtime.py --list
python3 implementations/classic/td_lambda.py --steps 1200 --seeds 0 1 2 3 4 --out results/MY_NEW_RUN
```

深度例子先在自己的隔离环境安装PyTorch：

```bash
python3 -m pip install -r examples/deep_requirements.txt
python3 implementations/deep/ppo.py --steps 1200 --seeds 0 1 2 3 4 --out results/PPO_NEW_RUN
```

[逐步阅读与运行](docs/learning-code.md) · [真实覆盖与缺口](docs/implementation-coverage.md) · [作者工程入口](docs/author-projects.md)

本包包含独立运行所需源码和共享依赖；测试与完整教材请看原教程checkout。
作者工程只提供固定源码获取工具与清单，未包含完整作者源码、checkpoint或训练成果。
教学曲线不构成一般算法效能或原论文重训证据；输出目录必须是新目录。
许可证索引见LICENSE；教学代码见LICENSE-CODE，文档见LICENSE-DOCS.md。
上游作者工程保留各自许可证。
'''
    for file in code_paths:
        dest = public/file.relative_to(ROOT)
        dest.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(file, dest)
    with zipfile.ZipFile(public/'learning-code.zip', 'w', zipfile.ZIP_DEFLATED) as archive:
        for file in bundle_paths:
            name = file.relative_to(ROOT).as_posix()
            info = zipfile.ZipInfo(name, (2026, 1, 1, 0, 0, 0))
            info.compress_type = zipfile.ZIP_DEFLATED
            archive.writestr(info, file.read_bytes())
        info=zipfile.ZipInfo('README.md',(2026,1,1,0,0,0))
        info.compress_type=zipfile.ZIP_DEFLATED
        archive.writestr(info,bundle_readme.encode('utf-8'))
    entries = []
    for key, path in runtime.discover().items():
        meta = metas[key]
        baseline = meta['baseline']
        if key not in all_curves or baseline not in all_curves:
            raise ValueError('Missing actual method/baseline results: '+key)
        runtime.comparable([meta, metas[baseline]])
        source_dir, manifest = manifests[key]
        target = public/'results'/key
        target.mkdir(parents=True, exist_ok=True)
        selected = {a: all_curves[a] for a in dict.fromkeys([key,baseline])}
        (target/'curves.svg').write_text(runtime.svg_plot(selected, [metas[a] for a in selected]), encoding='utf-8')
        (target/'curves.json').write_text(json.dumps(selected, ensure_ascii=False, indent=2)+'\n', encoding='utf-8')
        shutil.copyfile(source_dir/'manifest.json', target/'manifest.json')
        diagnostic_plots=[]
        if meta['task']=='retention_regression':
            old_curves={a:diagnostics[a] for a in selected}
            old_metas=[{**metas[a],'metric':'旧任务无噪声预测 MSE','unit':'squared_error','higher_better':False} for a in selected]
            (target/'old-task-mse.svg').write_text(runtime.svg_plot(old_curves,old_metas),encoding='utf-8')
            (target/'old-task-mse.json').write_text(json.dumps(old_curves,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
            diagnostic_plots.append({'key':'old-task-mse','label':'旧任务保留误差','metric':'旧任务无噪声预测 MSE','unit':'squared_error','higher_better':False,
                                     'svg':'old-task-mse.svg','json':'old-task-mse.json','curves':old_curves,
                                     'scope':'同完整训练seed人口，冻结解析旧任务MSE；不是额外独立样本。'})
        # The unchanged group manifest binds every group artifact; include all of
        # them so downloaded evidence can actually be checked against that manifest.
        with zipfile.ZipFile(target/'raw-runs.zip', 'w', zipfile.ZIP_DEFLATED) as archive:
            for name in sorted(manifest['artifacts']):
                archive.write(source_dir/name,name)
            archive.write(source_dir/'manifest.json', 'manifest.json')
        entries.append({**meta, 'file': path.relative_to(ROOT).as_posix(),
                        'source': path.read_text(encoding='utf-8'),
                        'source_sha256': hashlib.sha256(path.read_bytes()).hexdigest(),
                        'seeds': manifest['seeds'], 'steps': manifest['steps'],
                        'curves': selected, 'baseline_name': metas[baseline]['name'],
                        'diagnostic_plots':diagnostic_plots,
                        'run_command': f'python3 {path.relative_to(ROOT).as_posix()} --steps {manifest["steps"]} --seeds '+ ' '.join(map(str,manifest['seeds']))+' --out results/MY_NEW_RUN'})
    # Every supported method must be represented; stale rows cannot silently survive.
    catalogue = {'schema_version': 1, 'source_hashes': sources, 'entries': entries,
                 'scope': 'CPU teaching tasks. Mean ± one training-seed standard deviation. Not paper-scale reproduction.'}
    (site/'src/data/crl/code-gallery.json').write_text(json.dumps(catalogue, ensure_ascii=False, indent=2)+'\n', encoding='utf-8')
    (public/'catalog.json').write_text(json.dumps(catalogue, ensure_ascii=False, indent=2)+'\n', encoding='utf-8')
    for name in ('implementation-coverage.md', 'author-projects.md', 'learning-code.md'):
        file = ROOT/'docs'/name
        if file.exists():
            # Static hosts do not consistently send charset for text/markdown.
            # A UTF-8 signature also makes direct browser navigation unambiguous;
            # source documents and the downloadable code archive stay unchanged.
            (public/name).write_text(file.read_text(encoding='utf-8-sig'), encoding='utf-8-sig')
    author_manifest = ROOT/'integrations/author_projects.json'
    if author_manifest.exists():
        shutil.copyfile(author_manifest, public/'author-projects.json')
    coverage = ROOT/'integrations/algorithm_coverage.json'
    if coverage.exists():
        shutil.copyfile(coverage, public/'algorithm-coverage.json')
        shutil.copyfile(coverage, site/'src/data/crl/algorithm-coverage.json')
    print(f'Exported {len(entries)} algorithms with actual baseline curves to {site}')


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--out', type=Path, required=True)
    p.add_argument('--seeds', type=int, nargs='+', default=[0,1,2,3,4])
    p.add_argument('--steps', type=int, default=1200)
    p.add_argument('--export-only', action='store_true')
    p.add_argument('--site', type=Path)
    a = p.parse_args()
    if not a.export_only:
        run_gallery(a.out, a.seeds, a.steps)
    if a.site:
        export_gallery(a.out, a.site)


if __name__ == '__main__':
    main()
