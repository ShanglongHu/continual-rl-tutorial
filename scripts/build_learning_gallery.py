#!/usr/bin/env python3
"""Run every independent algorithm once per seed; export inspectable comparisons.

This is a teaching suite, not a benchmark leaderboard. Each algorithm is trained
once and its stored observations are reused for its named baseline comparison.
Outputs are immutable: select a new --out for a new run. --site only copies an
already verified snapshot into an explicitly named website checkout.
"""
import argparse
import ast
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

GUIDES = {
    'extended_classic': ('extended-classic', '估计、信用分配与平均奖励：公式到实验'),
    'extended_adaptation': ('extended-adaptation', '状态、梯度与持续适应：公式到实验'),
    'extended_knowledge': ('extended-knowledge', '探索、目标、选项与奖励：公式到实验'),
    'multiagent': ('extended-multiagent', '多智能体信用与价值分解：公式到实验'),
    'average_systems': ('average-systems', '平均奖励：预测、控制与规划的多状态实验'),
    'nonlinear_diagnostics': ('nonlinear-learning-depth', '共享参数与非线性学习：困难、推导与诊断'),
    'streaming_composition': ('streaming-composition', 'GVF、资格迹与技能：流式非线性实现'),
    'integrated_agents': ('integrated-agents', '持续智能体：从模块正确到接口正确'),
    'learner_control': ('learner-control', '持续控制：共同检查点后的学习与冻结'),
}

ADDITIONAL_DIAGNOSTICS = {
    'continuing_information_recovery_corridor': [
        ('recent_reward_rate','recent-rate','最近100步实际奖励率','reward_per_step',True,
         '短期变化与恢复；主图是完整生命期奖励率，两者不能互相替代。'),
        ('recovery_steps','recovery-steps','累计真实恢复步数','environment_steps',False,
         '失败后的实际时间成本；没有外部重置，也没有从图中删除恢复阶段。'),
        ('parameter_updates','parameter-updates','累计参数更新次数','updates',False,
         '冻结对照在共同400步检查点后停止参数写入，但保留观察记忆与环境状态。')],
    'context-bandit': [('old_return','old-return','旧上下文策略回报','return',True,
                       '评价同一冻结策略在旧上下文的期望奖励；主指标是当前上下文奖励。')],
    'switch-regression': [('old_mse','old-mse','旧目标函数预测误差','squared_error',False,
                          '新旧目标在相同输入上冲突；旧函数误差不能单独证明可塑性或保持能力。')],
    'average_prediction_offpolicy': [
        ('gain_abs_error','gain-error','目标策略奖励率绝对误差','reward_per_step',False,
         '与真实目标策略的解析奖励率比较；行为样本均值回答另一个问题。'),
        ('raw_bias_offset','bias-offset','未锚定差分价值的参考状态值','value',False,
         '漂移诊断，不是优劣分数。主图去掉了这一共同偏移，单看主图会漏掉错误奖励率。')],
    'average_control_learned_model': [
        ('experienced_reward_rate','experienced-rate','全程真实行为奖励率','reward_per_step',True,
         '包含学习与探索成本；主图评价冻结贪心策略，二者不是同一对象。'),
        ('total_backups','total-backups','真实与模型备份总数','updates',False,
         '计算量记账，不是越少越优的性能量。Dyna每个真实步额外做模型更新。')],
    'integrated_continuing_ring': [
        ('average_reward','lifetime-rate','全程真实外部奖励率','reward_per_environment_step',True,
         '包括变化后的适应损失，agent不在变化处重置。主图是近期窗口，不能替代全程目标。'),
        ('model_backups','model-backups','累计模型规划备份数','updates',False,
         '环境步相同不等于计算量相同；规划按高层动作结束调度，技能时长改变计算频率。'),
        ('invalidations','invalidations','技能版本失效次数','events',False,
         '接口维护成本，不是直接性能指标。零次可能因为错误关闭失效处理，不能解释为更稳定。')],
}


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
            for field,*_ in ADDITIONAL_DIAGNOSTICS.get(metas[record['algorithm']]['task'],[]):
                for c,r in zip(csv_rows,rows):
                    value=r.get(field)
                    if type(value) not in (int,float) or not math.isfinite(value) or float(c.get(field,'nan'))!=value:
                        raise ValueError('Diagnostic differs from raw evidence: '+field)
            records.append({**record,'rows':rows})
        curves = json.loads((path.parent/'curves.json').read_text())
        if curves != runtime.summarize(records):
            raise ValueError('Plot data differs from complete raw population')
        if m['algorithms'][0]['task']=='retention_regression':
            diagnostic_records=[{**record,'rows':[{**row,'value':row['old_task_mse']} for row in record['rows']]} for record in records]
            diagnostics.update(runtime.summarize(diagnostic_records))
        for field,*_ in ADDITIONAL_DIAGNOSTICS.get(m['algorithms'][0]['task'],[]):
            diagnostic_records=[{**record,'rows':[{**row,'value':row[field]} for row in record['rows']]} for record in records]
            diagnostics.update({(key,field):points for key,points in runtime.summarize(diagnostic_records).items()})
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
        'docs/implementation-coverage.md')]+[ROOT/'docs'/(slug+'.md') for slug,_ in GUIDES.values()]+[
        ROOT/'tests'/name for name in ('test_extended_classic.py','test_extended_adaptation.py',
                                       'test_extended_knowledge.py','test_extended_multiagent.py',
                                       'test_average_systems.py','test_nonlinear_diagnostics.py',
                                       'test_streaming_composition.py','test_integrated_agents.py',
                                       'test_reading_derivations.py','test_learner_control.py')]
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

本包包含独立运行所需源码、共享依赖及配套机制实验测试。安装PyTorch后运行：

```bash
python3 -m unittest discover -s tests -p 'test_extended_*.py' -v
python3 -m unittest discover -s tests -p 'test_average_systems.py' -v
python3 -m unittest discover -s tests -p 'test_nonlinear_diagnostics.py' -v
python3 -m unittest discover -s tests -p 'test_streaming_composition.py' -v
python3 -m unittest discover -s tests -p 'test_integrated_agents.py' -v
python3 -m unittest discover -s tests -p 'test_reading_derivations.py' -v
python3 -m unittest discover -s tests -p 'test_learner_control.py' -v
```

测试检查更新公式、梯度、终止与数据边界；通过测试不表示原论文性能已经复现。完整教材与全仓检查请看教程仓库。
作者工程只提供固定源码获取工具与清单，未包含完整作者源码、checkpoint或训练成果。
教学曲线不构成一般算法效能或原论文重训证据；输出目录必须是新目录。
许可证索引见LICENSE；教学代码见LICENSE-CODE，文档见LICENSE-DOCS.md。
上游作者工程保留各自许可证。
'''
    # Textbook derivations link directly to inspectable tests as well as the ZIP.
    published_tests = [file for file in bundle_paths if file.parent == ROOT/'tests']
    for file in code_paths + published_tests:
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
    guide_directory = site/'src/data/crl/implementation-guides'
    guide_directory.mkdir(parents=True, exist_ok=True)
    for slug,title in GUIDES.values():
        body = (ROOT/'docs'/(slug+'.md')).read_text(encoding='utf-8')
        (public/(slug+'.md')).write_text(body,encoding='utf-8-sig')
        # The original README-style links are portable in the code ZIP; the web
        # guide instead points at the exact published source assets.
        body = body.replace('../implementations/', '/crl-code/implementations/')
        body = body.replace('\\[', '$$').replace('\\]', '$$').replace('\\(', '$').replace('\\)', '$')
        (guide_directory/(slug+'.md')).write_text(
            '---\ntitle: '+json.dumps(title,ensure_ascii=False)+'\nslug: '+slug+'\n---\n'+body,
            encoding='utf-8')
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
        for field,slug,label,unit,higher,scope in ADDITIONAL_DIAGNOSTICS.get(meta['task'],[]):
            curves={a:diagnostics[(a,field)] for a in selected}
            plot_metas=[{**metas[a],'metric':label,'unit':unit,'higher_better':higher} for a in selected]
            (target/(slug+'.svg')).write_text(runtime.svg_plot(curves,plot_metas),encoding='utf-8')
            (target/(slug+'.json')).write_text(json.dumps(curves,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
            diagnostic_plots.append(dict(key=slug,label=label,metric=label,unit=unit,higher_better=higher,
                                         svg=slug+'.svg',json=slug+'.json',curves=curves,scope=scope))
        # The unchanged group manifest binds every group artifact; include all of
        # them so downloaded evidence can actually be checked against that manifest.
        with zipfile.ZipFile(target/'raw-runs.zip', 'w', zipfile.ZIP_DEFLATED) as archive:
            for name in sorted(manifest['artifacts']):
                archive.write(source_dir/name,name)
            archive.write(source_dir/'manifest.json', 'manifest.json')
        support_paths = []
        if meta['family'] == 'integrated_agents':
            support_paths = [ROOT/'implementations/integrated_agents/_system.py']
        elif meta['family'] == 'learner_control':
            support_paths = [ROOT/'implementations/learner_control/_common.py']
            if path.name == 'frozen_parameters.py':
                support_paths.append(ROOT/'implementations/learner_control/online_differential_q.py')
        supporting_sources = [{'file': file.relative_to(ROOT).as_posix(),
                               'source': file.read_text(encoding='utf-8'),
                               'sha256': hashlib.sha256(file.read_bytes()).hexdigest()} for file in support_paths]
        entries.append({**meta, 'file': path.relative_to(ROOT).as_posix(),
                        'supporting_sources': supporting_sources,
                        'source': path.read_text(encoding='utf-8'),
                        'algorithm_notes': ast.get_docstring(ast.parse(path.read_text(encoding='utf-8'))) or '',
                        'source_sha256': hashlib.sha256(path.read_bytes()).hexdigest(),
                        'seeds': manifest['seeds'], 'steps': manifest['steps'],
                        'curves': selected, 'baseline_name': metas[baseline]['name'],
                        'diagnostic_plots':diagnostic_plots,
                        **({'guide_path':'/zh/continual-rl/code/guides/'+GUIDES[meta['family']][0]+'/',
                            'guide_download':'/crl-code/'+GUIDES[meta['family']][0]+'.md'} if meta['family'] in GUIDES else {}),
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
