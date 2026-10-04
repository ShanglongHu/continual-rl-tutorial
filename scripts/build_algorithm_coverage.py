#!/usr/bin/env python3
"""Synchronize the preserved textbook inventory with literal implementation META.

An executable component is not promoted to a complete method. Original formula
entries retain their stable IDs and source provenance. Running this command does
not certify experimental results; build_learning_gallery verifies those separately.
"""
import copy
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from implementations import runtime

LABELS = {
    'independent_implementation': '独立教学算法与基线（小型问题，不是论文规模复现）',
    'independent_component': '独立子机制实验（有学习循环、对照与明确适用范围）',
    'formula_core': '仅综合 lab 公式核（尚无独立实验）',
    'author_project': '作者工程固定源码入口（未完整训练）',
    'unintegrated': '正文涉及，尚未接入',
}


def build(previous, paths=None):
    paths = runtime.discover() if paths is None else paths
    metas = {key: runtime.metadata(path) for key, path in paths.items()}
    # Restore the exact old inventory entry before upgrading it. This keeps the
    # transformation idempotent and avoids growing nested metadata on each run.
    original = {e['id']: copy.deepcopy(e.get('formula_origin', e))
                for e in previous['entries']}
    formula_ids = {key for key, e in original.items() if e['status'] == 'formula_core'}
    refs = {}
    for key, meta in metas.items():
        for ref in meta.get('coverage_refs', []):
            if ref not in formula_ids:
                raise ValueError('Unknown original formula entry: '+ref)
            if ref in refs:
                raise ValueError('Ambiguous primary implementation for '+ref)
            refs[ref] = key
        if meta.get('implementation_kind', 'teaching_method') not in ('teaching_method', 'component_experiment'):
            raise ValueError('Unknown implementation kind: '+key)
        if meta['baseline'] not in metas:
            raise ValueError('Missing baseline: '+key)
        runtime.comparable([meta, metas[meta['baseline']]])

    def implementation(key, entry_id, origin=None):
        m = metas[key]
        component = m.get('implementation_kind') == 'component_experiment'
        chapters = list(dict.fromkeys((origin or {}).get('chapter_paths', [])+m['chapter_paths']))
        e = dict(id=entry_id, name=(origin or {}).get('name', m['name']), algorithm=m['name'],
                 status='independent_component' if component else 'independent_implementation',
                 implementation_id=key, file=paths[key].relative_to(ROOT).as_posix(),
                 chapter_paths=chapters, implementation_chapter_paths=m['chapter_paths'],
                 scope=m['scope'], role=m.get('implementation_kind', 'teaching_method'),
                 task=m['task'], metric=m['metric'], unit=m['unit'], budget=m['budget'],
                 baseline=m['baseline'], description=m['description'],
                 limitations=m.get('limitations', '小型教学实例，不代表论文规模复现。'),
                 run_evidence='运行是否完成及结果来源，以对应实验 manifest 和全部种子记录为准。',
                 reference=m['sources'], metadata=m)
        if origin:
            e['formula_origin'] = origin
        return e

    result, used = [], set()
    for old in previous['entries']:
        origin = original[old['id']]
        if origin['id'] in refs:
            key = refs[origin['id']]
            result.append(implementation(key, origin['id'], origin))
            used.add(key)
        elif origin.get('implementation_id') in metas:
            key = origin['implementation_id']
            result.append(implementation(key, origin['id']))
            used.add(key)
        else:
            result.append(origin)
    for key in metas:
        if key not in used:
            result.append(implementation(key, 'implementation-'+key))
    counts = {status: sum(e['status'] == status for e in result) for status in LABELS}
    return dict(schema_version=2, status_enum=LABELS, counts=counts,
                count_note='条目含算法、子机制和对照基线；不能据此计算独特算法覆盖率或声称全部论文已复现。',
                formula_inventory=dict(total=len(formula_ids), independent=len(refs), remaining=len(formula_ids-refs.keys())),
                entries=result)


def markdown(coverage):
    lines = ['# 正文算法实现覆盖盘点', '',
             '先选择教材问题，再确认实现范围。独立教学算法与独立子机制实验都提供单文件更新、运行入口、对照与记录。子机制实验只回答限定的问题，不能代替原论文完整系统。', '',
             '机器清单：`integrations/algorithm_coverage.json`。本表不认证训练完成；实际结果须核对对应实验的全部种子记录、配置与源码哈希。', '',
             '## 范围与阅读方式', '',
             '- **独立教学算法**：在明确的小型问题中实现算法更新与学习循环。',
             '- **独立子机制实验**：运行一种估计器、更新机制或受限特例，保留未实现的部分。',
             '- **公式核**：只有综合实验中的公式函数，尚未形成独立实验。',
             '- **作者工程**：保留作者源码入口；不表示已完成论文规模训练。',
             '- **尚未接入**：教材已有讲解，但未提供可运行入口。', '',
             '计数包含算法变体和基线，不是互斥的研究方法数。', '']
    for status, label in LABELS.items():
        items = [e for e in coverage['entries'] if e['status'] == status]
        lines += ['## '+label+'（'+str(len(items))+' 项）', '']
        for e in items:
            lines += ['### '+e['name'], '']
            if e.get('implementation_id'):
                key = e['implementation_id']
                lines += ['- 源码：['+e['file']+'](../'+e['file']+')。',
                          '- 页面：[问题、源码与真实对照](https://yingwen.io/zh/continual-rl/code/'+key+'/)。',
                          '- 任务与对照：`'+e['task']+'`；`'+e['baseline']+'`。',
                          '- 指标与预算：'+e['metric']+'；`'+e['budget']+'`。',
                          '- 设定：'+e['description'], '- 范围：'+e['limitations']]
            else:
                lines += [e.get('description') or str(e.get('reference', '')), '']
                if e.get('file'):
                    lines += ['源码：['+e['file']+'](../'+e['file']+')。']
                if e.get('limitations'):
                    lines += ['范围：'+e['limitations']]
            lines += ['- 教材：'+' · '.join('['+p.rstrip('/')+'](https://yingwen.io/zh/continual-rl/'+p+')' for p in e['chapter_paths']), '']
    return '\n'.join(lines).rstrip()+'\n'


def main():
    path = ROOT/'integrations/algorithm_coverage.json'
    current = json.loads(path.read_text(encoding='utf-8'))
    result = build(current)
    path.write_text(json.dumps(result, ensure_ascii=False, indent=2)+'\n', encoding='utf-8')
    (ROOT/'docs/implementation-coverage.md').write_text(markdown(result), encoding='utf-8')
    print(json.dumps({'counts': result['counts'], 'formula_inventory': result['formula_inventory']}, ensure_ascii=False))


if __name__ == '__main__':
    main()
