"""Generate an auditable human-readable report from saved experiment results."""
import json
from pathlib import Path
import sys
import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT))
from common.research import fingerprint


def main():
    run = ROOT/'weights/Rail Corrugation/runs/2026-09-19-model-experiments'
    read = lambda p:json.loads(p.read_text(encoding='utf-8'))
    q = read(run/'qualification.json')
    results = [read(p) for p in run.glob('screen_*.json') if p.name != 'screen_protocol.json']
    results.sort(key=lambda r:r['mean'],reverse=True)
    lines = ['# Rail model experiment results','',
        'Models and decision parameters were fitted using training data only. No leaderboard submissions were made.',
        'The active models, active prediction CSVs and original submission ZIP were preserved.', '',
        '## Screening','',
        '| Candidate | Block macro-F1 | Repeat SD | Side I F1 | Side II F1 |',
        '|---|---:|---:|---:|---:|']
    for r in results:
        lines.append(f"| {r['name']} | {r['mean']:.4f} | {r['std']:.4f} | {r['per_class']['Side I']:.4f} | {r['per_class']['Side II']:.4f} |")
    lines += ['', '## Confirmation','',
        '| Candidate | Reserved block macro-F1 | Stratified macro-F1 | Qualified | Failed gates |',
        '|---|---:|---:|---|---|']
    for n,r in q.items():
        failed = ', '.join(g for g,v in r['gates'].items() if not v) or 'None'
        lines.append(f"| {n} | {r['confirmation']:.4f} | {r['stratified']:.4f} | {r['qualified']} | {failed} |")
    complementarity = {}
    for r in results:
        if r['name'].startswith('blend__'):
            a,b = r['name'][len('blend__'):].split('__')
            pa = np.load(run/f'screen_{a}_oof.npz')
            pb = np.load(run/f'screen_{b}_oof.npz')
            ea,eb = pa['prediction'] != pa['labels'],pb['prediction'] != pb['labels']
            complementarity[r['name']] = dict(a_wrong_b_right=int((ea & ~eb).sum()),
                b_wrong_a_right=int((eb & ~ea).sum()),both_wrong=int((ea & eb).sum()),
                note='Counts pooled across three rotations; recordings repeat.')
    (run/'blend_complementarity.json').write_text(json.dumps(complementarity,indent=2),encoding='utf-8')
    base = read(run/'confirm_incumbent.json')
    strat = read(run/'strat_incumbent.json')
    lines += ['', f"Matched incumbent: confirmation block macro-F1 **{base['mean']:.4f}**; stratified check **{strat['mean']:.4f}**.",
        'Compare candidates against these matched results, not against older scores from different splits or threshold-selection procedures.',
        '', '## Packaged candidates','']
    for out in read(run/'outputs.json'):
        folder = Path(out)
        meta = read(folder/'submission.json')
        lines += [f"### {meta['name']}",'',f"Archive: `{folder.relative_to(ROOT).as_posix()}/predictions.zip`",'',
            f"ZIP SHA-256: `{meta['zip_sha256']}`",'',f"Final decision: `{meta['decision']}`",'',
            'Per-class confirmation metrics (mean across the three rotations):','',
            '| Class | Precision | Recall | F1 |','|---|---:|---:|---:|']
        report = read(run/f"confirm_{meta['name']}.json")
        for cls in ['Normal','Side I','Side II']:
            metrics = [sum(x[cls][key] for x in report['classification_reports'])/3 for key in ['precision','recall','f1-score']]
            lines.append(f'| {cls} | {metrics[0]:.4f} | {metrics[1]:.4f} | {metrics[2]:.4f} |')
        lines += ['']
    duplicate_path = run/'duplicate_candidates.json'
    if duplicate_path.exists() and read(duplicate_path):
        lines += ['## Equivalent predictions','',
            'These qualified candidates generated the same test labels as a higher-ranked saved candidate and were not counted as extra submissions: '+', '.join(read(duplicate_path))+'.','']
    lines += ['## Limits and interpretation','',
        '- Screening and confirmation rotations reuse the same 272 recordings; the latter are partition-stability checks, not an independent test set.',
        '- Repeat SD is not a confidence interval. There are only 14 Side I fault recordings.',
        '- The feature follow-ups were motivated by screening results. Searching more models can overfit validation; the hidden scores remain unknown.',
        '- A second classifier/settings round was required because the initial third qualified model duplicated an existing test prediction vector. Confirmation partitions were reused for this round and were no longer untouched; gates were unchanged.',
        '- Both MiniROCKET variants failed screening. They used anti-aliased 2.5 kHz vibration signals; this does not reject all learned representations.',
        '- Cross-car features measure spectral consistency. No shared-defect timing assumption or test-batch adaptation was used.',
        '- An ensemble is accepted only if it also beats its strongest component in both screening and confirmation.',
        '- Keep the final upload available to restore the best observed archive. These local results do not guarantee leaderboard improvements.',
        '', 'See `docs/research/rail_model_experiments.md` for the protocol, feature definitions, acceptance gates and reproduction commands.', '']
    (run/'REPORT.md').write_text('\n'.join(lines),encoding='utf-8')
    import importlib.metadata
    metadata = dict(python=sys.version, versions={n:importlib.metadata.version(n) for n in ['numpy','pandas','scipy','scikit-learn','sktime','numba','joblib','lightgbm']},
                    sources={p.name:fingerprint(p) for p in Path(__file__).parent.glob('*.py')},
                    trials=len(results), shortlisted=len(q), qualified=sum(r['qualified'] for r in q.values()))
    (run/'run_summary.json').write_text(json.dumps(metadata,indent=2),encoding='utf-8')
    print(run/'REPORT.md')


if __name__ == '__main__':
    main()
