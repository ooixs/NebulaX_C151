"""Report the saved speed-ablation campaign without fitting or loading models."""
import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import shutil
import statistics
import sys


ROOT = Path(__file__).resolve().parents[2]
RUN = ROOT / 'weights/Rail Corrugation/runs/2026-09-19-ensemble-speed-ablation'
CLASSES = ('Normal', 'Side I', 'Side II')
LABELS = {
    'ensemble': 'Submitted ensemble (matched comparator)',
    'sg_speed_removed': 'Remove SG speed inputs',
    'both_speed_removed': 'Remove SG and legacy speed inputs',
}


def read(path):
    return json.loads(path.read_text(encoding='utf-8'))


def fingerprint(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def number(value):
    return f'{value:.4f}'


def robustness_state(qualification, result):
    if result is not None:
        return 'completed'
    if not any(q.get('block_qualified', False) for q in qualification.values()):
        return 'skipped: neither variant passed the block gates'
    return 'not yet available; required before any upload recommendation'


def artifacts(run, qualification):
    """A file is reportable as a candidate only with explicit qualification."""
    path = run / 'outputs.json'
    if not path.exists():
        return [], 'No outputs.json exists; no new submission ZIP is listed.'
    output = read(path)
    if not isinstance(output, list):
        raise ValueError('outputs.json must be a list of candidate folders or ZIP paths')
    verified = []
    for entry in output:
        if not isinstance(entry, str):
            raise ValueError('Each outputs.json entry must be an artifact path string')
        folder = Path(entry)
        if not folder.is_absolute():
            folder = ROOT / folder
        archive = folder if folder.suffix.lower() == '.zip' else folder / 'predictions.zip'
        metadata = archive.parent / 'submission.json'
        if not archive.is_file() or not metadata.is_file():
            raise ValueError(f'Listed output is missing its ZIP or submission metadata: {entry}')
        name = read(metadata).get('name')
        if not qualification.get(name, {}).get('qualified', False):
            raise ValueError(f'Listed output lacks complete qualification: {entry}')
        verified.append(dict(name=name, path=str(archive), sha256=fingerprint(archive)))
    return verified, ('Qualified output paths verified.' if verified else
                      'outputs.json is empty; no new submission ZIP is listed.')


def snapshot_sources(run, plan):
    directory = run / 'source'
    directory.mkdir(exist_ok=True)
    sources = {}
    paths = list(Path(__file__).parent.glob('*.py'))
    paths += [ROOT / 'common/metrics.py', ROOT / 'common/research.py']
    for path in sorted(paths):
        relative = path.relative_to(ROOT)
        target = directory / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        # Preserve a previous snapshot if code changes after report generation.
        actual = fingerprint(path)
        if target.exists() and fingerprint(target) != actual:
            target = target.with_name(f'{target.stem}-{actual[:12]}{target.suffix}')
        shutil.copyfile(path, target)
        expected = plan.get('sources', {}).get(path.name) if path.parent == Path(__file__).parent else None
        sources[relative.as_posix()] = dict(
            sha256=actual, planned_sha256=expected,
            matches_plan=(actual == expected) if expected else None,
            snapshot=str(target.relative_to(run)),
        )
    return sources


def generate(run):
    required = ['plan.json', 'qualification.json', 'screen_results.json', 'confirm_results.json']
    missing = [name for name in required if not (run / name).is_file()]
    if missing:
        raise ValueError('Campaign is incomplete; missing required results: ' + ', '.join(missing))
    plan, qualification, screen, confirm = [read(run / name) for name in required]
    names = list(plan['variants'])
    for tag, result in [('screen', screen), ('confirm', confirm)]:
        for name in names:
            if name not in result:
                raise ValueError(f'{tag} results are missing {name}')
    optional = {tag: read(run / f'{tag}_results.json') if (run / f'{tag}_results.json').exists() else None
                for tag in ['strat', 'seed_1000', 'seed_2000', 'speed']}
    outputs, output_status = artifacts(run, qualification)
    sources = snapshot_sources(run, plan)
    qualified = [name for name, q in qualification.items() if q.get('qualified', False)]
    if qualified:
        outcome = 'Passed the saved validation gates: ' + ', '.join(qualified) + '.'
    elif not any(q.get('block_qualified', False) for q in qualification.values()):
        outcome = 'Neither speed-ablation variant passed the block validation gates. No new upload is recommended.'
    else:
        outcome = 'No variant has passed all required validation gates. No new upload is recommended.'
    lines = ['# Submitted ensemble: explicit speed-input ablations', '', outcome, '',
             'The comparator is the successful submitted ensemble, evaluated on the same duplicate-safe '
             'outer and inner splits as both variants. All models use 800 trees and fixed 75:25 component '
             'weights. Primary thresholds are selected using inner folds only. No leaderboard upload '
             'is performed by this report.', '',
             '## Matched block results', '',
             '| Candidate | Screening macro-F1 ± repeat SD | Confirmation macro-F1 ± repeat SD | Confirmation gain |',
             '|---|---:|---:|---:|']
    for name in names:
        s, c = screen[name], confirm[name]
        lines.append(f"| {LABELS.get(name, name)} | {s['mean']:.4f} ± {s['std']:.4f} | "
                     f"{c['mean']:.4f} ± {c['std']:.4f} | {c['mean']-confirm['ensemble']['mean']:+.4f} |")
    lines += ['', 'Confirmation class F1 scores are means across the three repeated partitions.', '',
              '| Candidate | Normal F1 | Side I F1 | Side II F1 |', '|---|---:|---:|---:|']
    for name in names:
        lines.append('| ' + LABELS.get(name, name) + ' | ' + ' | '.join(
            number(confirm[name]['per_class'][cls]) for cls in CLASSES) + ' |')
    lines += ['', '### Partition scores', '',
              '| Candidate | Screening rotations | Confirmation rotations |', '|---|---|---|']
    for name in names:
        lines.append('| ' + LABELS.get(name, name) + ' | ' + ', '.join(map(number, screen[name]['scores'])) +
                     ' | ' + ', '.join(map(number, confirm[name]['scores'])) + ' |')
    lines += ['', '## Prespecified gates', '', plan['block_gates'], '', plan['robustness_gates'], '',
              '| Variant | Passed block gates | Fully qualified | Failed recorded gates |', '|---|---|---|---|']
    for name, q in qualification.items():
        failed = ', '.join(gate for gate, passed in q['gates'].items() if not passed) or 'None'
        lines.append(f"| {LABELS.get(name, name)} | {q.get('block_qualified', False)} | "
                     f"{q.get('qualified', False)} | {failed} |")
    lines += ['', '## Secondary fixed-threshold sensitivity', '',
              'These scores use the submitted threshold τ = 0.275 for every outer fold. '
              'They are a diagnostic and do not replace the prespecified nested-threshold gates.', '',
              '| Candidate | Screening mean | Confirmation mean | Confirmation gain |', '|---|---:|---:|---:|']
    fixed = {}
    reference = statistics.mean(confirm['ensemble']['fixed_threshold_scores'])
    for name in names:
        s = statistics.mean(screen[name]['fixed_threshold_scores'])
        c = statistics.mean(confirm[name]['fixed_threshold_scores'])
        fixed[name] = dict(screen=s, confirm=c, confirmation_gain=c-reference)
        lines.append(f'| {LABELS.get(name, name)} | {s:.4f} | {c:.4f} | {c-reference:+.4f} |')
    lines += ['', '## Robustness checks', '']
    states = {tag: robustness_state(qualification, result) for tag, result in optional.items()}
    for tag, result in optional.items():
        lines += [f'### {tag}', '', states[tag] + '.', '']
        if result is None:
            continue
        lines += ['| Candidate | Macro-F1 ± repeat SD | Gain over matched ensemble |', '|---|---:|---:|']
        for name, metrics in result.items():
            lines.append(f"| {LABELS.get(name, name)} | {metrics['mean']:.4f} ± {metrics['std']:.4f} | "
                         f"{metrics['mean']-result['ensemble']['mean']:+.4f} |")
        lines += ['']
        if tag == 'speed':
            lines += ['Speed diagnostics leave each range out of training. Confusion rows are true '
                      'classes and columns are predicted classes in order: Normal, Side I, Side II. '
                      'Class support is shown because individual ranges can contain few or no faults.', '']
            for name, metrics in result.items():
                for record in metrics.get('ranges', []):
                    lines += [f"- {LABELS.get(name, name)}, speed bin {record['speed_bin']}: "
                              f"support `{json.dumps(record['support'], sort_keys=True)}`; "
                              f"confusion `{json.dumps(record['confusion'])}`."]
            lines += ['']
    lines += ['## Submission artifacts', '', output_status, '']
    for out in outputs:
        lines += [f"- {out['name']}: `{out['path']}` (SHA-256 `{out['sha256']}`)."]
    lines += ['', '## Feature changes and limits', '',
              '- SG inputs removed: ' + ', '.join(f'`{name}`' for name in plan['sg_drop']) + '.',
              '- The second variant additionally removes these legacy inputs: ' +
              ', '.join(f'`{name}`' for name in plan['legacy_drop']) + '.',
              '- Legacy speed-conditioned normal references, wavelength features, and indirect speed '
              'information in vibration remain. This does not establish that all speed information is removed.',
              '- Both forests retain `max_features=sqrt`. Removing columns changes the number sampled '
              'per split (legacy-plus-SG: 21 to 20; SG: 11 to 10). This evaluates the configured ablation '
              'and does not isolate a causal effect of speed alone.',
              '- Screening and confirmation reuse the same recordings with different partitions. '
              'They are partition-stability checks, not independent held-out datasets. Repeat SD is not a confidence interval.',
              '- There are only 14 Side I recordings. Fault-class estimates and speed-range checks '
              'can be unstable because of small support.',
              '- The follow-up was chosen after previous local experiments and leaderboard feedback. '
              'Repeated use of these recordings can overfit validation. No per-file hidden labels are available.',
              '- Additional seed checks, when present, use only the first five confirmation folds '
              '(one complete partition), with matched baseline seeds.',
              '- These local results cannot guarantee hidden-test improvement. Preserve the successful '
              'ensemble archive as the fallback.', '', '## Provenance', '',
              f"Submitted ensemble ZIP SHA-256 recorded at plan creation: `{plan['winner_zip_sha256']}`.", '',
              'Current source files are snapshotted under `source/`. `summary.json` records snapshot '
              'hashes, plan hashes, result hashes, and any source mismatch. A current snapshot is not '
              'claimed to reconstruct an earlier source version whose hash differs.', '']
    mismatches = [name for name, item in sources.items() if item['matches_plan'] is False]
    if mismatches:
        lines += ['Source files that differ from the plan-time hash: ' + ', '.join(f'`{n}`' for n in mismatches) + '.', '']
    inputs = {p.name: fingerprint(p) for p in run.glob('*.json') if p.name != 'summary.json'}
    summary = dict(generated_utc=datetime.now(timezone.utc).isoformat(), outcome=outcome,
                   plan=plan, screen=screen, confirmation=confirm, qualification=qualification,
                   fixed_threshold_sensitivity=fixed, robustness=optional, robustness_status=states,
                   outputs=outputs, output_status=output_status, source_snapshots=sources,
                   source_mismatches=mismatches, input_sha256=inputs)
    (run / 'REPORT.md').write_text('\n'.join(lines), encoding='utf-8')
    (run / 'summary.json').write_text(json.dumps(summary, indent=2, allow_nan=False), encoding='utf-8')
    return run / 'REPORT.md'


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run', type=Path, default=RUN)
    args = parser.parse_args()
    try:
        report = generate(args.run)
    except (ValueError, KeyError, OSError) as exc:
        print(f'Report not generated: {exc}', file=sys.stderr)
        return 2
    print(report)
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
