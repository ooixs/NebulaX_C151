"""Summarize the registered architecture campaign without fitting any models."""
from __future__ import annotations

import argparse
import hashlib
import json
from collections import Counter
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
RUN = ROOT / 'weights/Rail Corrugation/runs/2026-09-19-architecture-experiments'
TAGS = ['screen', 'confirm', 'strat', 'seed_1000', 'seed_2000', 'speed']
CLASSES = ['Normal', 'Side I', 'Side II']


def read(path):
    return json.loads(path.read_text(encoding='utf-8'))


def compact(result):
    return {key: result[key] for key in ['mean', 'std', 'scores', 'per_class']}


def screening_gates(name, results):
    candidate, baseline = results[name], results['ensemble']
    gates = {
        'screen_gain': candidate['mean'] > baseline['mean'] + .005,
        'screen_consistent': all(a >= b - 1e-12 for a, b in
                                 zip(candidate['scores'], baseline['scores'])),
    }
    if name.startswith('blend__'):
        gates['screen_beats_component'] = (
            candidate['mean'] > results[name.removeprefix('blend__')]['mean'] + .005)
    return gates


def table(headers, rows):
    return '\n'.join(['| ' + ' | '.join(headers) + ' |',
                      '| ' + ' | '.join(['---'] * len(headers)) + ' |'] +
                     ['| ' + ' | '.join(map(str, row)) + ' |' for row in rows])


def score(value):
    return f"{value['mean']:.6f} +/- {value['std']:.6f}"


def main(run=RUN):
    RUN = Path(run).resolve()
    plan = read(RUN / 'plan.json')
    names = ['ensemble'] + plan['candidates']
    results = {tag: {name: compact(read(path)) for name in names
                     if (path := RUN / f'{tag}_{name}.json').exists()}
               for tag in TAGS}
    missing = set(names) - set(results['screen'])
    if missing:
        raise ValueError(f'Screening has not finished: {sorted(missing)}')
    qualification = read(RUN / 'qualification.json') if (RUN / 'qualification.json').exists() else {}
    shortlist = list(qualification)
    qualified = [name for name in shortlist if qualification[name].get('qualified')]
    disqualified = {name: value['submission_disqualification'] for name, value in qualification.items()
                    if value.get('submission_disqualification')}
    screening = {name: screening_gates(name, results['screen']) for name in plan['candidates']}
    with np.load(RUN / 'screen_ensemble_predictions.npz') as data:
        labels = data['labels']
    counts = {name: int(np.sum(labels == name)) for name in CLASSES}
    packages = {path.stem.removeprefix('package_'): read(path)
                for path in sorted(RUN.glob('package_*.json'))}
    locally_qualified = [name for name, value in qualification.items()
                         if value.get('qualified') or value.get('local_qualified')]
    for name, package in packages.items():
        if package.get('duplicate_of') or package.get('status') in [
                'qualified-but-duplicate', 'disqualified-historical-duplicate']:
            duplicate = package.get('duplicate_of', 'a previously generated or submitted prediction set')
            description = duplicate if isinstance(duplicate, str) else ', '.join(duplicate)
            disqualified.setdefault(name, 'Exact duplicate of ' + description + '; excluded from submission.')
    qualified = [name for name in qualified if name not in disqualified]
    outputs = read(RUN / 'outputs.json') if (RUN / 'outputs.json').exists() else []
    verification = {}
    for output in outputs:
        path = Path(output) / 'verification.json'
        if path.exists():
            verification[output] = read(path)
    audit = read(RUN / 'cache_signature_audit.json') if (RUN / 'cache_signature_audit.json').exists() else []
    history_path = RUN / 'historical_prediction_reconstruction.json'
    history = read(history_path) if history_path.exists() else None
    calibration_records = {}
    for path in sorted(RUN.glob('final_calibration_*.json')):
        record = read(path)
        if isinstance(record, dict) and 'calibration_macro_f1' in record:
            calibration_records[path.stem.removeprefix('final_calibration_')] = {
                key: record[key] for key in ['method', 'decision', 'weight', 'calibration_macro_f1',
                                             'score_context', 'input_signature', 'record_sha256'] if key in record}
    grouping_path = RUN / 'final_calibration_grouping_check.json'
    grouping_check = read(grouping_path) if grouping_path.exists() else None
    branch_audit = ROOT / 'docs/research/rail_branch_audit_20260919.md'
    speed_support = []
    if (RUN / 'speed_protocol.json').exists():
        for fold in read(RUN / 'speed_protocol.json'):
            support = Counter(labels[fold['validation']].tolist())
            speed_support.append(dict(speed_bin=fold['speed_bin'],
                                      support={cl: support[cl] for cl in CLASSES}))
    caveats = [
        f"All {len(plan['candidates'])} registered hypotheses and prior campaigns reuse the same 272 recordings; "
        'selection across them creates optimism that these checks cannot eliminate.',
        'Confirmation rotations are shifted contiguous-block cross-validation on the same '
        'recordings, not an independent holdout. Rotation standard deviations are not '
        'confidence intervals or independent-replicate uncertainty.',
        'Only 14 recordings have Side I labels. Small changes can cause large fault-class '
        'F1 changes, and hidden-set improvements are unproven.',
        'The speed test pools all predictions from leave-one-speed-range-out folds into '
        'one macro-F1. It does not guarantee performance in every individual speed range, '
        'and its standard deviation is zero because there is one pooled score.',
        'The successful ensemble has user-reported hidden Rail macro-F1 0.7687255273462169. '
        'A local improvement does not establish a hidden improvement; predictions identical '
        'to a previously submitted candidate do not justify another upload.',
    ]
    summary = dict(plan=plan, recordings=len(labels), class_support=counts,
                   registered_hypotheses=len(plan['candidates']), results=results,
                   screening_gates=screening, qualification=qualification,
                   qualified=qualified, locally_qualified=locally_qualified, speed_range_support=speed_support,
                   submission_disqualifications=disqualified,
                   historical_prediction_reconstruction=history,
                   final_calibration=calibration_records, final_calibration_grouping_check=grouping_check,
                   branch_audit=str(branch_audit) if branch_audit.exists() else None,
                   cache_signature_audit=audit, packages=packages, outputs=outputs,
                   verification=verification, caveats=caveats)
    evidence = [RUN / 'plan.json', RUN / 'qualification.json', Path(__file__)]
    evidence += [RUN / f'{tag}_{name}.json' for tag, values in results.items() for name in values]
    evidence += list(RUN.glob('final_calibration_*.json'))
    if branch_audit.exists():
        evidence.append(branch_audit)
    summary['evidence_sha256'] = {str(path.relative_to(ROOT) if path.is_relative_to(ROOT) else path): hashlib.sha256(path.read_bytes()).hexdigest()
                                 for path in evidence if path.exists()}
    (RUN / 'summary.json').write_text(json.dumps(summary, indent=2, allow_nan=False), encoding='utf-8')

    title = ('Rail weighting and spectral-pooling experiments' if 'weighting' in plan else
             'Rail forest experiments' if plan.get('final_calibration') else 'Rail architecture experiments')
    provenance = []
    if 'sg_nospeed_mf05' in plan['candidates']:
        provenance += [
            '`sg_nospeed_mf05` is the previously evaluated SG-only random forest with '
            '`max_features=0.5`, removing the six explicit speed/movement/speed-normalized '
            'SG inputs. It is **not a newly discovered representation**. The branch campaign '
            'rejected it against `sg_rf_mf03`, which later underperformed on the hidden set. '
            'This campaign compares it with the successful submitted ensemble and adds '
            'registered seed and speed-transfer gates. The preceding ensemble speed ablations '
            'changed different models and did not pass; their negative results remain valid.', '',
            'The other hypotheses use paired multiclass models or hierarchical fault-detection '
            'and side-localization models, with SG or legacy-plus-SG inputs. Their 75:25 mixtures '
            'and blends with the successful ensemble are included in the complete screening table.', '',
        ]
    for key in ['rationale', 'weighting', 'spectral']:
        if key in plan:
            provenance += [f"**{key.capitalize()}:** {plan[key]}", '']
    provenance += [table(['Registered component', 'Configuration'],
                        [[f'`{name}`', '`' + json.dumps(config, sort_keys=True) + '`']
                         for name, config in plan['configs'].items()]), '',
                   'Each named mixture uses 75% of its legacy-plus-SG component and '
                   '25% of its SG component. The `blend__` variants additionally combine '
                   'the named component or mixture with the submitted ensemble using '
                   'inner-fold weight selection.', '']
    lines = [f'# {title}', '']
    if qualified:
        lines += [f"**Eligible after the recorded validation checks:** {', '.join('`' + n + '`' for n in qualified)}.", '']
    else:
        lines += ['**No candidate is currently eligible for another submission.**', '']
    for name, reason in disqualified.items():
        lines += [f'**Do not submit `{name}`:** {reason}', '']
    lines += [
        f"The campaign screened {len(plan['candidates'])} registered hypotheses against the submitted "
        '75:25 legacy-plus-SG/SG random-forest ensemble. Its user-reported hidden Rail score is '
        '**0.7687255273462169**. All scores below are local macro-F1; a passing candidate is '
        'a defensible final experiment, not an assured leaderboard improvement.', '',
        '## Candidate provenance', '', *provenance,
        '## Validation protocol and gates', '',
        f"There are {len(labels)} labelled recordings: " + ', '.join(f'{counts[cl]} {cl}' for cl in CLASSES) + '. '
        'The known exact-duplicate families (Train107/Train115 and Train165/Train187) '
        'stay together in all inner and outer splits. Both side views of each file stay '
        'together. Any learned Normal reference and classifier are fitted on training '
        'folds only.', '',
        'Screening uses five contiguous blocks at offsets 0, 27 and 54; confirmation '
        'uses offsets 14, 41 and 68. Each outer fold uses three inner blocks to choose '
        'the global maximum-side-probability threshold. A file is Normal unless a side '
        'exceeds that threshold; the larger side score determines its fault side. '
        'For `blend__` candidates, weights 0.25, 0.5 and 0.75 are also selected inside '
        'the training fold. Outer labels do not select these decision parameters.', '',
        '- Both block-round means must beat the ensemble by more than 0.005, with no '
        'regression in any paired rotation. Confirmation F1 for either fault class may '
        'fall by at most 0.03.',
        '- A blend must also beat its non-blended candidate by more than 0.005 in both rounds.',
        '- Grouped stratified macro-F1 may fall by at most 0.02.',
        '- Additional fixed seed offsets 1000 and 2000 rerun the first confirmation '
        'rotation for candidate and baseline. Neither paired gain may be negative; '
        'their mean gain must exceed 0.005. These seeds are checks, not seed selection.',
        '- Pooled leave-one-speed-range-out macro-F1 may fall by at most 0.02.',
        '- Packaging additionally requires a distinct test label vector and exact replay '
        'from the saved model. Gates were not relaxed to obtain an archive.', '',
        'The following tables show mean +/- population standard deviation across '
        'rotations. Single pooled evaluations have zero reported standard deviation.', '',
        '## All screening results', '',
    ]
    rows = []
    baseline = results['screen']['ensemble']['mean']
    for name in sorted(names, key=lambda n: -results['screen'][n]['mean']):
        result = results['screen'][name]
        if name == 'ensemble':
            status = 'Reference'
        else:
            failures = [gate for gate, passed in screening[name].items() if not passed]
            status = 'Pass screening' if not failures else 'Fail: ' + ', '.join(failures)
        rows.append([f'`{name}`', score(result), f"{result['mean']-baseline:+.6f}", status])
    lines += [table(['Candidate', 'Screening', 'Gain', 'Screening gate result'], rows), '',
              '## Shortlist: confirmation and robustness', '']
    for name in shortlist:
        lines += [f'### `{name}`', '']
        if name in disqualified:
            lines += [f'**Submission rejected:** {disqualified[name]}', '']
        elif qualification[name].get('block_qualified') and not qualification[name].get('qualified'):
            if any(name not in results[tag] for tag in TAGS[2:]):
                lines += ['**Status: robustness checks pending.** This candidate passed block '
                          'validation and has not been rejected.', '']
            else:
                lines += ['**Status: failed robustness qualification.** See the recorded gates below.', '']
        rows = []
        for tag in TAGS:
            ref = results[tag].get('ensemble')
            candidate = results[tag].get(name)
            if ref and candidate:
                rows.append([tag, score(ref), score(candidate), f"{candidate['mean']-ref['mean']:+.6f}"])
            else:
                status = 'Not run: block gates failed' if not qualification[name].get('block_qualified') else 'Pending'
                rows.append([tag, 'Unavailable' if not ref else score(ref), status, '—'])
        lines += [table(['Evaluation', 'Submitted ensemble', 'Candidate', 'Paired gain'], rows), '']
        if name in results['confirm']:
            ref, candidate = results['confirm']['ensemble'], results['confirm'][name]
            lines += [table(['Confirmation class F1', 'Submitted ensemble', 'Candidate', 'Gain'],
                            [[cl, f"{ref['per_class'][cl]:.6f}", f"{candidate['per_class'][cl]:.6f}",
                              f"{candidate['per_class'][cl]-ref['per_class'][cl]:+.6f}"] for cl in CLASSES]), '']
        lines += [table(['Registered gate', 'Result'],
                        [[gate, 'PASS' if passed else 'FAIL'] for gate, passed in qualification[name]['gates'].items()]), '']
    lines += ['## Speed-transfer interpretation', '',
              'The speed evaluation pools predictions across four held-out speed ranges; '
              'it does not average four range macro-F1 values. Some ranges contain no '
              'Side I or no Side II examples. Passing its pooled tolerance does not '
              'establish reliable performance in every range. The pooled score should be '
              'read with the per-class scores in `summary.json` and support counts below. '
              'If a candidate fails block gates, the speed test is not required and remains '
              'unrun for that candidate.', '']
    if speed_support:
        lines += [table(['Speed bin', 'Normal support', 'Side I support', 'Side II support'],
                        [[row['speed_bin']] + [row['support'][cl] for cl in CLASSES] for row in speed_support]), '']
    if plan.get('final_calibration'):
        lines += ['## Final model calibration', '',
                  f"The registered deployment rule is `{plan['final_calibration']}`. "
                  f"It was frozen at `{plan.get('final_calibration_frozen_utc', 'not recorded')}`.", '']
        if plan['final_calibration'] == 'full_train_threefold_oof_v1':
            lines += [
                'Apply the same duplicate-safe three-fold inner-calibration procedure to '
                'all 272 training recordings, with fitting seeds 10000, 10001 and 10002. '
                'Select the global threshold and any applicable blend weight using those '
                'out-of-fold predictions and the validated grids and tie rules, then fit '
                'the final model on all training recordings with seed 0. The selected '
                'parameters are frozen before test-feature extraction and are not retried '
                'to obtain distinct predictions.', '',
                'This replaces the older median-of-outer-fold-parameters deployment '
                'heuristic. The adjustment followed discovery of an earlier historical '
                'duplicate and was documented before inspection of any forest-candidate '
                'test predictions; it is not described as an untouched preregistration. '
                'The nested validation scores assess the inner selection algorithm. '
                '**The final calibration score is a training-selection score, not an '
                'independent validation result.**', '',
            ]
        else:
            lines += [str(plan.get('final_calibration_policy', 'See the registered plan for the deployment rule.')), '']
        if calibration_records:
            lines += [table(['Candidate', 'Selected decision', 'Blend weight', 'Training calibration macro-F1'],
                            [[f'`{name}`', '`' + json.dumps(record.get('decision'), sort_keys=True) + '`',
                              record.get('weight') if record.get('weight') is not None else 'Not applicable',
                              f"{record['calibration_macro_f1']:.6f}"]
                             for name, record in calibration_records.items()]), '']
        if grouping_check is not None:
            lines += ['`final_calibration_grouping_check.json` records the comparison between '
                      'the final calibration grouping algorithm and the saved validation protocols.', '']
    cache_text = ('The registered source hashes and protocol files document fitting and cached '
                  'baseline provenance. No cache-signature correction audit is recorded in this run.')
    if audit:
        cache_text = ('The prior SG speed-ablation cache was reused only after verifying protocol, '
                      'feature ordering, model settings, seeds and source hashes. An independent '
                      'retrain of its first outer fold reproduced all 110 side probabilities exactly. '
                      'A cache-signature bookkeeping correction did not change model fitting. '
                      '`cache_signature_audit.json` records each affected cache and exact first-fold '
                      'replay; all recorded replay checks passed.'
                      if all(a.get('outer_fold0_exact') for a in audit) else
                      'The cache audit contains an unsuccessful replay; inspect the audit before relying on reused results.')
    lines += ['## Cache and reproducibility audit', '', cache_text, '',
              '`summary.json` includes the registered plan, all results, gates, evidence '
              'hashes, cache audit and any package verification records.', '',
              '## Submission artifacts', '']
    if outputs:
        for output in outputs:
            package_name = next((name for name, package in packages.items()
                                 if package.get('folder') == output), None)
            if package_name in disqualified:
                lines += [f'- `{Path(output).name}`: **DO NOT SUBMIT**; retained for audit only.']
            elif package_name in qualified:
                lines += [f'- [{Path(output).name}/predictions.zip](<{(Path(output)/"predictions.zip").as_posix()}>)']
            else:
                lines += [f'- `{Path(output).name}`: qualification unresolved; no upload recommendation.']
        lines += ['']
    else:
        lines += ['No new ZIP is currently listed as eligible for submission by this campaign.', '']
    for name, package in packages.items():
        lines += [f"`{name}`: **{package['status']}**; changes "
                  f"{package['differences_from_winner']} of 68 Rail predictions relative to the submitted ensemble.", '']
        if package.get('duplicate_of'):
            duplicate = package['duplicate_of']
            lines += ['Duplicate output is excluded from submission: ' +
                      (duplicate if isinstance(duplicate, str) else ', '.join(duplicate)), '']
        if package.get('unavailable_historical_archives'):
            lines += ['These historical archives were missing from disk during the original packaging check: '
                      + ', '.join(package['unavailable_historical_archives']), '']
        if package.get('reconstructed_history'):
            lines += ['The missing historical prediction vectors were reconstructed deterministically '
                      'and included in this distinctness check. Their reconstruction evidence and hashes '
                      'are recorded in the package metadata; the missing ZIP files therefore did not '
                      'exclude those prior predictions from comparison.', '']
    if history is not None:
        lines += ['`historical_prediction_reconstruction.json` records the later deterministic '
                  'reconstruction of missing prior predictions. A historical duplicate '
                  'overrides local validation qualification and any earlier packaging status.', '']
    if verification:
        lines += ['Packaging verification checks all four CSVs at the ZIP root, exact replay '
                  'of all 68 Rail files, byte-identical Door/ACV/SHM outputs, and unchanged '
                  'active deployment and successful-ensemble files. See each archive folder\'s '
                  '`verification.json` for evidence.', '']
    if branch_audit.exists():
        lines += ['## Related branch review', '',
                  f'The [Rail branch audit](<{branch_audit.as_posix()}>) records the inspected '
                  'remote commits, the feature provenance, and limitations in historical '
                  'validation protocols. It found no stronger validated branch model for '
                  'direct import. Historical branch scores are not matched estimates for '
                  'this campaign.', '']
    lines += ['## Limits on the conclusion', ''] + ['- ' + caveat for caveat in caveats] + ['']
    (RUN / 'REPORT.md').write_text('\n'.join(lines), encoding='utf-8')
    print(f'Report written: {RUN / "REPORT.md"}')
    print(f'Qualified: {qualified}; published archives: {len(outputs)}')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run', type=Path, default=RUN, help='Campaign results folder')
    main(parser.parse_args().run)
