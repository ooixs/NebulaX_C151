"""Package one fully qualified experiment without activating any model.

Run only after the architecture campaign has completed every robustness gate:
    python "Rail Corrugation/code/package_architectures.py" --backend architecture --name NAME
"""
from __future__ import annotations

import argparse
import hashlib
import inspect
import io
import json
import shutil
import sys
import uuid
import zipfile
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from joblib import Parallel, delayed

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from common.research import fingerprint, json_value
from rail_corrugation import experiment_architectures as campaign
from rail_corrugation import experiment_models as base
from rail_corrugation.decision import GRID, decode, legacy_decision, select
from rail_corrugation.pipeline import aggregate, file_channel_table, side_relative_rows
from rail_corrugation.predict_candidate import component_probability, predict_one

ARCH_RUN = campaign.RUN
RUN = ARCH_RUN
WINNER = campaign.previous.WINNER
BACKEND = 'architecture'
MEMBERS = {'door_predictions.csv', 'acv_predictions.csv',
           'rail_predictions.csv', 'shm_predictions.csv'}


def read_json(path):
    return json.loads(path.read_text(encoding='utf-8'))


def write_json(path, value):
    path.write_text(json.dumps(value, indent=2, default=json_value, allow_nan=False),
                    encoding='utf-8')


def set_backend(backend):
    """Select the campaign before validating its candidate name or qualification."""
    global campaign, RUN, WINNER, BACKEND
    if backend == 'followup':
        from rail_corrugation import experiment_followup
        experiment_followup.configure()
        campaign = experiment_followup
    elif backend == 'forest':
        from rail_corrugation import experiment_forest_search
        experiment_forest_search.configure()
        campaign = experiment_forest_search
    elif backend != 'architecture':
        raise ValueError(f'Unknown campaign backend: {backend}')
    RUN = campaign.RUN
    WINNER = campaign.previous.WINNER
    BACKEND = backend


def registered_config(component, plan):
    if component in campaign.CONFIGS:
        config = campaign.CONFIGS[component]
        if config != plan['configs'][component]:
            raise ValueError(f'Configuration changed after registration: {component}')
        return config
    if component in ['legacy_sg_rf', 'sg_rf']:
        return campaign.previous.CONFIGS[component]
    raise ValueError(f'Unregistered model component: {component}')


def canonical_hash(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, default=json_value,
                                     allow_nan=False).encode('utf-8')).hexdigest()


def frozen_json(path, value):
    """Create once; a changed calibration input requires an error, never a retry."""
    try:
        with path.open('x', encoding='utf-8') as stream:
            json.dump(value, stream, indent=2, default=json_value, allow_nan=False)
    except FileExistsError:
        if canonical_hash(read_json(path)) != canonical_hash(value):
            raise ValueError(f'Frozen calibration record changed: {path}')


def full_training_calibration(c, name, plan):
    """Repeat the validated inner selection on all training files, once.

    This selection precedes test extraction. Its score is used to calibrate the
    final model and must never be described as independent validation evidence.
    """
    indices = np.arange(c.n)
    parts = campaign.previous.grouped_inner(indices, c.groups)
    if len(parts) != 3 or sorted(np.concatenate([te for _, te in parts])) != indices.tolist():
        raise ValueError('Final calibration requires an exhaustive three-fold partition')
    folds = []
    for j, (tr, te) in enumerate(parts):
        if set(tr).intersection(te) or set(tr).union(te) != set(indices):
            raise ValueError('Final calibration folds overlap or omit training files')
        if set(c.groups[tr]).intersection(c.groups[te]):
            raise ValueError('Duplicate group crossed a final calibration fold')
        folds.append(dict(fold=j, train=tr, validation=te, seed=10000+j))

    configs = {}
    mixes = {}

    def register(node):
        if node == 'ensemble':
            mixes[node] = ['legacy_sg_rf', 'sg_rf']
            register('legacy_sg_rf')
            register('sg_rf')
        elif node in campaign.MIXES:
            pair = list(campaign.MIXES[node])
            if pair != list(plan['mixes'][node]):
                raise ValueError(f'Mixture changed after registration: {node}')
            mixes[node] = pair
            for child in pair:
                register(child)
        else:
            configs[node] = registered_config(node, plan)

    register(name.removeprefix('blend__'))
    if name.startswith('blend__'):
        register('ensemble')
    if list(plan['blend_grid']) != [.25, .5, .75]:
        raise ValueError('Final calibration requires the validated blend grid')
    source_files = {Path(__file__), ROOT/'common/metrics.py'}
    source_files.update(Path(inspect.getfile(cls)) for cls in type(c).__mro__ if cls is not object)
    source_files.update(Path(__file__).with_name(n) for n in [
        'decision.py', 'pipeline.py', 'train.py', 'representations.py', 'sg_features.py',
        'paired_models.py', 'predict_candidate.py'])
    sources = {str(p.relative_to(ROOT)): fingerprint(p) for p in sorted(source_files)}
    inputs = dict(method='full_train_threefold_oof_v1', name=name, folds=folds,
                  configs=configs, mixes=mixes, fixed_mixture_weight=.75,
                  blend_grid=plan['blend_grid'], global_threshold_grid=GRID,
                  source_hashes=sources, plan_sha256=fingerprint(RUN/'plan.json'),
                  labels_sha256=fingerprint(base.DATA/'Train_Labels.csv'),
                  filenames=c.lab.filename.tolist(), labels=c.y, groups=c.groups,
                  fit_source_sha256=hashlib.sha256(inspect.getsource(c.fit_predict).encode()).hexdigest())
    signature = canonical_hash(inputs)
    path = RUN/f'final_calibration_{name}.json'
    registration_path = RUN/f'final_calibration_{name}_registration.json'
    frozen_json(registration_path, dict(input_signature=signature, inputs=inputs))
    if path.exists():
        cached = read_json(path)
        if cached.get('input_signature') != signature or canonical_hash(cached['inputs']) != signature:
            raise ValueError(f'Final calibration inputs/sources changed; refusing to recalibrate {name}')
        record_hash = cached['record_sha256']
        if canonical_hash({k: v for k, v in cached.items() if k != 'record_sha256'}) != record_hash:
            raise ValueError(f'Frozen final calibration contents changed: {name}')
        for component in configs:
            probability = np.asarray(cached['probabilities'][component])
            if probability.shape != (c.n, 2) or not np.isfinite(probability).all():
                raise ValueError(f'Invalid frozen calibration probabilities: {component}')
        print(f'Reusing frozen final calibration for {name}', flush=True)
        return cached

    probabilities = {}

    def oof(node):
        if node not in probabilities:
            if node in mixes:
                first, second = mixes[node]
                probability = .75*oof(first) + .25*oof(second)
            else:
                probability = np.full((c.n, 2), np.nan)
                for fold in folds:
                    tr, te = fold['train'], fold['validation']
                    probability[te], _ = c.fit_predict(configs[node], tr, te, fold['seed'])
                    print(f'Final calibration {node}: fold {fold["fold"]+1}/3', flush=True)
            if probability.shape != (c.n, 2) or not np.isfinite(probability).all():
                raise ValueError(f'Invalid calibration probabilities: {node}')
            probabilities[node] = probability
        return probabilities[node]

    candidate = oof(name.removeprefix('blend__'))
    weight = None
    if name.startswith('blend__'):
        incumbent = oof('ensemble')
        best = None
        for value in plan['blend_grid']:
            mixed = value*candidate + (1-value)*incumbent
            decision, score = select('global', c.y, mixed)
            # Identical tie policy to Campaign.evaluate: keep the earliest weight.
            if best is None or score > best[0]+1e-12:
                best = score, value, decision, mixed
        score, weight, decision, chosen_probability = best
    else:
        decision, score = select('global', c.y, candidate)
        chosen_probability = candidate
    for filename, digest in sources.items():
        if fingerprint(ROOT/filename) != digest:
            raise ValueError(f'Calibration source changed while fitting: {filename}')
    result = dict(method='full_train_threefold_oof_v1', input_signature=signature, inputs=inputs,
                  decision=decision, weight=weight, probabilities=probabilities,
                  selected_probabilities=chosen_probability, calibration_macro_f1=score,
                  score_context='Calibration score on the same OOF labels used for parameter selection; not independent validation.',
                  policy='Frozen once before test feature extraction; do not try alternate values to obtain distinct predictions.')
    result['record_sha256'] = canonical_hash(result)
    frozen_json(path, result)
    return result


def protected_files():
    """Capture the active deployment and submitted winner before any fitting."""
    hashes = {}
    for subsystem in ['Door', 'ACV', 'Rail Corrugation', 'SHM']:
        manifest_path = ROOT / subsystem / 'model/active_model.json'
        manifest = read_json(manifest_path)
        model_path = manifest_path.parent / manifest['file']
        if fingerprint(model_path) != manifest['sha256']:
            raise ValueError(f'Active model does not match its manifest: {subsystem}')
        for path in [manifest_path, model_path]:
            hashes[str(path)] = fingerprint(path)
    for path in [ROOT/'predictions.zip', *sorted((ROOT/'predictions').glob('*.csv')),
                 WINNER/'predictions.zip', WINNER/'rail_candidate.joblib',
                 WINNER/'submission.json', *[WINNER/n for n in sorted(MEMBERS)]]:
        if path.exists():
            hashes[str(path)] = fingerprint(path)
    return hashes


def verify_protected(hashes):
    for filename, expected in hashes.items():
        if fingerprint(Path(filename)) != expected:
            raise ValueError(f'Protected deployment/submission changed: {filename}')


def archive_members(path):
    with zipfile.ZipFile(path) as archive:
        if len(archive.namelist()) != 4 or set(archive.namelist()) != MEMBERS:
            raise ValueError(f'Expected the four CSV files at archive root: {path}')
        if archive.testzip() is not None:
            raise ValueError(f'Corrupt archive: {path}')
        return {name: archive.read(name) for name in sorted(MEMBERS)}


def label_vector(data, filenames):
    table = pd.read_csv(io.BytesIO(data))
    if list(table.columns) != ['file_id', 'prediction']:
        raise ValueError('Unexpected Rail prediction columns')
    if len(table) != 68 or not table.file_id.is_unique or set(table.file_id) != set(filenames):
        raise ValueError('Expected exactly the 68 test file IDs')
    if not set(table.prediction) <= {'Normal', 'Side I', 'Side II'}:
        raise ValueError('Unexpected Rail label')
    return tuple(table.set_index('file_id').loc[filenames, 'prediction'])


def reconstruction_paths():
    """Retain architecture-round history when a later campaign is selected."""
    paths = {ARCH_RUN/'historical_prediction_reconstruction.json',
             RUN/'historical_prediction_reconstruction.json'}
    return sorted(path for path in paths if path.exists())


def available_vectors(filenames):
    """Check surviving archives and reconstructed vectors from deleted archives."""
    paths = set((ROOT/'weights/submissions').glob('*/predictions.zip'))
    missing = set()
    for output_file in (ROOT/'weights/Rail Corrugation/runs').glob('*/outputs.json'):
        outputs = read_json(output_file)
        if not isinstance(outputs, list):
            continue
        for folder in outputs:
            if not isinstance(folder, str):
                continue
            path = Path(folder)/'predictions.zip'
            if path.exists():
                paths.add(path)
            else:
                missing.add(str(path))
    vectors = {}
    for path in sorted(paths):
        vectors[str(path)] = label_vector(archive_members(path)['rail_predictions.csv'], filenames)
    for path in reconstruction_paths():
        for name, value in read_json(path).items():
            labels = value['labels']
            if not isinstance(labels, list) or len(labels) != 68:
                raise ValueError(f'Expected 68 reconstructed labels: {path} / {name}')
            # Reconstruction labels are explicitly ordered Test1.csv ... Test68.csv.
            data = pd.DataFrame(dict(file_id=[f'Test{i}.csv' for i in range(1, 69)],
                                     prediction=labels)).to_csv(index=False).encode('utf-8')
            vectors[f'{path}#{name}'] = label_vector(data, filenames)
    return vectors, sorted(missing)


def snapshot_sources(destination):
    """Keep package import paths intact so the snapshot can run independently."""
    hashes = {}
    for directory in [ROOT/'Rail Corrugation/code', ROOT/'rail_corrugation', ROOT/'common']:
        for source in sorted(directory.rglob('*.py')):
            relative = source.relative_to(ROOT)
            target = destination/'source'/relative
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(source, target)
            hashes[relative.as_posix()] = fingerprint(target)
    return hashes


def package(name, jobs=8):
    # This must precede construction of Campaign, extraction, or any full-data fit.
    if any((ARCH_RUN/f'rejected-{name}').glob('DO_NOT_SUBMIT*')):
        raise ValueError(f'{name!r} is an archived rejected submission and must not be retried')
    qualifications = read_json(RUN/'qualification.json')
    qualified = qualifications.get(name)
    if name not in campaign.CANDIDATES or not qualified or qualified.get('qualified') is not True:
        raise ValueError(f'{name!r} has not passed all registered validation gates')
    if not qualified.get('gates') or not all(v is True for v in qualified['gates'].values()):
        raise ValueError(f'{name!r} contains an unsuccessful validation gate')
    validation = {tag: read_json(RUN/f'{tag}_{name}.json')
                  for tag in ['screen', 'confirm', 'strat', 'seed_1000', 'seed_2000', 'speed']}
    plan = read_json(RUN/'plan.json')
    calibration_method = plan.get('final_calibration')
    if calibration_method not in [None, 'full_train_threefold_oof_v1']:
        raise ValueError(f'Unknown final calibration method: {calibration_method}')
    destination = ROOT/'weights/submissions'/f'2026-09-19-{name}-{BACKEND}'
    if destination.exists():
        raise FileExistsError(f'Refusing to overwrite an existing artifact: {destination}')
    hashes = protected_files()
    try:
        winner_manifest = read_json(WINNER/'submission.json')
        winner_sha = fingerprint(WINNER/'predictions.zip')
        if winner_sha != winner_manifest['zip_sha256'] or winner_sha != plan['winner_zip_sha256']:
            raise ValueError('Successful submitted ensemble archive has changed')
        if fingerprint(WINNER/'rail_candidate.joblib') != winner_manifest['model_sha256']:
            raise ValueError('Successful submitted ensemble model has changed')
        winner_members = archive_members(WINNER/'predictions.zip')
        for member, contents in winner_members.items():
            if hashlib.sha256(contents).hexdigest() != winner_manifest['members'][member]:
                raise ValueError(f'Successful ensemble member hash mismatch: {member}')
            if (WINNER/member).read_bytes() != contents:
                raise ValueError(f'Successful ensemble CSV and archive disagree: {member}')

        files = sorted((base.DATA/'Test').glob('*.csv'),
                       key=lambda p: int(''.join(filter(str.isdigit, p.stem))))
        filenames = [p.name for p in files]
        if len(files) != 68 or set(filenames) != {f'Test{i}.csv' for i in range(1, 69)}:
            raise ValueError('Expected Test1.csv through Test68.csv')
        winner_labels = label_vector(winner_members['rail_predictions.csv'], filenames)
        vectors, unavailable = available_vectors(filenames)
        c = campaign.Campaign(jobs=jobs)
        calibration = None
        if calibration_method == 'full_train_threefold_oof_v1':
            calibration = full_training_calibration(c, name, plan)
            for suffix in ['', '_registration']:
                path = RUN/f'final_calibration_{name}{suffix}.json'
                hashes[str(path)] = fingerprint(path)
        saved_run = base.RUN
        try:
            base.RUN = campaign.previous.OLD
            testdata = base.local_data(files, 'test', jobs)
        finally:
            base.RUN = saved_run
        if hasattr(c, 'augment_testdata'):
            testdata = c.augment_testdata(testdata, files)
        all_train = np.arange(c.n)
        test_indices = np.arange(len(files))
        ref, cols, _ = c.table(all_train)
        tables = Parallel(n_jobs=min(jobs, 4))(delayed(file_channel_table)(path) for path in files)
        testlegacy = pd.DataFrame([
            row for ct, speed in tables
            for row in side_relative_rows(aggregate(ref.excess(ct, speed), speed, paired=True),
                                           engineered=True)
        ])[cols].fillna(-9).to_numpy()
        predictions, artifacts = {}, {}

        def final(component):
            if component in predictions:
                return predictions[component], artifacts[component]
            if component in ['ensemble', 'legacy_sg_rf', 'sg_rf']:
                artifact = joblib.load(WINNER/'rail_candidate.joblib')
                if component != 'ensemble':
                    artifact = artifact['components'][0 if component == 'legacy_sg_rf' else 1]
                    if artifact['config'] != registered_config(component, plan):
                        raise ValueError(f'Saved winner component configuration differs: {component}')
                values = []
                for i, (ct, speed) in enumerate(tables):
                    rows = slice(2*i, 2*i+2)
                    data = (pd.DataFrame(testdata['sg'][rows]),
                            pd.DataFrame(testdata['crosscar'][rows]), testdata['raw'][rows])
                    if 'sg_pooled' in testdata:
                        data += (pd.DataFrame(testdata['sg_pooled'][rows]),)
                    values.append(component_probability(artifact, data, ct, speed))
                probability = np.asarray(values)
                if component == 'ensemble':
                    reproduced = decode(probability[:, 0], probability[:, 1], artifact['decision'])
                    if tuple(reproduced) != winner_labels:
                        raise ValueError('Saved winning model no longer reproduces its submitted labels')
            elif component in campaign.MIXES:
                if list(campaign.MIXES[component]) != list(plan['mixes'][component]):
                    raise ValueError(f'Mixture changed after registration: {component}')
                first, second = campaign.MIXES[component]
                pa, aa = final(first)
                pb, ab = final(second)
                probability = .75*pa + .25*pb
                artifact = dict(components=[aa, ab], weight=.75, name=component)
            else:
                config = registered_config(component, plan)
                probability, artifact = c.fit_predict(config, all_train, test_indices, 0, True,
                                                     testdata, testlegacy)
            if np.asarray(probability).shape != (68, 2) or not np.isfinite(probability).all():
                raise ValueError(f'Invalid probability output: {component}')
            predictions[component], artifacts[component] = probability, artifact
            return probability, artifact

        decisions = validation['screen']['decisions']
        if calibration is None:
            if not decisions or any(d['decision']['rule'] != 'global' for d in decisions):
                raise ValueError('Expected screening inner-fold global threshold selections')
            decision = legacy_decision(float(np.median([d['decision']['tau'] for d in decisions])))
        else:
            decision = calibration['decision']
        if name.startswith('blend__'):
            component = name.removeprefix('blend__')
            pa, aa = final(component)
            pb, ab = final('ensemble')
            weight = calibration['weight'] if calibration is not None else float(np.median([d['weight'] for d in decisions]))
            if weight not in plan['blend_grid']:
                raise ValueError('Median blend weight was outside the registered grid')
            probability = weight*pa + (1-weight)*pb
            artifact = dict(components=[aa, ab], weight=weight)
        else:
            probability, artifact = final(name)
        calibration_metadata = dict(method='median_screening_inner_selections') if calibration is None else dict(
            method=calibration['method'], input_signature=calibration['input_signature'],
            record_sha256=calibration['record_sha256'],
            decision=calibration['decision'], weight=calibration['weight'],
            calibration_macro_f1=calibration['calibration_macro_f1'], score_context=calibration['score_context'])
        artifact.update(decision=decision, name=name, run=str(RUN), full_training_seed=0,
                        final_calibration=calibration_metadata)
        labels = decode(probability[:, 0], probability[:, 1], decision)
        vector = tuple(labels)
        duplicates = [path for path, prior in vectors.items() if prior == vector]
        comparison = dict(name=name, backend=BACKEND, decision=decision, validation=qualified,
                          final_calibration=calibration_metadata,
                          differences_from_winner=sum(a != b for a, b in zip(vector, winner_labels)),
                          checked_prediction_sources=sorted(vectors),
                          checked_archives=sorted(p for p in vectors if '#' not in p),
                          reconstructed_history={str(p): fingerprint(p) for p in reconstruction_paths()},
                          unavailable_historical_archives=unavailable)
        if duplicates:
            comparison.update(status='qualified-but-duplicate', duplicate_of=duplicates)
            write_json(RUN/f'package_{name}.json', comparison)
            print(json.dumps(comparison, default=json_value), flush=True)
            return None

        # Keep incomplete archives out of the published submission folder/output list.
        staging = destination.parent/f'.{destination.name}.pending-{uuid.uuid4().hex[:10]}'
        staging.mkdir(parents=True)
        members = dict(winner_members)
        members['rail_predictions.csv'] = pd.DataFrame(dict(file_id=filenames, prediction=labels)).to_csv(
            index=False, lineterminator='\n').encode('utf-8')
        joblib.dump(artifact, staging/'rail_candidate.joblib', compress=3)
        replay_artifact = joblib.load(staging/'rail_candidate.joblib')
        replay = Parallel(n_jobs=4)(delayed(predict_one)(path, replay_artifact) for path in files)
        replay_bytes = pd.DataFrame(dict(file_id=filenames, prediction=replay)).to_csv(
            index=False, lineterminator='\n').encode('utf-8')
        if tuple(replay) != vector or replay_bytes != members['rail_predictions.csv']:
            raise ValueError(f'Saved candidate failed full raw-file replay; inspect {staging}')
        for member, contents in members.items():
            (staging/member).write_bytes(contents)
        with zipfile.ZipFile(staging/'predictions.zip', 'w', zipfile.ZIP_DEFLATED) as archive:
            for member, contents in members.items():
                archive.writestr(member, contents)
        verified = archive_members(staging/'predictions.zip')
        for member, contents in verified.items():
            if contents != members[member] or contents != (staging/member).read_bytes():
                raise ValueError(f'Archive/CSV byte mismatch: {member}')
            if member != 'rail_predictions.csv' and contents != winner_members[member]:
                raise ValueError(f'Non-Rail output changed: {member}')
        source_hashes = snapshot_sources(staging)
        evidence = staging/'validation'
        evidence.mkdir()
        for path in sorted(RUN.glob('*.json')):
            if path.name in ['plan.json', 'qualification.json', 'data_verification.json'] or \
                    path.name.endswith('_protocol.json') or path.name.endswith(f'_{name}.json') or \
                    path.name.endswith('_ensemble.json') or path.name == f'final_calibration_{name}_registration.json':
                shutil.copy2(path, evidence/path.name)
        for path in sorted(RUN.glob(f'*_{name}_predictions.npz')):
            shutil.copy2(path, evidence/path.name)
        for i, path in enumerate(reconstruction_paths()):
            shutil.copy2(path, evidence/f'historical_prediction_reconstruction_{i}.json')
        verification = dict(raw_recordings_replayed=68, model_replay_exact=True,
                            zip_members_verified=True, other_subsystems_byte_identical=True,
                            protected_files=hashes)
        manifest = dict(name=name, backend=BACKEND, validation=qualified, validation_results=validation,
                        decision=decision, full_training_seed=0, final_calibration=calibration_metadata,
                        zip_sha256=fingerprint(staging/'predictions.zip'),
                        model_sha256=fingerprint(staging/'rail_candidate.joblib'),
                        members={n: hashlib.sha256(d).hexdigest() for n, d in members.items()},
                        label_counts=pd.Series(labels).value_counts().to_dict(),
                        winner_zip_sha256=winner_sha, comparison=comparison, sources=source_hashes,
                        caveat='Repeated local validation only; hidden improvement is unproven. Candidate is not activated.')
        write_json(staging/'submission.json', manifest)
        write_json(staging/'verification.json', verification)
        source_script = destination/'source/Rail Corrugation/code/predict_candidate.py'
        calibration_description = (
            'Its threshold and any learned blend weight were selected once from duplicate-safe '
            'three-fold out-of-fold predictions on all training recordings, before test feature '
            'extraction. The selection uses seeds 10000, 10001 and 10002 and matches the inner '
            'procedure evaluated by nested validation. These calibration scores are not independent '
            'validation scores. The choices are frozen and were not retried for distinct predictions. '
            if calibration is not None else
            'Its threshold and any learned blend weight are medians of screening inner-fold selections. '
        )
        (staging/'README.md').write_text(
            f'# {name}\n\nQualified local experiment; no hidden score is available.\n\n'
            f'The candidate changes {comparison["differences_from_winner"]} of 68 Rail labels '
            'relative to the submitted ensemble. The ZIP contains four CSV files at its root; '
            'Door, ACV and SHM are byte-identical to that ensemble. All 68 Rail outputs were '
            'reproduced from raw recordings using the saved model.\n\n'
            'The model uses full-training seed 0. ' + calibration_description +
            'Validation reuses existing recordings '
            'and does not guarantee leaderboard improvement.\n\n'
            'Reproduce with the included source snapshot from the repository:\n\n'
            f'```powershell\n.venv/Scripts/python.exe "{source_script}" '
            f'--model "{destination / "rail_candidate.joblib"}" '
            f'--input "{base.DATA / "Test"}" --output "{destination / "reproduced_rail.csv"}"\n```\n',
            encoding='utf-8')
        verify_protected(hashes)
        staging.rename(destination)
        output_path = RUN/'outputs.json'
        outputs = read_json(output_path) if output_path.exists() else []
        if not isinstance(outputs, list):
            raise ValueError('Existing outputs.json is not a list')
        if str(destination) not in outputs:
            outputs.append(str(destination))
            write_json(output_path, outputs)
        comparison.update(status='packaged-and-verified', folder=str(destination))
        write_json(RUN/f'package_{name}.json', comparison)
        print(f'PACKAGED AND REPLAYED {destination}', flush=True)
        return destination
    finally:
        verify_protected(hashes)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--backend', choices=['architecture', 'followup', 'forest'], default='architecture')
    parser.add_argument('--name', required=True)
    parser.add_argument('--jobs', type=int, default=8)
    args = parser.parse_args()
    if args.jobs < 1:
        parser.error('--jobs must be positive')
    set_backend(args.backend)
    if args.name not in campaign.CANDIDATES:
        parser.error(f'--name must be one of {", ".join(campaign.CANDIDATES)} for {args.backend}')
    package(args.name, args.jobs)


if __name__ == '__main__':
    main()
