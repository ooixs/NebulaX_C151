"""Leakage-controlled Rail model experiments, resumable and isolated from active models.

Run --stage screen, then --stage confirm --names comma,separated,shortlist,
then --stage package. Only confirmation-qualified, distinct outputs get ZIPs.
"""
from __future__ import annotations

import argparse
import hashlib
import io
import json
import shutil
import sys
import zipfile
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from joblib import Parallel, delayed
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from common.metrics import rail_macro_f1
from common.research import fingerprint, json_value, nested_folds
from rail_corrugation.decision import block_protocol, decode, legacy_decision, select
from rail_corrugation.pipeline import CLASSES
from rail_corrugation.representations import extract, normalized_raw
from rail_corrugation.train import research_columns, research_model, research_table

DATA = ROOT / 'data/Rail_Corrugation'
RUN = ROOT / 'weights/Rail Corrugation/runs/2026-09-19-model-experiments'
ARCHIVE = ROOT / 'weights/submissions/2026-09-18-submission1-best/predictions.zip'
CONFIGS = {
    'incumbent': dict(rep='legacy', fixed=True),
    'sg_et': dict(rep='sg'),
    'sg_rf': dict(rep='sg', kind='rf'),
    'sg_lgbm': dict(rep='sg', kind='lgbm'),
    'legacy_sg': dict(rep='legacy_sg'),
    'legacy_crosscar': dict(rep='legacy_crosscar'),
    'crosscar_et': dict(rep='crosscar'),
    'sg_crosscar': dict(rep='sg_crosscar'),
    'legacy_sg_crosscar': dict(rep='legacy_sg_crosscar'),
    'minirocket': dict(rep='rocket', kernels=1680, C=0.1),
    'minirocket_normalized': dict(rep='rocket', kernels=1680, C=0.1, normalize=True),
    'legacy_rf': dict(rep='legacy', kind='rf'),
    'sg_crosscar_rf': dict(rep='sg_crosscar', kind='rf'),
    'legacy_sg_rf': dict(rep='legacy_sg', kind='rf'),
    'legacy_crosscar_rf': dict(rep='legacy_crosscar', kind='rf'),
    'sg_rf_leaf2': dict(rep='sg', kind='rf', leaf=2),
    'sg_rf_mf03': dict(rep='sg', kind='rf', max_features=0.3),
    'sg_rf_mf05': dict(rep='sg', kind='rf', max_features=0.5),
    'sg_et_mf03': dict(rep='sg', max_features=0.3),
    'sg_rf_unweighted': dict(rep='sg', kind='rf', class_weight=None),
    'legacy_sg_lgbm': dict(rep='legacy_sg', kind='lgbm'),
    'sg_svc': dict(rep='sg', kind='svc', C=1.0),
}


def save(name, data):
    (RUN / name).write_text(json.dumps(data, indent=2, default=json_value, allow_nan=False), encoding='utf-8')


def rows(indices):
    return (2*np.asarray(indices)[:, None] + np.arange(2)).ravel()


def prepare_reference(tr, ct, v, y):
    keyhash = hashlib.sha256(np.asarray(tr,dtype=np.int64).tobytes()).hexdigest()[:16]
    path = RUN/f'tables/{keyhash}.joblib'
    if not path.exists():
        ref,table = research_table(ct,v,y,tr)
        cols = research_columns(table.columns,'legacy')
        joblib.dump((ref,cols,table[cols].to_numpy()),path)


def local_data(files, name, jobs):
    path = RUN / f'{name}_representations.joblib'
    signature = [(p.name, p.stat().st_size, p.stat().st_mtime_ns) for p in files]
    source = fingerprint(Path(__file__).with_name('representations.py'))
    if path.exists():
        cached = joblib.load(path)
        assert cached['signature'] == signature and cached['source'] == source
        return cached['data']
    print(f'Extracting {len(files)} {name} recordings ...', flush=True)
    data = Parallel(n_jobs=jobs)(delayed(extract)(p) for p in files)
    result = dict(sg=pd.concat([x[0] for x in data], ignore_index=True).to_numpy(),
                  crosscar=pd.concat([x[1] for x in data], ignore_index=True).to_numpy(),
                  raw=np.concatenate([x[2] for x in data]))
    joblib.dump(dict(signature=signature, source=source, data=result), path)
    return result


class Campaign:
    def __init__(self, jobs):
        self.jobs = jobs
        self.lab = pd.read_csv(DATA / 'Train_Labels.csv')
        self.y = self.lab.label.to_numpy()
        self.sy = (self.y[:, None] == np.array(['Side I', 'Side II'])).astype(int).ravel()
        self.n = len(self.y)
        self.data = local_data([DATA/'Train'/x for x in self.lab.filename], 'train', jobs)
        cached = joblib.load(ROOT/'weights/Rail Corrugation/train_channel_tables.joblib')
        self.ct, self.v = [x[0] for x in cached], [x[1] for x in cached]
        self.tables = {}
        self.numbers = self.lab.filename.str.extract(r'(\d+)', expand=False).astype(int).to_numpy()

    def table(self, tr):
        key = tuple(tr)
        if key not in self.tables:
            keyhash = hashlib.sha256(np.asarray(tr, dtype=np.int64).tobytes()).hexdigest()[:16]
            path = RUN/f'tables/{keyhash}.joblib'
            if path.exists():
                value = joblib.load(path)
            else:
                ref, table = research_table(self.ct, self.v, self.y, tr)
                cols = research_columns(table.columns, 'legacy')
                value = (ref, cols, table[cols].to_numpy())
                path.parent.mkdir(exist_ok=True)
                joblib.dump(value, path)
            self.tables[key] = value
        return self.tables[key]

    def design(self, rep, tr, data=None, legacy=None):
        data = self.data if data is None else data
        parts = []
        if 'legacy' in rep:
            parts.append(self.table(tr)[2] if legacy is None else legacy)
        if 'sg' in rep:
            parts.append(data['sg'])
        if 'crosscar' in rep:
            parts.append(data['crosscar'])
        return np.nan_to_num(np.concatenate(parts, axis=1), nan=-9, posinf=1e6, neginf=-1e6)

    def fit_predict(self, cfg, tr, te, seed, final=False, testdata=None, testlegacy=None):
        ri, ti = rows(tr), rows(te)
        if cfg['rep'] == 'rocket':
            from sktime.transformations.panel.rocket import MiniRocketMultivariate
            raw = self.data['raw']
            other = raw if testdata is None else testdata['raw']
            if cfg.get('normalize'):
                raw, other = normalized_raw(raw), normalized_raw(other)
            transform = MiniRocketMultivariate(num_kernels=cfg['kernels'], n_jobs=self.jobs, random_state=seed)
            xtr = transform.fit_transform(raw[ri]).to_numpy()
            xte = transform.transform(other[ti]).to_numpy()
            model = make_pipeline(StandardScaler(), LogisticRegression(C=cfg['C'], class_weight='balanced',
                                                                       max_iter=1500, solver='liblinear', random_state=seed))
            model.fit(xtr, self.sy[ri])
            p = model.predict_proba(xte)[:, 1]
            artifact = dict(transform=transform, model=model, config=cfg)
        else:
            x = self.design(cfg['rep'], tr)
            xt = x if testdata is None else self.design(cfg['rep'], tr, testdata, testlegacy)
            model = research_model(cfg, seed, self.jobs).fit(x[ri], self.sy[ri])
            p = model.predict_proba(xt[ti])[:, list(model.classes_).index(1)]
            artifact = dict(model=model, config=cfg)
            if 'legacy' in cfg['rep']:
                artifact.update(ref=self.table(tr)[0], cols=self.table(tr)[1])
        return p.reshape(-1, 2), artifact if final else None

    def probabilities(self, name, protocol, tag):
        path = RUN / f'{tag}_{name}_probabilities.joblib'
        if path.exists():
            return joblib.load(path)
        if name.startswith('blend__'):
            a, b = name[len('blend__'):].split('__')
            aa, bb = self.probabilities(a, protocol, tag), self.probabilities(b, protocol, tag)
            return [(x, z) for x, z in zip(aa, bb)]
        cfg = CONFIGS[name]
        result = []
        for f in protocol:
            k, tr, te = f['fold'], f['train'], f['validation']
            outer, _ = self.fit_predict(cfg, tr, te, k)
            inner = np.full((self.n, 2), np.nan)
            # Baseline does not select anything; inner scores are still needed for blends.
            for j, (itr, ite) in enumerate(f['inner']):
                inner[ite], _ = self.fit_predict(cfg, itr, ite, 10000+10*k+j)
            assert np.isfinite(inner[tr]).all()
            result.append(dict(outer=outer, inner=inner[tr]))
            print(f'{tag} {name}: fold {k+1}/{len(protocol)}', flush=True)
        joblib.dump(result, path)
        return result

    def evaluate(self, name, protocol, tag):
        path = RUN/f'{tag}_{name}.json'
        if path.exists():
            return json.loads(path.read_text())
        probs = self.probabilities(name, protocol, tag)
        repeats = max(f['repeat'] for f in protocol)+1
        pred = np.full((repeats, self.n), '', dtype='<U7')
        oof = np.full((repeats, self.n, 2), np.nan)
        decisions = []
        for f, p in zip(protocol, probs):
            tr, te, repeat = f['train'], f['validation'], f['repeat']
            if name.startswith('blend__'):
                best = None
                for weight in (0.25, 0.5, 0.75):
                    inner = weight*p[0]['inner']+(1-weight)*p[1]['inner']
                    decision, score = select('global', self.y[tr], inner)
                    if best is None or score > best[0]:
                        best = (score, weight, decision)
                _, weight, decision = best
                outer = weight*p[0]['outer']+(1-weight)*p[1]['outer']
            else:
                weight = None
                decision = legacy_decision(.25) if name == 'incumbent' else select('global', self.y[tr], p['inner'])[0]
                outer = p['outer']
            pred[repeat, te] = decode(outer[:, 0], outer[:, 1], decision)
            oof[repeat, te] = outer
            decisions.append(dict(fold=f['fold'], decision=decision, weight=weight))
        assert (pred != '').all() and np.isfinite(oof).all()
        scores = [rail_macro_f1(self.y, p)['macro_f1'] for p in pred]
        result = dict(name=name, scores=scores, mean=float(np.mean(scores)), std=float(np.std(scores)),
                      per_class={c: float(np.mean([rail_macro_f1(self.y, p)['per_class'][c] for p in pred])) for c in CLASSES},
                      classification_reports=[classification_report(self.y, p, labels=CLASSES, output_dict=True, zero_division=0) for p in pred],
                      confusion=[confusion_matrix(self.y, p, labels=CLASSES).tolist() for p in pred], decisions=decisions)
        save(path.name, result)
        np.savez_compressed(RUN/f'{tag}_{name}_oof.npz', probabilities=oof, prediction=pred, labels=self.y.astype('U7'))
        print(f"RESULT {tag} {name}: {result['mean']:.6f} +/- {result['std']:.6f} {result['per_class']}", flush=True)
        return result


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--stage', choices=['screen', 'prime', 'confirm', 'package'], default='screen')
    ap.add_argument('--names')
    ap.add_argument('--jobs', type=int, default=8)
    args = ap.parse_args()
    RUN.mkdir(parents=True, exist_ok=True)
    if not (RUN/'plan.json').exists():
        save('plan.json', dict(configs=CONFIGS, screening_offsets=[0,27,54], confirmation_offsets=[14,41,68],
            acceptance='Screen gain > max(0.01, baseline repeat std), all 3 repeats improve; confirmation gain > 0.01, all repeats improve; stratified regression <= 0.02; neither fault-class F1 drops > 0.03 on confirmation.',
            caveat='Rotations reuse recordings: stability checks, not independent test estimates. Repeated model search creates selection optimism.',
            physics='No documented car spacing; no delay-aligned correlation or assertion of shared track overlap. Cross-car features measure spectral consistency only.',
            normalization='Per-recording only; no adaptation to the hidden test batch.',
            archive_sha256=fingerprint(ARCHIVE), labels_sha256=fingerprint(DATA/'Train_Labels.csv')))
    c = Campaign(args.jobs)
    if args.stage == 'screen':
        protocol = block_protocol(c.numbers, 5, 3, (0,27,54))
        if not (RUN/'screen_protocol.json').exists():
            save('screen_protocol.json', protocol)
        names = args.names.split(',') if args.names else list(CONFIGS)
        save('registered_candidates.json', CONFIGS)
        for name in names:
            c.evaluate(name, protocol, 'screen')
    elif args.stage == 'prime':
        protocols = block_protocol(c.numbers,5,3,(14,41,68)) + nested_folds(c.y,folds=5,repeats=1,inner_folds=3,seed=20260919)
        pending = {}
        for fold in protocols:
            for tr in [fold['train']] + [a for a,b in fold['inner']]:
                pending[tuple(tr)] = tr
        print(f'Preparing {len(pending)} confirmation reference tables',flush=True)
        Parallel(n_jobs=args.jobs)(delayed(prepare_reference)(tr,c.ct,c.v,c.y) for tr in pending.values())
        print('Confirmation references ready',flush=True)
    elif args.stage == 'confirm':
        names = args.names.split(',')
        history_path = RUN/'confirmation_shortlists.json'
        history = json.loads(history_path.read_text()) if history_path.exists() else []
        if not history and (RUN/'confirmation_shortlist.json').exists():
            history.append(json.loads((RUN/'confirmation_shortlist.json').read_text()))
        if names not in history:
            history.append(names)
        save('confirmation_shortlists.json', history)
        if not (RUN/'confirmation_shortlist.json').exists():
            save('confirmation_shortlist.json', names)
        block = block_protocol(c.numbers, 5, 3, (14,41,68))
        strat = nested_folds(c.y, folds=5, repeats=1, inner_folds=3, seed=20260919)
        save('confirm_protocol.json', block)
        save('strat_protocol.json', strat)
        for name in ['incumbent'] + names:
            if name.startswith('blend__'):
                for component in name[len('blend__'):].split('__'):
                    c.evaluate(component, block, 'confirm')
                    c.evaluate(component, strat, 'strat')
            c.evaluate(name, block, 'confirm')
            c.evaluate(name, strat, 'strat')
        decisions = json.loads((RUN/'qualification.json').read_text()) if (RUN/'qualification.json').exists() else {}
        read = lambda tag, name: json.loads((RUN/f'{tag}_{name}.json').read_text())
        for name in names:
            s,b,t = [read(tag,name) for tag in ['screen','confirm','strat']]
            sb,bb,tb = [read(tag,'incumbent') for tag in ['screen','confirm','strat']]
            gates = dict(screen_gain=s['mean']-sb['mean'] > max(.01,sb['std']),
                screen_consistent=bool(np.all(np.array(s['scores']) > sb['scores'])),
                confirmation_gain=b['mean']-bb['mean'] > .01,
                confirmation_consistent=bool(np.all(np.array(b['scores']) > bb['scores'])),
                stratified_stability=t['mean'] >= tb['mean']-.02,
                fault_class_stability=all(b['per_class'][cl] >= bb['per_class'][cl]-.03 for cl in ['Side I','Side II']))
            if name.startswith('blend__'):
                components = name[len('blend__'):].split('__')
                gates['blend_beats_components_screen'] = s['mean'] > max(read('screen',x)['mean'] for x in components)+.005
                gates['blend_beats_components_confirmation'] = b['mean'] > max(read('confirm',x)['mean'] for x in components)+.005
            decisions[name] = dict(qualified=all(gates.values()), gates=gates,
                                   screen=s['mean'], confirmation=b['mean'], stratified=t['mean'])
        save('qualification.json', decisions)
        print(json.dumps(decisions, indent=2), flush=True)
    else:
        package(c)


def package(c):
    qualifications = json.loads((RUN/'qualification.json').read_text())
    names = [n for n,v in qualifications.items() if v['qualified']]
    names.sort(key=lambda n: qualifications[n]['confirmation'], reverse=True)
    files = sorted((DATA/'Test').glob('*.csv'), key=lambda p:int(''.join(filter(str.isdigit,p.stem))))
    testdata = local_data(files, 'test', c.jobs)
    from rail_corrugation.pipeline import file_channel_table, aggregate, side_relative_rows
    alltr = np.arange(c.n)
    ref, cols, _ = c.table(alltr)
    tables = [file_channel_table(p) for p in files]
    testlegacy = pd.DataFrame([r for ct,v in tables for r in side_relative_rows(aggregate(ref.excess(ct,v),v,paired=True),engineered=True)])[cols].fillna(-9).to_numpy()
    predictions, artifacts = {}, {}
    def final(name):
        if name not in predictions:
            p, art = c.fit_predict(CONFIGS[name], alltr, np.arange(len(files)), 0, True, testdata, testlegacy)
            predictions[name], artifacts[name] = p, art
        return predictions[name]
    with zipfile.ZipFile(ARCHIVE) as z:
        original = {n:z.read(n) for n in z.namelist()}
    assert set(original) == {'door_predictions.csv','acv_predictions.csv','rail_predictions.csv','shm_predictions.csv'}
    original_rail = pd.read_csv(io.BytesIO(original['rail_predictions.csv'])).set_index('file_id')
    seen = {tuple(original_rail.loc[[p.name for p in files], 'prediction'])}
    outputs, duplicates = [], []
    for name in names:
        screen = json.loads((RUN/f'screen_{name}.json').read_text())
        if name.startswith('blend__'):
            a,b = name[len('blend__'):].split('__')
            weight = float(np.median([d['weight'] for d in screen['decisions']]))
            p = weight*final(a)+(1-weight)*final(b)
            art = dict(components=[artifacts[a],artifacts[b]], weight=weight)
        else:
            p, art = final(name), artifacts[name]
        # Median inner-selected parameter, never tuned to the hidden test labels.
        decision = legacy_decision(float(np.median([d['decision']['tau'] for d in screen['decisions']])))
        label = decode(p[:,0],p[:,1],decision)
        csv = pd.DataFrame(dict(file_id=[p.name for p in files],prediction=label)).to_csv(index=False,lineterminator='\n').encode()
        signature = tuple(label)
        if signature in seen:
            duplicates.append(name)
            continue
        seen.add(signature)
        dest = ROOT/'weights/submissions'/f'2026-09-19-{name}'
        dest.mkdir(exist_ok=True)
        members = dict(original, **{'rail_predictions.csv':csv})
        with zipfile.ZipFile(dest/'predictions.zip','w',zipfile.ZIP_DEFLATED) as z:
            for filename, data in members.items():
                z.writestr(filename,data)
                (dest/filename).write_bytes(data)
        art.update(decision=decision, name=name, run=str(RUN))
        joblib.dump(art,dest/'rail_candidate.joblib',compress=3)
        with zipfile.ZipFile(dest/'predictions.zip') as z:
            assert set(z.namelist()) == set(original)
            for filename in original:
                if filename != 'rail_predictions.csv':
                    assert z.read(filename) == original[filename]
        manifest = dict(name=name, validation=qualifications[name], decision=decision,
                        zip_sha256=fingerprint(dest/'predictions.zip'), model_sha256=fingerprint(dest/'rail_candidate.joblib'),
                        members={n:hashlib.sha256(d).hexdigest() for n,d in members.items()},
                        label_counts=pd.Series(label).value_counts().to_dict(),
                        caveat='Local validation only; hidden score unknown. Candidate is not activated.',
                        sources={p.name:fingerprint(p) for p in Path(__file__).parent.glob('*.py')})
        (dest/'submission.json').write_text(json.dumps(manifest,indent=2,default=json_value),encoding='utf-8')
        source_dir = dest/'source'
        source_dir.mkdir(exist_ok=True)
        for source in Path(__file__).parent.glob('*.py'):
            shutil.copy2(source, source_dir/source.name)
        (dest/'README.md').write_text(
            f'# {name}\n\nLocal-validation candidate; no hidden score is available.\n\n'
            'The ZIP contains all four CSV files at its top level. Door, ACV and SHM are byte-identical to submission 1.\n\n'
            'Reproduce from the NebulaX repository with:\n\n'
            f'```powershell\n.venv/Scripts/python.exe "Rail Corrugation/code/predict_candidate.py" '
            f'--model "{dest.relative_to(ROOT).as_posix()}/rail_candidate.joblib" '
            f'--input data/Rail_Corrugation/Test --output "{dest.relative_to(ROOT).as_posix()}/reproduced_rail.csv"\n```\n', encoding='utf-8')
        outputs.append(str(dest))
        print(f'PACKAGED {dest}',flush=True)
        if len(outputs) == 3:
            break
    save('outputs.json',outputs)
    save('duplicate_candidates.json', duplicates)


if __name__ == '__main__':
    main()
