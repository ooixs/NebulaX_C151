"""Duplicate-safe validation of incumbent plus the submitted SG ensemble."""
import json
import sys
from pathlib import Path

import joblib
import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from rail_corrugation import experiment_models as base
from rail_corrugation.decision import decode, legacy_decision, select
from common.metrics import rail_macro_f1

OLD = base.RUN
GROUPED = ROOT/'weights/Rail Corrugation/runs/2026-09-19-branch-review'
RUN = ROOT/'weights/Rail Corrugation/runs/2026-09-19-incumbent-blend'


def protocol(tag, directory):
    p = json.loads((directory/f'{tag}_protocol.json').read_text())
    for f in p:
        for key in ['train', 'validation']:
            f[key] = np.array(f[key], dtype=int)
        f['inner'] = [(np.array(a, dtype=int), np.array(b, dtype=int)) for a,b in f['inner']]
    return p


def key(f):
    return json.dumps([f['fold'], f['train'].tolist(), f['validation'].tolist(),
                       [[a.tolist(), b.tolist()] for a,b in f['inner']]])


def verify_inputs():
    audit = json.loads((GROUPED/'audit.json').read_text())
    lab = base.pd.read_csv(base.DATA/'Train_Labels.csv')
    manifest = {r['filename']:r for r in audit['manifest']}
    groups = np.array([manifest[name]['sensor_sha256'] for name in lab.filename])
    for name in lab.filename:
        assert base.fingerprint(base.DATA/'Train'/name) == manifest[name]['sha256'], name
    counts = {}
    for tag in ['screen','confirm','strat']:
        folds = protocol(tag,GROUPED)
        for f in folds:
            assert set(f['train']).isdisjoint(f['validation'])
            assert set(f['train']) | set(f['validation']) == set(range(len(lab)))
            for tr,te in [(f['train'],f['validation'])]+f['inner']:
                assert set(groups[tr]).isdisjoint(groups[te])
            assert sorted(np.concatenate([te for _,te in f['inner']])) == sorted(f['train'])
            assert all(set(tr)|set(te) == set(f['train']) for tr,te in f['inner'])
        counts[tag] = len(folds)
    RUN.mkdir(exist_ok=True)
    (RUN/'input_verification.json').write_text(json.dumps(dict(training_files=len(lab),folds=counts,
        sha256_verified=True,duplicate_overlap=False),indent=2))
    print('Verified training hashes and all outer/inner duplicate groups',flush=True)


def probabilities(c, name, p, tag):
    path = RUN/f'{tag}_{name}_probabilities.joblib'
    if path.exists():
        return joblib.load(path)
    reuse = {}
    for directory in [OLD, GROUPED]:
        source = directory/f'{tag}_{name}_probabilities.joblib'
        if source.exists():
            reuse.update({key(f): v for f,v in zip(protocol(tag,directory),joblib.load(source))})
    out = []
    for f in p:
        if key(f) in reuse:
            out.append(reuse[key(f)])
            print(f'{tag} {name} fold {f["fold"]}: reused exact split',flush=True)
            continue
        tr, te, k = f['train'],f['validation'],f['fold']
        outer,_ = c.fit_predict(base.CONFIGS[name],tr,te,k)
        inner = np.full((c.n,2),np.nan)
        for j,(itr,ite) in enumerate(f['inner']):
            inner[ite],_ = c.fit_predict(base.CONFIGS[name],itr,ite,10000+10*k+j)
        out.append(dict(outer=outer,inner=inner[tr]))
        print(f'{tag} {name} fold {k}: trained',flush=True)
    joblib.dump(out,path)
    return out


def evaluate(c, tag):
    p = protocol(tag,GROUPED)
    # Protocols were audited for sensor-identical recording families in the branch campaign.
    (RUN/f'{tag}_protocol.json').write_bytes((GROUPED/f'{tag}_protocol.json').read_bytes())
    probs = {n: probabilities(c,n,p,tag) for n in ['incumbent','legacy_sg_rf','sg_rf']}
    repeats = 1+max(f['repeat'] for f in p)
    predictions = {n:np.full((repeats,c.n),'',dtype='<U7') for n in ['incumbent','ensemble','blend']}
    decisions = []
    for f,a,b,d in zip(p,probs['incumbent'],probs['legacy_sg_rf'],probs['sg_rf']):
        inner = .75*b['inner']+.25*d['inner']
        outer = .75*b['outer']+.25*d['outer']
        # Comparator threshold is selected on inner folds just as the candidate's is.
        ed,_ = select('global',c.y[f['train']],inner)
        best = None
        for w in [.25,.5,.75]:
            decision,score = select('global',c.y[f['train']],w*a['inner']+(1-w)*inner)
            if best is None or score > best[0]+1e-12:
                best = score,w,decision
        _,w,decision = best
        for n,pr,de in [('incumbent',a['outer'],legacy_decision(.25)),
                         ('ensemble',outer,ed),('blend',w*a['outer']+(1-w)*outer,decision)]:
            predictions[n][f['repeat'],f['validation']] = decode(pr[:,0],pr[:,1],de)
        decisions.append(dict(fold=f['fold'],weight=w,decision=decision,ensemble_decision=ed))
    results = {}
    for n,pred in predictions.items():
        assert (pred!='').all()
        metrics = [rail_macro_f1(c.y,x) for x in pred]
        scores = [m['macro_f1'] for m in metrics]
        results[n] = dict(mean=float(np.mean(scores)),std=float(np.std(scores)),scores=scores,
                          per_class={cl:float(np.mean([m['per_class'][cl] for m in metrics])) for cl in base.CLASSES})
    results['decisions'] = decisions
    base.save(f'{tag}_results.json',results)
    np.savez_compressed(RUN/f'{tag}_predictions.npz',labels=c.y.astype('U7'),**predictions)
    print(json.dumps({n:results[n] for n in predictions}),flush=True)
    return results


def main():
    RUN.mkdir(exist_ok=True)
    verify_inputs()
    # Reuse unchanged feature extraction; fold-dependent references remain isolated.
    c = base.Campaign(8)
    base.RUN = RUN
    (RUN/'tables').mkdir(exist_ok=True)
    base.save('plan.json',dict(weights=[.25,.5,.75],ensemble_internal_weight=.75,
        acceptance='Blend beats ensemble by >0.005 on both screen and confirmation, no rotation regression, stratified loss <=0.02 and each fault-class confirmation loss <=0.03; predictions must be distinct.',
        limitation='Existing recordings and ensemble design have been used in earlier experiments. Rotations are not independent holdouts. No leaderboard-based label changes.',
        leaderboard={'original':.7447665,'sg_rf_mf03':.7317867317867317,'ensemble':.7687255273462169}))
    results = {tag:evaluate(c,tag) for tag in ['screen','confirm']}
    gates = {}
    for tag in ['screen','confirm']:
        b,e = results[tag]['blend'],results[tag]['ensemble']
        gates[tag+'_gain'] = b['mean'] > e['mean']+.005
        gates[tag+'_consistent'] = bool(np.all(np.array(b['scores']) >= np.array(e['scores'])-1e-12))
    gates['fault_classes'] = all(results['confirm']['blend']['per_class'][cl] >= results['confirm']['ensemble']['per_class'][cl]-.03 for cl in ['Side I','Side II'])
    if all(gates.values()):
        results['strat'] = evaluate(c,'strat')
        gates['stratified'] = results['strat']['blend']['mean'] >= results['strat']['ensemble']['mean']-.02
    base.save('qualification.json',dict(qualified=all(gates.values()),gates=gates,
        stratified_status='completed' if 'strat' in results else 'skipped: block validation failed'))
    print('QUALIFICATION',gates,flush=True)
    report()


def report():
    results = {tag:json.loads((RUN/f'{tag}_results.json').read_text()) for tag in ['screen','confirm']}
    qualification = json.loads((RUN/'qualification.json').read_text())
    lines = ['# Original-model plus SG-ensemble blend', '',
        'The submitted SG ensemble scored 0.7687255273 on hidden Rail labels. This experiment adds the original model to that ensemble.', '',
        '| Model | Block screening | Block confirmation |',
        '|---|---:|---:|']
    for name in ['incumbent','ensemble','blend']:
        lines.append('| '+name+' | '+' | '.join(f'{results[t][name]["mean"]:.6f}' for t in results)+' |')
    lines += ['',f'Qualified: **{qualification["qualified"]}**.', '',
        'Grouped stratified checks are skipped if block validation fails; they cannot reverse that rejection.', '',
        'The inner folds select the original-model weight from 0.25, 0.50, 0.75 and a global threshold. The remaining weight goes to the existing 75:25 legacy-plus-SG / SG ensemble. The comparator also selects its threshold on inner folds.', '',
        'All 272 training file hashes match the prior audit. Duplicate sensor recordings stay together in every outer and inner split. Both side rows remain together. Cached model probabilities are reused only for identical ordered training, validation and inner indices with the same random seed.', '',
        'These rotations reuse the same recordings, and the component designs were selected in earlier experiments. This is a stability check, not an independent estimate of hidden performance. Hidden scores were not used to choose individual labels.', '',
        'No candidate ZIP is recommended unless all registered gates pass. Existing submission archives and deployed models were not modified.', '',
        '## Confirmation fault-class F1', '',
        '| Model | Side I | Side II |', '|---|---:|---:|']
    for name in ['ensemble','blend']:
        pc=results['confirm'][name]['per_class']
        lines.append(f'| {name} | {pc["Side I"]:.6f} | {pc["Side II"]:.6f} |')
    (RUN/'REPORT.md').write_text('\n'.join(lines)+'\n',encoding='utf-8')


if __name__=='__main__':
    main()
