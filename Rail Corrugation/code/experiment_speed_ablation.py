"""Controlled speed-input ablations of the submitted 75:25 Rail ensemble."""
import argparse
import hashlib
import inspect
import json
import sys
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT))
from rail_corrugation import experiment_models as base
from rail_corrugation.experiment_incumbent_blend import protocol
from rail_corrugation.decision import decode, legacy_decision, select
from rail_corrugation.pipeline import speed_bin
from common.metrics import rail_macro_f1
from common.research import fingerprint, json_value

OLD = ROOT/'weights/Rail Corrugation/runs/2026-09-19-model-experiments'
PREVIOUS = ROOT/'weights/Rail Corrugation/runs/2026-09-19-incumbent-blend'
GROUPED = ROOT/'weights/Rail Corrugation/runs/2026-09-19-branch-review'
RUN = ROOT/'weights/Rail Corrugation/runs/2026-09-19-ensemble-speed-ablation'
WINNER = ROOT/'weights/submissions/2026-09-19-blend__legacy_sg_rf__sg_rf'
SG_DROP = ('speed_mps','speed_kmh','is_moving','s1_vib_speed_norm','s2_vib_speed_norm','all_vib_speed_norm')
LEGACY_DROP = ('speed','speed_bin')
CONFIGS = {
    'legacy_sg_rf':dict(rep='legacy_sg',kind='rf'),
    'sg_rf':dict(rep='sg',kind='rf'),
    'legacy_sg_drop_sg':dict(rep='legacy_sg',kind='rf',drop_sg=True),
    'sg_drop_sg':dict(rep='sg',kind='rf',drop_sg=True),
    'legacy_sg_drop_both':dict(rep='legacy_sg',kind='rf',drop_sg=True,drop_legacy=True),
}
VARIANTS = {'ensemble':('legacy_sg_rf','sg_rf'),
            'sg_speed_removed':('legacy_sg_drop_sg','sg_drop_sg'),
            'both_speed_removed':('legacy_sg_drop_both','sg_drop_sg')}


def save(name,value):
    (RUN/name).write_text(json.dumps(value,indent=2,default=json_value,allow_nan=False),encoding='utf-8')


def kept_indices(columns, removed):
    missing = set(removed)-set(columns)
    if missing:
        raise ValueError(f'Missing expected speed columns: {sorted(missing)}')
    return [i for i,col in enumerate(columns) if col not in removed]


def validate_protocol(folds, groups):
    n=len(groups)
    for repeat in sorted({f['repeat'] for f in folds}):
        assert sorted(np.concatenate([f['validation'] for f in folds if f['repeat']==repeat]))==list(range(n))
    for f in folds:
        for tr,te in [(f['train'],f['validation'])]+f['inner']:
            assert set(tr).isdisjoint(te)
            assert set(groups[tr]).isdisjoint(groups[te])
        assert set(f['train'])|set(f['validation'])==set(range(n))
        assert sorted(np.concatenate([b for a,b in f['inner']]))==sorted(f['train'])
        for a,b in f['inner']:
            assert set(a)|set(b)==set(f['train'])


def grouped_inner(indices,groups):
    parts=[set(x.tolist()) for x in np.array_split(indices,3)]
    for group in np.unique(groups[indices]):
        family=set(indices[groups[indices]==group].tolist())
        target=next(i for i,p in enumerate(parts) if min(family) in p)
        for p in parts:
            p.difference_update(family)
        parts[target].update(family)
    return [(np.setdiff1d(indices,list(p)),np.array(sorted(p))) for p in parts]


def speed_protocol(speeds,groups):
    bins=np.array([max(2,speed_bin(v)) for v in speeds])
    for group in np.unique(groups):
        ix=np.flatnonzero(groups==group)
        bins[ix]=max(2,speed_bin(float(np.median(np.asarray(speeds)[ix]))))
    folds=[]
    for b in sorted(set(bins)):
        te=np.flatnonzero(bins==b)
        tr=np.flatnonzero(bins!=b)
        folds.append(dict(fold=len(folds),repeat=0,speed_bin=int(b),train=tr,validation=te,inner=grouped_inner(tr,groups)))
    validate_protocol(folds,groups)
    return folds


class SpeedCampaign(base.Campaign):
    def __init__(self,jobs=8):
        base.RUN=OLD
        super().__init__(jobs)
        base.RUN=RUN
        self.sg_columns=list(base.extract(base.DATA/'Train'/self.lab.filename.iloc[0])[0].columns)
        audit=json.loads((GROUPED/'audit.json').read_text())
        records={r['filename']:r for r in audit['manifest']}
        self.groups=np.array([records[x]['sensor_sha256'] for x in self.lab.filename])
        label_sha=json.loads((OLD/'plan.json').read_text())['labels_sha256']
        assert fingerprint(base.DATA/'Train_Labels.csv')==label_sha
        for name in self.lab.filename:
            assert fingerprint(base.DATA/'Train'/name)==records[name]['sha256'],name
        save('data_verification.json',dict(train_files=self.n,raw_hashes_verified=True,labels_sha256=label_sha))

    def table(self,tr):
        key=tuple(tr)
        if key not in self.tables:
            h=hashlib.sha256(np.asarray(tr,dtype=np.int64).tobytes()).hexdigest()[:16]
            for directory in [RUN,PREVIOUS,OLD]:
                path=directory/'tables'/f'{h}.joblib'
                if path.exists():
                    self.tables[key]=joblib.load(path)
                    break
        return self.tables[key] if key in self.tables else super().table(tr)

    def fit_predict(self,cfg,tr,te,seed,final=False,testdata=None,testlegacy=None):
        parts=[]
        other=[]
        art=dict(config=cfg)
        if 'legacy' in cfg['rep']:
            ref,cols,x=self.table(tr)
            keep=kept_indices(cols,LEGACY_DROP) if cfg.get('drop_legacy') else list(range(len(cols)))
            parts.append(x[:,keep])
            other.append(x[:,keep] if testlegacy is None else testlegacy[:,keep])
            art.update(ref=ref,cols=[cols[i] for i in keep])
        if 'sg' in cfg['rep']:
            keep=kept_indices(self.sg_columns,SG_DROP) if cfg.get('drop_sg') else list(range(len(self.sg_columns)))
            parts.append(self.data['sg'][:,keep])
            other.append((self.data if testdata is None else testdata)['sg'][:,keep])
            art['sg_indices']=keep
        x=np.nan_to_num(np.concatenate(parts,axis=1),nan=-9,posinf=1e6,neginf=-1e6)
        xt=np.nan_to_num(np.concatenate(other,axis=1),nan=-9,posinf=1e6,neginf=-1e6)
        model=base.research_model(cfg,seed,self.jobs).fit(x[base.rows(tr)],self.sy[base.rows(tr)])
        art['model']=model
        p=model.predict_proba(xt[base.rows(te)])[:,list(model.classes_).index(1)].reshape(-1,2)
        return p,art if final else None

    def probabilities(self,name,folds,tag,seed_offset=0):
        cfg=CONFIGS[name]
        lineage=dict(config=cfg,folds=folds,seed_offset=seed_offset,sg_columns=self.sg_columns,
            sg_drop=SG_DROP,legacy_drop=LEGACY_DROP,labels=fingerprint(base.DATA/'Train_Labels.csv'))
        plan=json.loads((RUN/'plan.json').read_text())
        for n in ['pipeline.py','train.py','representations.py','sg_features.py']:
            assert plan['sources'][n]==fingerprint(Path(__file__).with_name(n))
        if 'fit_source' in plan:
            assert plan['fit_source']==hashlib.sha256(inspect.getsource(type(self).fit_predict).encode()).hexdigest()
        signature=hashlib.sha256(json.dumps(lineage,default=json_value,sort_keys=True).encode()).hexdigest()
        path=RUN/f'{tag}_{name}_probabilities.joblib'
        if path.exists():
            cached=joblib.load(path)
            assert cached['signature']==signature
            return cached['values']
        source=PREVIOUS/f'{tag}_{name}_probabilities.joblib'
        if seed_offset==0 and tag in ['screen','confirm'] and name in ['legacy_sg_rf','sg_rf'] and source.exists():
            assert json.dumps(folds,default=json_value,sort_keys=True)==json.dumps(protocol(tag,PREVIOUS),default=json_value,sort_keys=True)
            values=joblib.load(source)
            joblib.dump(dict(signature=signature,values=values,source=str(source)),path)
            return values
        values=[]
        partial=RUN/'folds'
        partial.mkdir(exist_ok=True)
        for f in folds:
            k,tr,te=f['fold'],f['train'],f['validation']
            checkpoint=partial/f'{signature}_{k}.joblib'
            if checkpoint.exists():
                values.append(joblib.load(checkpoint))
                continue
            outer,_=self.fit_predict(cfg,tr,te,seed_offset+k)
            inner=np.full((self.n,2),np.nan)
            for j,(itr,ite) in enumerate(f['inner']):
                inner[ite],_=self.fit_predict(cfg,itr,ite,seed_offset+10000+10*k+j)
            assert np.isfinite(inner[tr]).all()
            item=dict(outer=outer,inner=inner[tr])
            joblib.dump(item,checkpoint)
            values.append(item)
            print(f'{tag}: {name} fold {k+1}/{len(folds)}',flush=True)
        joblib.dump(dict(signature=signature,values=values),path)
        return values

    def evaluate(self,folds,tag,names,seed_offset=0):
        validate_protocol(folds,self.groups)
        save(f'{tag}_protocol.json',folds)
        components=list(dict.fromkeys(x for n in names for x in VARIANTS[n]))
        prob={x:self.probabilities(x,folds,tag,seed_offset) for x in components}
        result={}
        repeats=1+max(f['repeat'] for f in folds)
        for name in names:
            pred=np.full((repeats,self.n),'',dtype='<U7')
            fixed=pred.copy()
            decisions=[]
            a,b=VARIANTS[name]
            for f,pa,pb in zip(folds,prob[a],prob[b]):
                tr,te,repeat=f['train'],f['validation'],f['repeat']
                inner=.75*pa['inner']+.25*pb['inner']
                outer=.75*pa['outer']+.25*pb['outer']
                decision,_=select('global',self.y[tr],inner)
                pred[repeat,te]=decode(outer[:,0],outer[:,1],decision)
                fixed[repeat,te]=decode(outer[:,0],outer[:,1],legacy_decision(.275))
                decisions.append(dict(fold=f['fold'],decision=decision))
            assert (pred!='').all()
            metrics=[rail_macro_f1(self.y,p) for p in pred]
            scores=[m['macro_f1'] for m in metrics]
            result[name]=dict(mean=float(np.mean(scores)),std=float(np.std(scores)),scores=scores,
                per_class={cl:float(np.mean([m['per_class'][cl] for m in metrics])) for cl in base.CLASSES},
                decisions=decisions,fixed_threshold_scores=[rail_macro_f1(self.y,p)['macro_f1'] for p in fixed])
            if tag=='speed':
                from sklearn.metrics import confusion_matrix
                result[name]['ranges']=[dict(speed_bin=f['speed_bin'],support=pd.Series(self.y[f['validation']]).value_counts().to_dict(),
                    confusion=confusion_matrix(self.y[f['validation']],pred[0,f['validation']],labels=base.CLASSES).tolist()) for f in folds]
            np.savez_compressed(RUN/f'{tag}_{name}_predictions.npz',labels=self.y.astype('U7'),prediction=pred,fixed_prediction=fixed)
            print(f'RESULT {tag} {name}: {result[name]["mean"]:.6f} {result[name]["per_class"]}',flush=True)
        save(f'{tag}_results.json',result)
        return result


def block_gates(screen,confirm,name):
    gates={}
    for tag,result in [('screen',screen),('confirm',confirm)]:
        c,b=result[name],result['ensemble']
        gates[tag+'_gain']=c['mean']>b['mean']+.005
        gates[tag+'_consistent']=bool(np.all(np.array(c['scores'])>=np.array(b['scores'])-1e-12))
    gates['fault_classes']=all(confirm[name]['per_class'][cl]>=confirm['ensemble']['per_class'][cl]-.03 for cl in ['Side I','Side II'])
    return gates


def main():
    ap=argparse.ArgumentParser()
    ap.add_argument('--stage',choices=['blocks','robust'],default='blocks')
    args=ap.parse_args()
    RUN.mkdir(exist_ok=True)
    (RUN/'tables').mkdir(exist_ok=True)
    if not (RUN/'plan.json').exists():
        save('plan.json',dict(variants=VARIANTS,configs=CONFIGS,sg_drop=SG_DROP,legacy_drop=LEGACY_DROP,
            weight=.75,trees=800,seed_offsets=[0,1000,2000],
            block_gates='Gain >.005 in screen and confirmation, no rotation regression, fault-class F1 loss <=.03.',
            robustness_gates='Grouped stratified loss <=.02; confirmation offset14 gains nonnegative for each extra seed and mean >.005; pooled speed-range macro-F1 loss <=.02.',
            preserve='Legacy speed-conditioned NormalReference, wavelength features and other indirect speed information.',
            caveat='All protocols reuse existing recordings; this is not a fresh independent holdout. No hidden-label edits.',
            winner_zip_sha256=fingerprint(WINNER/'predictions.zip'),
            fit_source=hashlib.sha256(inspect.getsource(SpeedCampaign.fit_predict).encode()).hexdigest(),
            sources={p.name:fingerprint(p) for p in Path(__file__).parent.glob('*.py')}))
    c=SpeedCampaign()
    if args.stage=='blocks':
        screen=c.evaluate(protocol('screen',GROUPED),'screen',list(VARIANTS))
        confirm=c.evaluate(protocol('confirm',GROUPED),'confirm',list(VARIANTS))
        q={n:dict(gates=block_gates(screen,confirm,n)) for n in list(VARIANTS)[1:]}
        for value in q.values():
            value['block_qualified']=all(value['gates'].values())
            value['qualified']=False
            value['status']='pending robustness' if value['block_qualified'] else 'rejected on block validation'
        save('qualification.json',q)
        print(json.dumps(q),flush=True)
    else:
        q=json.loads((RUN/'qualification.json').read_text())
        names=[n for n,v in q.items() if v['block_qualified']]
        if not names:
            print('No block-qualified candidates; robustness checks not needed.',flush=True)
            return
        names=['ensemble']+names
        strat=c.evaluate(protocol('strat',GROUPED),'strat',names)
        folds=protocol('confirm',GROUPED)[:5]
        seeds=[c.evaluate(folds,f'seed_{seed}',names,seed) for seed in [1000,2000]]
        speed=c.evaluate(speed_protocol(c.v,c.groups),'speed',names)
        for n in names[1:]:
            gains=[r[n]['mean']-r['ensemble']['mean'] for r in seeds]
            gates=q[n]['gates']
            gates.update(stratified=strat[n]['mean']>=strat['ensemble']['mean']-.02,
                seed_consistency=min(gains)>=-1e-12,seed_mean_gain=float(np.mean(gains))>.005,
                speed_ranges=speed[n]['mean']>=speed['ensemble']['mean']-.02)
            q[n]['qualified']=all(gates.values())
            q[n]['status']='qualified for distinct-prediction check' if q[n]['qualified'] else 'rejected on robustness'
        save('qualification.json',q)
        print(json.dumps(q),flush=True)


if __name__=='__main__':
    main()
