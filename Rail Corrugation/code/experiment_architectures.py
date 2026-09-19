"""Registered architecture experiments against the successful Rail ensemble."""
import argparse
import hashlib
import inspect
import json
import sys
from pathlib import Path

import joblib
import numpy as np
from joblib import Parallel,delayed

ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT))
from rail_corrugation import experiment_speed_ablation as previous
from rail_corrugation import experiment_models as base
from rail_corrugation.experiment_incumbent_blend import protocol
from rail_corrugation.decision import decode,select
from common.metrics import rail_macro_f1
from common.research import fingerprint,json_value

RUN=ROOT/'weights/Rail Corrugation/runs/2026-09-19-architecture-experiments'
GROUPED=previous.GROUPED
CACHE=previous.RUN
CONFIGS={
    'multiclass_sg':dict(rep='sg',architecture='multiclass'),
    'multiclass_legacy_sg':dict(rep='legacy_sg',architecture='multiclass'),
    'hierarchical_sg':dict(rep='sg',architecture='hierarchical'),
    'hierarchical_legacy_sg':dict(rep='legacy_sg',architecture='hierarchical'),
    'sg_nospeed_mf05':dict(rep='sg',kind='rf',max_features=.5,drop_sg=True),
}
MIXES={'multiclass_ensemble':('multiclass_legacy_sg','multiclass_sg'),
       'hierarchical_ensemble':('hierarchical_legacy_sg','hierarchical_sg')}
CANDIDATES=list(CONFIGS)+list(MIXES)
CANDIDATES+=['blend__'+name for name in list(CANDIDATES)]


def save(name,value):
    (RUN/name).write_text(json.dumps(value,indent=2,default=json_value,allow_nan=False),encoding='utf-8')


def signature(folds,cfg,seed_offset):
    registered=json.loads((RUN/'plan.json').read_text())
    inherited=Path(__file__).with_name('experiment_speed_ablation.py')
    if 'inherited_source_sha256' in registered:
        assert fingerprint(inherited)==registered['inherited_source_sha256']
    sources={n:fingerprint(Path(__file__).with_name(n)) for n in ['paired_models.py','pipeline.py','train.py','representations.py','sg_features.py']}
    value=dict(folds=folds,config=cfg,seed_offset=seed_offset,sources=sources,
               fit_source=FIT_SOURCE_HASH,
               labels=fingerprint(base.DATA/'Train_Labels.csv'))
    return hashlib.sha256(json.dumps(value,sort_keys=True,default=json_value).encode()).hexdigest()


def metrics(y,pred):
    scores=[rail_macro_f1(y,p) for p in pred]
    return dict(mean=float(np.mean([s['macro_f1'] for s in scores])),
        std=float(np.std([s['macro_f1'] for s in scores])),scores=[s['macro_f1'] for s in scores],
        per_class={cl:float(np.mean([s['per_class'][cl] for s in scores])) for cl in base.CLASSES})


def reference_job(tr,ct,v,y):
    h=hashlib.sha256(np.asarray(tr,dtype=np.int64).tobytes()).hexdigest()[:16]
    path=RUN/'tables'/f'{h}.joblib'
    if not path.exists():
        ref,table=base.research_table(ct,v,y,tr)
        cols=base.research_columns(table.columns,'legacy')
        joblib.dump((ref,cols,table[cols].to_numpy()),path)


def prime_references(c,folds):
    pending={}
    for f in folds:
        for tr in [f['train']]+[a for a,b in f['inner']]:
            h=hashlib.sha256(np.asarray(tr,dtype=np.int64).tobytes()).hexdigest()[:16]
            if not any((d/'tables'/f'{h}.joblib').exists() for d in [RUN,previous.PREVIOUS,previous.OLD]):
                pending[tuple(tr)]=tr
    if pending:
        print(f'Preparing {len(pending)} fold-local reference tables',flush=True)
        Parallel(n_jobs=4)(delayed(reference_job)(tr,c.ct,c.v,c.y) for tr in pending.values())


class Campaign(previous.SpeedCampaign):
    def __init__(self,jobs=8):
        previous.RUN=RUN
        super().__init__(jobs)

    def fit_predict(self,cfg,tr,te,seed,final=False,testdata=None,testlegacy=None):
        from rail_corrugation.paired_models import fit_paired_model,paired_probability
        if 'architecture' not in cfg:
            return super().fit_predict(cfg,tr,te,seed,final,testdata,testlegacy)
        x=self.design(cfg['rep'],tr)
        xt=x if testdata is None else self.design(cfg['rep'],tr,testdata,testlegacy)
        model=fit_paired_model(x[base.rows(tr)],self.y[tr],cfg,seed,self.jobs)
        p=paired_probability(model,xt[base.rows(te)])
        art=dict(config=cfg,paired_model=model)
        if 'legacy' in cfg['rep']:
            art.update(ref=self.table(tr)[0],cols=self.table(tr)[1])
        return p,art if final else None

    def probabilities(self,name,folds,tag,seed_offset=0):
        if name=='ensemble':
            a=self.probabilities('legacy_sg_rf',folds,tag,seed_offset)
            b=self.probabilities('sg_rf',folds,tag,seed_offset)
            return [{k:.75*x[k]+.25*y[k] for k in ['outer','inner']} for x,y in zip(a,b)]
        if name in MIXES:
            a=self.probabilities(MIXES[name][0],folds,tag,seed_offset)
            b=self.probabilities(MIXES[name][1],folds,tag,seed_offset)
            return [{k:.75*x[k]+.25*y[k] for k in ['outer','inner']} for x,y in zip(a,b)]
        cfg=CONFIGS[name] if name in CONFIGS else previous.CONFIGS[name]
        sig=signature(folds,cfg,seed_offset)
        path=RUN/f'{tag}_{name}_probabilities.joblib'
        if path.exists():
            value=joblib.load(path)
            assert value['signature']==sig
            return value['values']
        if name in ['legacy_sg_rf','sg_rf'] and seed_offset==0 and tag in ['screen','confirm']:
            assert json.dumps(folds,default=json_value,sort_keys=True)==json.dumps(protocol(tag,CACHE),default=json_value,sort_keys=True)
            value=joblib.load(CACHE/f'{tag}_{name}_probabilities.joblib')['values']
            joblib.dump(dict(signature=sig,values=value),path)
            return value
        if name=='sg_nospeed_mf05' and seed_offset==0 and tag in ['screen','confirm','strat']:
            # The earlier SG-only ablation used exactly these columns, forest settings and seeds.
            assert json.dumps(folds,default=json_value,sort_keys=True)==json.dumps(protocol(tag,GROUPED),default=json_value,sort_keys=True)
            registered=json.loads((GROUPED/'registered_candidates.json').read_text())
            assert registered['branch_sg_nospeed_mf05']==dict(rep='sg',kind='rf',max_features=.5,no_speed=True)
            assert [c for c in self.sg_columns if 'speed' not in c and c!='is_moving']==[c for c in self.sg_columns if c not in previous.SG_DROP]
            for filename in ['train.py','representations.py','sg_features.py']:
                assert fingerprint(GROUPED/'source'/filename)==fingerprint(Path(__file__).with_name(filename))
            value=joblib.load(GROUPED/f'{tag}_branch_sg_nospeed_mf05_probabilities.joblib')
            joblib.dump(dict(signature=sig,values=value,source='branch_sg_nospeed_mf05'),path)
            return value
        result=[]
        for f in folds:
            k,tr,te=f['fold'],f['train'],f['validation']
            checkpoint=RUN/'folds'/f'{sig}_{k}.joblib'
            if checkpoint.exists():
                result.append(joblib.load(checkpoint))
                continue
            outer,_=self.fit_predict(cfg,tr,te,seed_offset+k)
            inner=np.full((self.n,2),np.nan)
            for j,(itr,ite) in enumerate(f['inner']):
                inner[ite],_=self.fit_predict(cfg,itr,ite,seed_offset+10000+10*k+j)
            assert np.isfinite(outer).all() and np.isfinite(inner[tr]).all()
            value=dict(outer=outer,inner=inner[tr])
            joblib.dump(value,checkpoint)
            result.append(value)
            print(f'{tag} {name} fold {k+1}/{len(folds)}',flush=True)
        joblib.dump(dict(signature=sig,values=result),path)
        return result

    def evaluate(self,folds,tag,names,seed_offset=0):
        previous.validate_protocol(folds,self.groups)
        save(f'{tag}_protocol.json',folds)
        probabilities={}
        def get(name):
            if name not in probabilities:
                probabilities[name]=self.probabilities(name,folds,tag,seed_offset)
            return probabilities[name]
        results={}
        repeats=max(f['repeat'] for f in folds)+1
        for name in names:
            pred=np.full((repeats,self.n),'',dtype='<U7')
            oof=np.full((repeats,self.n,2),np.nan)
            decisions=[]
            values=get(name.removeprefix('blend__'))
            incumbent=get('ensemble') if name.startswith('blend__') else None
            for i,(f,p) in enumerate(zip(folds,values)):
                tr,te,repeat=f['train'],f['validation'],f['repeat']
                if incumbent is None:
                    weight=None
                    decision,_=select('global',self.y[tr],p['inner'])
                    outer=p['outer']
                else:
                    best=None
                    for w in [.25,.5,.75]:
                        inner=w*p['inner']+(1-w)*incumbent[i]['inner']
                        de,score=select('global',self.y[tr],inner)
                        if best is None or score>best[0]+1e-12:
                            best=score,w,de
                    _,weight,decision=best
                    outer=weight*p['outer']+(1-weight)*incumbent[i]['outer']
                pred[repeat,te]=decode(outer[:,0],outer[:,1],decision)
                oof[repeat,te]=outer
                decisions.append(dict(fold=f['fold'],decision=decision,weight=weight))
            assert (pred!='').all()
            result=metrics(self.y,pred)
            result['decisions']=decisions
            results[name]=result
            save(f'{tag}_{name}.json',result)
            np.savez_compressed(RUN/f'{tag}_{name}_predictions.npz',labels=self.y.astype('U7'),prediction=pred,probabilities=oof)
            print(f'RESULT {tag} {name}: {result["mean"]:.6f} {result["scores"]}',flush=True)
        return results


FIT_SOURCE_HASH=hashlib.sha256(inspect.getsource(Campaign.fit_predict).encode()).hexdigest()


def main():
    ap=argparse.ArgumentParser()
    ap.add_argument('--stage',choices=['screen','confirm','robust','reference'],default='screen')
    ap.add_argument('--names')
    args=ap.parse_args()
    RUN.mkdir(exist_ok=True)
    (RUN/'tables').mkdir(exist_ok=True)
    (RUN/'folds').mkdir(exist_ok=True)
    if not (RUN/'plan.json').exists():
        save('plan.json',dict(configs=CONFIGS,mixes=MIXES,candidates=CANDIDATES,blend_grid=[.25,.5,.75],
            screening_offsets=[0,27,54],confirmation_offsets=[14,41,68],
            gates='Gain >.005 in both screen/confirmation; no rotation regression; each fault F1 loss <=.03. Blend must also exceed its candidate component by >.005 in both rounds.',
            robustness='Grouped stratified loss <=.02; paired extra seeds1000,2000 on confirmation offset14: each gain >=0, mean gain>.005; pooled speed-range score loss<=.02.',
            stopping='Package only a candidate passing all gates and producing distinct hidden predictions; do not loosen gates.',
            caveat='Repeated model search reuses these272 recordings; this is not independent confirmation and cannot guarantee leaderboard gains.',
            revisit='sg_nospeed_mf05 was previously rejected against sg_rf_mf03, which underperformed on the leaderboard. It is rechecked against the successful ensemble with additional robustness gates, not called a new unseen trial.',
            inherited_source_sha256=fingerprint(Path(__file__).with_name('experiment_speed_ablation.py')),
            winner_zip_sha256=fingerprint(previous.WINNER/'predictions.zip')))
    c=Campaign()
    read=lambda tag,n:json.loads((RUN/f'{tag}_{n}.json').read_text())
    if args.stage=='screen':
        names=['ensemble']+CANDIDATES
        c.evaluate(protocol('screen',GROUPED),'screen',names)
    elif args.stage=='confirm':
        names=args.names.split(',') if args.names else CANDIDATES
        required=list(dict.fromkeys(['ensemble']+[n.removeprefix('blend__') for n in names]+names))
        c.evaluate(protocol('confirm',GROUPED),'confirm',required)
        q={}
        for n in names:
            screen={x:read('screen',x) for x in ['ensemble',n]}
            confirm={x:read('confirm',x) for x in ['ensemble',n]}
            gates=previous.block_gates(screen,confirm,n)
            if n.startswith('blend__'):
                for tag in ['screen','confirm']:
                    gates[tag+'_beats_component']=read(tag,n)['mean']>read(tag,n.removeprefix('blend__'))['mean']+.005
            q[n]=dict(block_qualified=all(gates.values()),qualified=False,gates=gates)
        save('qualification.json',q)
        print(json.dumps(q),flush=True)
    elif args.stage=='reference':
        sets=[('strat',protocol('strat',GROUPED),0),
              ('seed_1000',protocol('confirm',GROUPED)[:5],1000),
              ('seed_2000',protocol('confirm',GROUPED)[:5],2000),
              ('speed',previous.speed_protocol(c.v,c.groups),0)]
        prime_references(c,[f for _,folds,_ in sets for f in folds])
        for tag,folds,seed in sets:
            c.evaluate(folds,tag,['ensemble'],seed)
    else:
        q=json.loads((RUN/'qualification.json').read_text())
        names=args.names.split(',') if args.names else [n for n,v in q.items() if v['block_qualified']]
        if not names:
            print('No block-qualified candidates.',flush=True)
            return
        strat=c.evaluate(protocol('strat',GROUPED),'strat',['ensemble']+names)
        seed_results=[c.evaluate(protocol('confirm',GROUPED)[:5],f'seed_{s}',['ensemble']+names,s) for s in [1000,2000]]
        speed=c.evaluate(previous.speed_protocol(c.v,c.groups),'speed',['ensemble']+names)
        for n in names:
            gains=[r[n]['mean']-r['ensemble']['mean'] for r in seed_results]
            q[n]['gates'].update(stratified=strat[n]['mean']>=strat['ensemble']['mean']-.02,
                seeds_consistent=min(gains)>=-1e-12,seeds_mean=float(np.mean(gains))>.005,
                speed_ranges=speed[n]['mean']>=speed['ensemble']['mean']-.02)
            q[n]['qualified']=all(q[n]['gates'].values())
        save('qualification.json',q)
        print(json.dumps(q),flush=True)


if __name__=='__main__':
    main()
