"""Second registered round: file-class weighting and channel-power SG spectra."""
import hashlib
import inspect
import json
import sys
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from joblib import Parallel,delayed

ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT))
from rail_corrugation import experiment_architectures as driver
from rail_corrugation import experiment_models as base
from rail_corrugation import experiment_speed_ablation as previous
from common.research import fingerprint,json_value

ARCH_RUN=driver.RUN
RUN=ROOT/'weights/Rail Corrugation/runs/2026-09-19-weighting-spectra'
GROUPED=driver.GROUPED
CONFIGS={
    'fileweighted_sg':dict(rep='sg',kind='rf',file_balanced=True,class_weight=None),
    'fileweighted_legacy_sg':dict(rep='legacy_sg',kind='rf',file_balanced=True,class_weight=None),
    'pooled_sg':dict(rep='sg',kind='rf',pooled=True),
    'pooled_legacy_sg':dict(rep='legacy_sg',kind='rf',pooled=True),
}
MIXES={'fileweighted_ensemble':('fileweighted_legacy_sg','fileweighted_sg'),
       'pooled_ensemble':('pooled_legacy_sg','pooled_sg')}
CANDIDATES=list(CONFIGS)+list(MIXES)
CANDIDATES+=['blend__'+name for name in list(CANDIDATES)]
ParentCampaign=driver.Campaign


def file_weights(labels):
    labels=np.asarray(labels)
    unique,counts=np.unique(labels,return_counts=True)
    weights={cl:len(labels)/(len(unique)*n) for cl,n in zip(unique,counts)}
    return np.array([weights[cl] for cl in labels])


def local_pooled(files,sg,columns,tag,jobs=4):
    from rail_corrugation.spectral_variants import pooled_sg_rows
    path=RUN/f'{tag}_pooled_features.joblib'
    signature=dict(files=[(f.name,f.stat().st_size,f.stat().st_mtime_ns) for f in files],
        source=fingerprint(Path(__file__).with_name('spectral_variants.py')),columns=columns)
    if path.exists():
        cached=joblib.load(path)
        assert cached['signature']==signature
        return cached['values']
    print(f'Extracting channel-power spectra for {len(files)} {tag} recordings',flush=True)
    values=Parallel(n_jobs=jobs)(delayed(pooled_sg_rows)(f,pd.DataFrame(sg[2*i:2*i+2],columns=columns)) for i,f in enumerate(files))
    array=np.concatenate([np.asarray(x,dtype=np.float32) for x in values])
    joblib.dump(dict(signature=signature,values=array),path)
    return array


class Campaign(ParentCampaign):
    def __init__(self,jobs=8):
        super().__init__(jobs)
        self.pooled=local_pooled([base.DATA/'Train'/n for n in self.lab.filename],self.data['sg'],self.sg_columns,'train')

    def augment_testdata(self,testdata,files):
        result=dict(testdata)
        result['sg_pooled']=local_pooled(files,result['sg'],self.sg_columns,'test')
        return result

    def fit_predict(self,cfg,tr,te,seed,final=False,testdata=None,testlegacy=None):
        parts=[]
        other=[]
        art=dict(config=cfg)
        if 'legacy' in cfg['rep']:
            ref,cols,x=self.table(tr)
            parts.append(x)
            other.append(x if testlegacy is None else testlegacy)
            art.update(ref=ref,cols=cols)
        if 'sg' in cfg['rep']:
            x=self.pooled if cfg.get('pooled') else self.data['sg']
            xt=x if testdata is None else testdata['sg_pooled' if cfg.get('pooled') else 'sg']
            parts.append(x)
            other.append(xt)
        x=np.nan_to_num(np.concatenate(parts,axis=1),nan=-9,posinf=1e6,neginf=-1e6)
        xt=np.nan_to_num(np.concatenate(other,axis=1),nan=-9,posinf=1e6,neginf=-1e6)
        weights=np.repeat(file_weights(self.y[tr]),2) if cfg.get('file_balanced') else None
        model=base.research_model(cfg,seed,self.jobs).fit(x[base.rows(tr)],self.sy[base.rows(tr)],sample_weight=weights)
        art['model']=model
        p=model.predict_proba(xt[base.rows(te)])[:,list(model.classes_).index(1)].reshape(-1,2)
        return p,art if final else None

    def probabilities(self,name,folds,tag,seed_offset=0):
        if name in ['legacy_sg_rf','sg_rf']:
            path=ARCH_RUN/f'{tag}_{name}_probabilities.joblib'
            if path.exists():
                assert json.dumps(folds,default=json_value,sort_keys=True)==json.dumps(driver.protocol(tag,ARCH_RUN),default=json_value,sort_keys=True)
                return joblib.load(path)['values']
        plan=json.loads((RUN/'plan.json').read_text())
        assert fingerprint(Path(__file__).with_name('spectral_variants.py'))==plan['spectral_source_sha256']
        assert hashlib.sha256(inspect.getsource(file_weights).encode()).hexdigest()==plan['weighting_source_sha256']
        return super().probabilities(name,folds,tag,seed_offset)


def configure():
    driver.RUN=RUN
    driver.CONFIGS=CONFIGS
    driver.MIXES=MIXES
    driver.CANDIDATES=CANDIDATES
    driver.Campaign=Campaign
    driver.FIT_SOURCE_HASH=hashlib.sha256(inspect.getsource(Campaign.fit_predict).encode()).hexdigest()
    RUN.mkdir(exist_ok=True)
    if not (RUN/'plan.json').exists():
        plan=json.loads((ARCH_RUN/'plan.json').read_text())
        plan.update(configs=CONFIGS,mixes=MIXES,candidates=CANDIDATES,
            rationale='Original-file class weighting targets macro-F1 imbalance; channel power pooling tests phase cancellation while keeping the SG feature count fixed.',
            weighting='Within each training fold, original Normal/SideI/SideII file labels receive equal total weight; repeat each file weight on both side rows and disable additional class weights.',
            spectral='Replace exactly30SG spectral columns with Welch powers averaged across channels; preserve all other SG features and canonical side symmetry.',
            spectral_source_sha256=fingerprint(Path(__file__).with_name('spectral_variants.py')),
            weighting_source_sha256=hashlib.sha256(inspect.getsource(file_weights).encode()).hexdigest())
        plan.pop('revisit',None)
        plan.pop('cache_audit',None)
        (RUN/'plan.json').write_text(json.dumps(plan,indent=2,default=json_value),encoding='utf-8')


if __name__=='__main__':
    configure()
    driver.main()
