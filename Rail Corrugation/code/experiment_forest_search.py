"""Registered four-setting forest round on unchanged legacy-plus-SG features."""
import hashlib
import inspect
import json
import sys
from pathlib import Path

import joblib
import numpy as np

ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT))
from rail_corrugation import experiment_architectures as driver
from rail_corrugation import experiment_models as base
from rail_corrugation import experiment_speed_ablation as previous
from common.research import json_value

ARCH_RUN=driver.RUN
RUN=ROOT/'weights/Rail Corrugation/runs/2026-09-19-forest-search'
GROUPED=driver.GROUPED
CONFIGS={f'legacy_sg_mf{int(mf*100):02d}_leaf{leaf}':dict(rep='legacy_sg',kind='rf',max_features=mf,leaf=leaf)
         for mf in [.3,.5] for leaf in [1,2]}
MIXES={name+'_ensemble':(name,'sg_rf') for name in CONFIGS}
CANDIDATES=list(CONFIGS)+list(MIXES)
CANDIDATES+=['blend__'+name for name in list(CANDIDATES)]
ParentCampaign=driver.Campaign


class Campaign(ParentCampaign):
    def table(self,tr):
        key=tuple(tr)
        if key not in self.tables:
            h=hashlib.sha256(np.asarray(tr,dtype=np.int64).tobytes()).hexdigest()[:16]
            p=ARCH_RUN/'tables'/f'{h}.joblib'
            if p.exists():
                self.tables[key]=joblib.load(p)
        return self.tables[key] if key in self.tables else super().table(tr)

    def probabilities(self,name,folds,tag,seed_offset=0):
        if name in ['legacy_sg_rf','sg_rf']:
            path=ARCH_RUN/f'{tag}_{name}_probabilities.joblib'
            if path.exists():
                assert json.dumps(folds,default=json_value,sort_keys=True)==json.dumps(driver.protocol(tag,ARCH_RUN),default=json_value,sort_keys=True)
                return joblib.load(path)['values']
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
            rationale='Test a fixed2x2grid on the unchanged legacy+SG representation: feature fractions .3/.5 and leaf sizes1/2. Each is also compared in the prior75:25 mixture with SG and in a nested blend with the successful ensemble.')
        plan.pop('revisit',None)
        plan.pop('cache_audit',None)
        (RUN/'plan.json').write_text(json.dumps(plan,indent=2,default=json_value),encoding='utf-8')


if __name__=='__main__':
    configure()
    driver.main()
