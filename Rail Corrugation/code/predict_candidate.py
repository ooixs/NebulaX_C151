"""Standalone candidate inference without changing the active Rail model.

python "Rail Corrugation/code/predict_candidate.py" --model .../rail_candidate.joblib
    --input data/Rail_Corrugation/Test --output .../rail_predictions.csv
"""
import argparse
from pathlib import Path
import sys

import joblib
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from rail_corrugation.decision import decode
from rail_corrugation.pipeline import aggregate, file_channel_table, side_relative_rows
from rail_corrugation.representations import extract, normalized_raw


def component_probability(art, data, ct, speed):
    if 'components' in art:
        a, b = art['components']
        w = art['weight']
        return w*component_probability(a, data, ct, speed)+(1-w)*component_probability(b, data, ct, speed)
    cfg = art['config']
    if cfg['rep'] == 'rocket':
        raw = normalized_raw(data[2]) if cfg.get('normalize') else data[2]
        design = art['transform'].transform(raw).to_numpy()
    else:
        parts = []
        if 'legacy' in cfg['rep']:
            feat = aggregate(art['ref'].excess(ct, speed), speed, paired=True)
            table = pd.DataFrame(side_relative_rows(feat, engineered=True))
            parts.append(table[art['cols']].fillna(-9).to_numpy())
        if 'sg' in cfg['rep']:
            sg = data[3].to_numpy() if cfg.get('pooled') else data[0].to_numpy()
            parts.append(sg[:, art['sg_indices']] if 'sg_indices' in art else sg)
        if 'crosscar' in cfg['rep']:
            parts.append(data[1].to_numpy())
        design = np.nan_to_num(np.concatenate(parts, axis=1), nan=-9, posinf=1e6, neginf=-1e6)
    if 'paired_model' in art:
        from rail_corrugation.paired_models import paired_probability
        return paired_probability(art['paired_model'], design).ravel()
    return art['model'].predict_proba(design)[:, list(art['model'].classes_).index(1)]


def predict_one(path, artifact):
    data = extract(path)
    def needs_pooled(art):
        return any(needs_pooled(x) for x in art['components']) if 'components' in art else art['config'].get('pooled', False)
    if needs_pooled(artifact):
        from rail_corrugation.spectral_variants import pooled_sg_rows
        data = (*data, pooled_sg_rows(path, data[0]))
    ct, speed = file_channel_table(path)
    p = component_probability(artifact,data,ct,speed)
    return str(decode([p[0]],[p[1]],artifact['decision'])[0])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--model',required=True)
    ap.add_argument('--input',required=True)
    ap.add_argument('--output',required=True)
    args = ap.parse_args()
    art = joblib.load(args.model)
    path = Path(args.input)
    files = sorted(path.glob('*.csv'), key=lambda p:int(''.join(filter(str.isdigit,p.stem)))) if path.is_dir() else [path]
    pd.DataFrame([dict(file_id=p.name,prediction=predict_one(p,art)) for p in files]).to_csv(args.output,index=False,lineterminator='\n')


if __name__ == '__main__':
    main()
