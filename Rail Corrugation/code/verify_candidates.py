"""Replay every saved candidate on raw test files and audit archive invariants."""
import hashlib
import json
from pathlib import Path
import sys
import zipfile

import joblib
import pandas as pd
from joblib import Parallel, delayed

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT))
from common.research import fingerprint
from rail_corrugation.predict_candidate import predict_one


def main():
    run = ROOT/'weights/Rail Corrugation/runs/2026-09-19-model-experiments'
    baseline = ROOT/'weights/submissions/2026-09-18-submission1-best'
    original = json.loads((baseline/'submission.json').read_text())
    assert fingerprint(baseline/'predictions.zip') == original['submitted_zip_sha256']
    for sub, model in original['active_models'].items():
        assert fingerprint(ROOT/sub/'model'/model['file']) == model['sha256'], sub
        assert json.loads((ROOT/sub/'model/active_model.json').read_text()) == model, sub
    assert fingerprint(ROOT/'predictions.zip') == original['submitted_zip_sha256']
    for name,digest in original['members'].items():
        assert fingerprint(ROOT/'predictions'/name) == digest, name
    outputs = json.loads((run/'outputs.json').read_text())
    checked, vectors, named_vectors = [],set(),{}
    for output in outputs:
        folder = Path(output)
        metadata = json.loads((folder/'submission.json').read_text())
        assert metadata['validation']['qualified']
        assert fingerprint(folder/'predictions.zip') == metadata['zip_sha256']
        assert fingerprint(folder/'rail_candidate.joblib') == metadata['model_sha256']
        with zipfile.ZipFile(folder/'predictions.zip') as z:
            assert set(z.namelist()) == set(original['members'])
            assert z.testzip() is None
            for name in z.namelist():
                data = z.read(name)
                assert data == (folder/name).read_bytes()
                assert hashlib.sha256(data).hexdigest() == metadata['members'][name]
                if name != 'rail_predictions.csv':
                    assert metadata['members'][name] == original['members'][name]
        csv = pd.read_csv(folder/'rail_predictions.csv')
        assert list(csv.columns) == ['file_id','prediction']
        assert len(csv) == 68 and csv.file_id.nunique() == 68
        assert set(csv.file_id) == {f'Test{i}.csv' for i in range(1,69)}
        assert set(csv.prediction) <= {'Normal','Side I','Side II'}
        signature = tuple(csv.sort_values('file_id').prediction)
        assert signature not in vectors
        vectors.add(signature)
        named_vectors[folder.name] = signature
        artifact = joblib.load(folder/'rail_candidate.joblib')
        actual = Parallel(n_jobs=4)(delayed(predict_one)(ROOT/'data/Rail_Corrugation/Test'/name,artifact) for name in csv.file_id)
        assert actual == csv.prediction.tolist(), folder.name
        checked.append(dict(folder=str(folder),raw_recordings_replayed=len(actual),zip_verified=True,
                            other_subsystems_byte_identical=True,model_replay_exact=True))
        print(f'VERIFIED {folder.name}: all 68 model predictions reproduced',flush=True)
    differences = {a+' vs '+b:sum(x != y for x,y in zip(av,bv))
                   for a,av in named_vectors.items() for b,bv in named_vectors.items() if a < b}
    (run/'verification.json').write_text(json.dumps(dict(active_models_unchanged=True,
        active_predictions_unchanged=True,archives=checked,pairwise_test_prediction_differences=differences),indent=2),encoding='utf-8')


if __name__ == '__main__':
    main()
