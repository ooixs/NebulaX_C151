"""Guard the exact Rail checkpoint selected for the app's September 19 release."""
import json
from pathlib import Path

from common.artifacts import load_rail_model, model_key, resolve_model

ROOT = Path(__file__).resolve().parents[1]


def test_selected_rail_ensemble_matches_scored_checkpoint():
    directory = ROOT / 'Rail Corrugation/model'
    path = resolve_model(directory, ('rail_model.joblib',))
    assert model_key(path)[1] == 'bc75821ad886735775c64f296a2f27210176a19bb3ff966d93aa19cfe2f16883'
    assert json.loads((directory / 'active_model.json').read_text())['run_id'] == '2026-09-19-legacy_sg_mf50_leaf2_ensemble-forest'
    artifact = load_rail_model(path)
    assert artifact['weight'] == .75
    assert artifact['decision'] == {'rule': 'global', 'tau': .2}
    assert artifact['full_training_seed'] == 0
    primary, sg = artifact['components']
    assert primary['config'] == {'rep': 'legacy_sg', 'kind': 'rf', 'max_features': .5, 'leaf': 2}
    assert sg['config'] == {'rep': 'sg', 'kind': 'rf'}
    for component in artifact['components']:
        assert component['model'].n_estimators == 800
        assert component['model'].random_state == 0
