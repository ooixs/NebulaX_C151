import numpy as np
import pandas as pd

from rail_corrugation.decision import block_protocol
from rail_corrugation.representations import extract


def test_side_swap_equivariance(tmp_path):
    """Physical side exchange must exchange rows for all three representations."""
    rng = np.random.default_rng(4)
    values = rng.normal(size=(10000,129))
    values[:,0] = np.arange(10000)//20 % 2
    cols = ['Rotating speed'] + [f'{kind} of bearing in position {pos} of car {car}'
            for car in range(1,9) for pos in range(1,9) for kind in ['Vibration','Shock']]
    a = tmp_path/'a.csv'
    b = tmp_path/'b.csv'
    pd.DataFrame(values,columns=cols).to_csv(a,index=False)
    values[:,1:] = values[:,1:].reshape(10000,32,2,2)[:,:,::-1,:].reshape(10000,128)
    pd.DataFrame(values,columns=cols).to_csv(b,index=False)
    original, swapped = extract(a),extract(b)
    for x,y in zip(original,swapped):
        np.testing.assert_allclose(np.asarray(x),np.asarray(y)[::-1],rtol=1e-6,atol=1e-6)


def test_block_protocol_no_recording_leakage():
    for fold in block_protocol(np.arange(1,273),5,3,(0,27,54,14,41,68)):
        tr,te = set(fold['train']),set(fold['validation'])
        assert tr.isdisjoint(te)
        seen = []
        for a,b in fold['inner']:
            assert set(a).isdisjoint(b)
            assert set(a)|set(b) == tr
            assert te.isdisjoint(a) and te.isdisjoint(b)
            seen.extend(b)
        assert sorted(seen) == sorted(tr)
