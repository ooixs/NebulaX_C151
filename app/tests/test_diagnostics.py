import copy
import unittest
from app import diagnostics

class ComparisonTests(unittest.TestCase):
    def test_peer_groups_and_minimum_sample_count(self):
        items=[dict(group='Open',metrics=[diagnostics.metric('x','Current',v)]) for v in [1]*9+[20]]
        items.append(dict(group='Close',metrics=[diagnostics.metric('x','Current',100)]))
        result=diagnostics.compare(items)
        self.assertTrue(result[9]['metrics'][0]['unusual'])
        self.assertEqual(result[9]['metrics'][0]['n'],10)
        self.assertIsNone(result[10]['metrics'][0]['low'])
        self.assertFalse(result[10]['metrics'][0]['unusual'])

    def test_merge_recomputes_without_changing_individual_checks(self):
        evidence=[{'items':[dict(metrics=[diagnostics.metric('x','Value',v)])]} for v in [2,3,4]]
        before=copy.deepcopy(evidence)
        result=diagnostics.merge(evidence)
        self.assertEqual(result['items'][0]['metrics'][0]['mean'],3)
        self.assertEqual(evidence,before)

    def test_missing_values_do_not_become_zero(self):
        items=diagnostics.compare([dict(metrics=[diagnostics.metric('x','Value',x)]) for x in [None,float('nan'),2]])
        self.assertIsNone(items[0]['metrics'][0]['value'])
        self.assertEqual(items[2]['metrics'][0]['n'],1)

    def test_trace_retains_extreme_original_sample(self):
        try:
            import numpy as np
        except ImportError:
            self.skipTest('Requires inference environment')
        a=np.zeros(10000);a[7351]=100
        t=diagnostics.trace(a,'Signal')
        self.assertEqual(t['flagged_count'],1)
        self.assertEqual(t['extremes'][0]['index'],7352)
        self.assertTrue(any(p['index']==7352 and p['value']==100 for p in t['points']))
        self.assertLess(len(t['points']),210)

if __name__=='__main__':unittest.main()
