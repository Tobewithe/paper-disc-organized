"""Decision and full-output bookkeeping regressions; no model training or scientific data."""
import tempfile
import unittest
from pathlib import Path
from rfc_eval_stats import summarize, GROUPS
from rfc_evaluation import PredictionWriter
from posttrain_run import read_json


def fixture(a=.02,q=.004,j=.003):
    rows=[]; gT=.010024491543988425
    other=(508*.0008909711434980322-257*gT)/251
    for i in range(508):
        target=i<257
        n=(3 if i<141 else 2) if target else (10 if i<268 else 9)
        for _ in range(n):
            g=gT if target else other
            values={k:0. for k in GROUPS}
            values.update(G=g,A=a if target else 0.,Q=q if target else 0.,J=j if target else 0.,
                          AeqQ=a if target else 0.,QeqA=q if target else 0.,
                          AeqJ=a if target else 0.,JeqA=j if target else 0.)
            rows.append({'image_id':i,'annotation_id':len(rows),'population':'T' if target else 'S',
                         **{k+'_delta':v for k,v in values.items()}})
    return rows


class EvaluationTests(unittest.TestCase):
    def test_N_improvement_is_not_G_increment(self):
        result=summarize(fixture(a=.011),{'A':{'gate':'damage_guardrail_passed'}})
        self.assertTrue(result['gates']['S2_repair_A_N'])
        self.assertFalse(result['gates']['extra_utility_A_G'])
        self.assertEqual(result['utility_status'],'failed')
    def test_precision_not_claimed_damage_failure(self):
        result=summarize(fixture(),{'A':{'gate':'blocked_insufficient_precision'}})
        self.assertEqual(result['utility_status'],'blocked')
    def test_joint_mechanism_not_one_favorable_control(self):
        result=summarize(fixture(j=.021),{'A':{'gate':'damage_guardrail_passed'}})
        self.assertEqual(result['utility_status'],'passed')
        self.assertFalse(result['mechanism_supported'])
    def test_missing_and_duplicate_pairs_rejected(self):
        rows=fixture(); rows[-1]=dict(rows[0])
        with self.assertRaises(Exception):summarize(rows,{'A':{'gate':'damage_guardrail_passed'}})
    def test_complete_rows_and_order(self):
        with tempfile.TemporaryDirectory() as d:
            p=Path(d)/'pred.json'; writer=PredictionWriter(p)
            writer.append([{'image_id':8},{'image_id':8}],[13,14]);writer.append([],[]);writer.close()
            self.assertEqual(len(read_json(p)),2)
            self.assertEqual(writer.indices,[13,14])
            writer=PredictionWriter(Path(d)/'invalid.json')
            writer.append([{'image_id':8},{'image_id':8}],[14,13])
            with self.assertRaises(Exception):writer.close()

if __name__=='__main__':unittest.main()
