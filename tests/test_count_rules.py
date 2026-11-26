"""Boundary cases that materially affect frozen FP-budget evaluation."""
from pathlib import Path
import sys, unittest
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'scripts'))
from evaluate_predictions import counts,fit

class CountRules(unittest.TestCase):
    def test_equal_score_group_cannot_be_split_to_meet_budget(self):
        rows=[dict(image='a',targets=[[0,0,10,10,0]],predictions=[[0,0,10,10,.8,0],[20,20,30,30,.8,0]])]
        cutoff=fit(rows,0)
        self.assertGreater(cutoff,.8)
        self.assertEqual(counts(rows,cutoff)['fp'],0)
        self.assertEqual(counts(rows,cutoff)['tp'],0)

    def test_duplicate_and_wrong_class_predictions_are_false_positives(self):
        rows=[dict(image='a',targets=[[0,0,10,10,0]],predictions=[[0,0,10,10,.9,0],[0,0,10,10,.8,0],[0,0,10,10,.7,1]])]
        result=counts(rows,.7)
        self.assertEqual((result['tp'],result['fp'],result['gt']),(1,2,1))

    def test_iou_exactly_half_is_matched(self):
        rows=[dict(image='a',targets=[[0,0,10,10,0]],predictions=[[0,0,5,10,.9,0]])]
        self.assertEqual(counts(rows,.9)['tp'],1)

    def test_fewer_fp_tie_break_and_budget_denominator(self):
        rows=[dict(image=str(i),targets=[[0,0,10,10,0]],predictions=([[0,0,10,10,.8,0],[20,20,30,30,.7,0]] if i==0 else [])) for i in range(4)]
        cutoff=fit(rows,.25)
        self.assertEqual(cutoff,.8)
        self.assertEqual(counts(rows,cutoff)['tp'],1)
        self.assertEqual(counts(rows,cutoff)['fp'],0)

if __name__=='__main__':unittest.main()
