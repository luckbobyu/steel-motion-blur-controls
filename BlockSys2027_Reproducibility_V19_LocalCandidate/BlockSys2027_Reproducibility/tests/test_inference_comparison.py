import copy
from pathlib import Path
import sys
import unittest

sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'scripts'))
from verify_checkpoint_inference import compare_predictions


class PredictionComparison(unittest.TestCase):
    def test_geometry_change_is_rejected(self):
        old=[{'image':'a.png','shape':[200,200],'targets':[[0,0,10,10,0]],'predictions':[]}]
        new=copy.deepcopy(old)
        new[0]['targets'][0][0]=1
        with self.assertRaisesRegex(ValueError,'ground truth'):
            compare_predictions(old,new)

    def test_score_difference_is_reported_without_replacing_counts(self):
        old=[{'image':'a.png','shape':[200,200],'targets':[],
              'predictions':[[0,0,10,10,.5,0]]}]
        new=copy.deepcopy(old)
        new[0]['predictions'][0][4]=.6
        self.assertEqual(compare_predictions(old,new)['images_with_prediction_difference_above_tolerance'],1)
        self.assertEqual(compare_predictions(old,old)['images_with_prediction_difference_above_tolerance'],0)


if __name__=='__main__':unittest.main()
