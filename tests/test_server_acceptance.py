import hashlib
import json
from pathlib import Path
import sys
import tempfile
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'scripts'))
from verify_server_predictions import verify


class ServerAcceptance(unittest.TestCase):
    def test_archived_counts_and_corrupted_cache(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            (root / 'configs').mkdir()
            assets = []
            for dataset in ('neu', 'gc10'):
                locks = {}
                for arm in ('low', 'high'):
                    rows = [{'image': 'same-image.jpg', 'targets': [[0, 0, 10, 10, 0]],
                             'predictions': [[0, 0, 10, 10, .9, 0]]}]
                    for angle in ('0','45','90','135'):
                        condition = 'unseen_27_' + angle
                        rel = 'records/jobs/' + dataset + '_seed2_' + arm + '_validation27_' + condition + '/predictions.json'
                        path = root / rel
                        path.parent.mkdir(parents=True)
                        path.write_text(json.dumps({condition: rows}))
                        report = path.with_name('report.json')
                        report.write_text(json.dumps({'status':'PASS','checkpoint':{'sha256':'fixture'},
                            'prediction_file': rel, 'artifacts':{rel:hashlib.sha256(path.read_bytes()).hexdigest()}}))
                        assets.append({'dataset':dataset,'arm':arm,'condition':condition,
                            'report':report.relative_to(root).as_posix(),'sha256':hashlib.sha256(report.read_bytes()).hexdigest()})
                    locks[arm] = {'0.25': {'threshold': .5,
                                          'checkpoint':{'sha256':'fixture'},
                                          'validation_counts': {'tp': 4, 'fp': 0, 'gt': 4, 'views': 4}}}
                (root / ('configs/dm_' + dataset + '_seed2_cutoffs.json')).write_text(json.dumps(locks))
            (root / 'configs/server_cpu_validation_inputs.json').write_text(json.dumps({'validation_jobs': assets}))
            result = verify(root, root)
            self.assertEqual(result['status'], 'PASS')
            self.assertEqual(len(result['checks']), 4)
            # Repeated image names in distinct views must remain distinct GTs.
            self.assertTrue(all(x['actual']['tp'] == 4 for x in result['checks']))
            (root / assets[0]['report']).with_name('predictions.json').write_text('[]')
            with self.assertRaisesRegex(ValueError, 'hash mismatch'):
                verify(root, root)

    def test_23_pixel_cache_rejected_for_27_pixel_protocol(self):
        # Wrong condition banks must be rejected before any numerical comparison.
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            (root / 'configs').mkdir()
            (root / 'configs/server_cpu_validation_inputs.json').write_text(json.dumps({'validation_jobs':[
                {'dataset':'neu','arm':'low','condition':'strong_23_0'}]}))
            (root / 'configs/dm_neu_seed2_cutoffs.json').write_text(json.dumps({'low':{}}))
            with self.assertRaisesRegex(ValueError, 'four linear27'):
                verify(root, root)


if __name__ == '__main__':
    unittest.main()
