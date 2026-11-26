from pathlib import Path
import hashlib,json,sys,tempfile,unittest
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'scripts'))
from collect_server_assets import inside,collect

class CollectionTests(unittest.TestCase):
    def test_traversal_is_rejected(self):
        with self.assertRaises(ValueError):inside(Path.cwd(),'../secret.txt')
        with self.assertRaises(ValueError):inside(Path.cwd(),'C:/secret.txt')

    def test_mismatch_is_reported_without_overwriting_source(self):
        # TemporaryDirectory cleanup is restricted to its explicitly created root.
        with tempfile.TemporaryDirectory(prefix='repro_test_') as temp:
            root=Path(temp);project=root/'project';project.mkdir();(project/'records').mkdir()
            f=project/'records/item.json';f.write_text('{}')
            manifest=root/'manifest.json';manifest.write_text(json.dumps({'assets':[{'path':'records/item.json','sha256':'0'*64,'kind':'predictions'}]}))
            report,_=collect(project,manifest,root/'out',inventory=False)
            self.assertEqual(report['asset_status_counts']['HASH_MISMATCH'],1)
            self.assertEqual(f.read_text(),'{}')
            self.assertFalse((root/'out/records/item.json').exists())

if __name__=='__main__':unittest.main()
