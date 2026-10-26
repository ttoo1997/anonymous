import json
import multiprocessing as mp
import tempfile
import unittest
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path
from dataclasses import asdict
from paper_suite.config import SyntheticSetting
from paper_suite.full_experiment import work,collect,job_is_complete,validate_resume

class ReleaseRunnerTests(unittest.TestCase):
    def test_spawn_workers_checkpoint_and_resume(self):
        st=SyntheticSetting(n=128,m=30,d=8,k=2,epsilon=32.)
        jobs=[{'job_id':0,'kind':'synthetic','experiment':'exp1_epsilon','repeat':0,'setting':asdict(st)},
              {'job_id':1,'kind':'synthetic','experiment':'exp4a_recall_m','repeat':0,'setting':asdict(st)}]
        with tempfile.TemporaryDirectory() as d:
            out=Path(d);(out/'jobs').mkdir()

            with ProcessPoolExecutor(max_workers=2,mp_context=mp.get_context('spawn')) as pool:
                futures=[pool.submit(work,j,str(out)) for j in jobs]
                results=[f.result(timeout=90) for f in futures]
            self.assertTrue(all(r['state']=='finished' for r in results),results)
            self.assertTrue(all(job_is_complete(out,j) for j in jobs))
            p=collect(out,jobs);self.assertTrue(p['all_requested_cells_ok']);self.assertEqual(p['completed_cells'],7)
            before={f.name:f.read_bytes() for f in (out/'jobs/0000').glob('*.row.json')}
            again=work(jobs[0],str(out));self.assertEqual(again['state'],'finished')
            after={f.name:f.read_bytes() for f in (out/'jobs/0000').glob('*.row.json')}
            self.assertEqual(before,after)
    def test_resume_refuses_changed_privacy_code(self):
        a={'jobs':[],'data_checksums':{},'source_checksums':{'dp_safety/rr.py':'old'}}
        b={**a,'source_checksums':{'dp_safety/rr.py':'new'}}
        with self.assertRaisesRegex(ValueError,'source code differs'):validate_resume(a,b)
