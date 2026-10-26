import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch
import numpy as np
from scripts.run_real import fit_seed


class RealEntryTests(unittest.TestCase):
    def test_all_methods_and_bad_finite_results_are_retained(self):
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory); data=root/'data'; data.mkdir(); out=root/'results'
            train_x=np.arange(72,dtype=float).reshape(24,3)/100
            train_y=np.linspace(-.5,.5,24)
            for name,array in {'train_x':train_x,'train_y':train_y,'train_days':np.arange(24),
                               'train_ids':np.arange(24)+500,'test_x':np.ones((5,3)),
                               'test_y':np.arange(5,dtype=float)}.items():
                np.save(data/(name+'.npy'),array)
            methods=['u2r','2slr','ldpiht','sample_sdp','lasso']
            spec={'n':2,'m':[10],'epsilon':[1.,4.],'methods':methods,'group_seed_offset':100000}
            profiles={method:{} for method in methods}
            calls=[]
            def fake_fit(method,X,y,m,epsilon,track,profile,seed,cache,data_id,meta,days=None):
                calls.append((method,X.copy(),y.copy(),epsilon,track))
                return np.full(3,.1 if method=='lasso' else 1e6),{}
            with patch('scripts.run_real.fit',side_effect=fake_fit),patch('scripts.run_real.validate_fit'):
                fit_seed(77,data,out,spec,profiles,{'data_id':'fixture','d':3,'y_scale':2.})
            rows=[json.loads(p.read_text()) for p in (out/'cells').glob('*.json')]
            self.assertEqual(len(rows),10)
            self.assertEqual({r['method'] for r in rows},set(methods))
            self.assertTrue(all(r['mse_ratio']>1e8 for r in rows if r['method']!='lasso'))
            self.assertEqual(sum(c[0]=='lasso' for c in calls),1)
            for _,X,y,_,track in calls:
                self.assertEqual(track,'native')
                self.assertEqual(len(X),20)
                self.assertTrue(np.isin(X[:,0],train_x[:,0]).all())
                self.assertTrue(np.isin(y,train_y).all())
            self.assertEqual(len({(r['draw_checksum']['crc32'],r['draw_checksum']['size_bytes']) for r in rows}),1)


if __name__=='__main__':
    unittest.main()
