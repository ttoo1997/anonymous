import ast
from dataclasses import replace
from pathlib import Path
import unittest
import numpy as np
from paper_suite.config import SyntheticSetting
from paper_suite.data import make_synthetic
from paper_suite.runner import run_algorithm, run_recall
from scripts.run_synthetic import build_jobs


class DataBoundaryTests(unittest.TestCase):
    def test_test_data_and_truth_do_not_change_fitted_coefficients(self):
        setting=SyntheticSetting(n=160,m=16,d=16,k=4,epsilon=4.)
        data=make_synthetic(setting,128,test_samples=100)
        changed=replace(data,x_test=np.full_like(data.x_test,1e6),y_test=np.full_like(data.y_test,-1e6),
                        beta_true=np.full_like(data.beta_true,123.))
        for method in ('proposed','2slr','ldpiht','sample_sdp','lasso'):
            with self.subTest(method=method):
                a,_=run_algorithm(method,data,setting,200)
                b,_=run_algorithm(method,changed,setting,200)
                np.testing.assert_array_equal(a,b)
        for method in ('proposed','2slr'):
            a,_=run_recall(method,data,setting,200)
            b,_=run_recall(method,changed,setting,200)
            np.testing.assert_array_equal(a,b)

    def test_synthetic_train_does_not_depend_on_test_sample_count(self):
        setting=SyntheticSetting(n=10,m=5,d=8,k=2)
        a=make_synthetic(setting,99,test_samples=4)
        b=make_synthetic(setting,99,test_samples=11)
        np.testing.assert_array_equal(a.user_x,b.user_x)
        np.testing.assert_array_equal(a.user_y,b.user_y)
        self.assertFalse(np.shares_memory(a.user_x,a.x_test))

    def test_real_fit_interface_has_no_holdout_arguments(self):
        import inspect
        from exp5.algorithms import fit
        self.assertFalse(any('test' in name or 'valid' in name for name in inspect.signature(fit).parameters))

    def test_new_user_grid_and_common_seeds(self):
        import json
        root=Path(__file__).resolve().parents[1]
        cfg=json.loads((root/'configs/experiments.json').read_text())
        self.assertEqual(cfg['studies']['users']['values'],[8000,10000,12000,14000,16000,18000,20000])
        self.assertEqual(cfg['studies']['users']['seeds'],cfg['studies']['privacy']['seeds'])
        self.assertEqual(len(cfg['studies']['privacy']['seeds']),30)
        self.assertEqual(len(cfg['studies']['dimension']['seeds']),10)
        self.assertEqual(len(cfg['real']['seeds']),10)
        jobs=build_jobs(cfg,['users'])
        self.assertEqual(len(jobs),210)
        self.assertTrue(all(j['setting']['epsilon']==4. for j in jobs))


if __name__=='__main__':
    unittest.main()
