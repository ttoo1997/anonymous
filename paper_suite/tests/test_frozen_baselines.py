import json
import math
import unittest
from pathlib import Path
from unittest.mock import patch
import numpy as np
from paper_suite.config import SyntheticSetting
from paper_suite.data import make_synthetic
from paper_suite.runner import run_algorithm,evaluate,ALGORITHMS
from paper_suite.algorithms.common import privunit_batch
from paper_suite.algorithms.nonprivate_lasso import fit_lasso

class FrozenBaselines(unittest.TestCase):
    def test_three_baselines_unchanged_and_sample_only_changes_threshold(self):
        fixture=Path(__file__).with_name('frozen_reference.json')
        previous=json.loads(fixture.read_text())
        setting=SyntheticSetting(n=120,m=30,d=16,k=8)
        data=make_synthetic(setting,81231,test_samples=30)
        for name,expected in previous.items():
            if name == 'sample_sdp':


                from paper_suite.algorithms.sample_sdp import fit_sample_sdp
                from paper_suite.algorithms import sample_sdp_core as core
                raw=[]; solve=np.linalg.solve
                def capture(G,h):
                    result=solve(G,h); raw.append(result.copy()); return result
                with patch.object(core.np.linalg,'solve',side_effect=capture):
                    beta,meta=fit_sample_sdp(data.user_x,data.user_y,setting.epsilon,81232,delta=1e-6)
                self.assertEqual(len(raw),1)
                N=setting.n*setting.m; d=setting.d; D=d*(d+1)//2+d
                clip=.1*d*math.log(N)
                old_lambda=2*clip*math.log(D)/(N*setting.epsilon)
                old_beta=core.project_l1_ball(np.sign(raw[0])*np.maximum(abs(raw[0])-old_lambda,0.),1.)
                np.testing.assert_allclose(old_beta,expected,rtol=1e-10,atol=1e-10)
                new_lambda=2*(.25*math.sqrt(math.log(d)/N)+
                    clip*math.sqrt(math.log(4*D/1e-6)*math.log(D))/(N*setting.epsilon))
                self.assertAlmostEqual(meta['soft_threshold'],new_lambda)
                expected=core.project_l1_ball(np.sign(raw[0])*np.maximum(abs(raw[0])-new_lambda,0.),1.)
            elif name == 'ldpiht':


                from paper_suite.algorithms.iht_selected_profile import selected_parameters
                beta,_=run_algorithm(name,data,setting,81232,
                                     ldpiht_parameters=selected_parameters(setting.k,setting.n*setting.m))
            else:
                beta,_=run_algorithm(name,data,setting,81232)
            np.testing.assert_allclose(beta,expected,rtol=1e-10,atol=1e-10)

    def test_nonprivate_lasso_does_not_call_any_private_selector(self):
        setting=SyntheticSetting(n=80,m=10,d=16,k=8)
        data=make_synthetic(setting,77,test_samples=10)
        with (patch('paper_suite.runner.fit_pdf_proposed',side_effect=AssertionError('shared screening')),
              patch('paper_suite.runner.fit_paper_two_slr',side_effect=AssertionError('shared screening'))):
            beta,meta=run_algorithm('lasso',data,setting,8)
        self.assertEqual(meta['privacy'],'none')
        np.testing.assert_array_equal(beta,fit_lasso(data)[0])

    def test_only_frozen_methods_are_registered(self):
        self.assertEqual(ALGORITHMS,('proposed','2slr','ldpiht','sample_sdp','lasso'))

    def test_iht_is_excluded_from_stage1_recall_experiment(self):
        setting=SyntheticSetting(n=80,m=10,d=16,k=8)
        data=make_synthetic(setting,9,test_samples=10)
        with self.assertRaisesRegex(ValueError,'Stage-1 recall'):
            from paper_suite.runner import run_recall
            run_recall('ldpiht',data,setting,10)
        row=evaluate('ldpiht',data,setting,10,recall_only=False)
        self.assertNotIn('support_recall',row)
        self.assertIn('final_support_recall',row)
        self.assertIn('topk_support_recall',row)

    def test_vectorized_privunit_unbiased(self):
        v=np.array([.2,-.1,.3]);x=np.tile(v,(40000,1))
        reports=privunit_batch(x,1.,4.,np.random.default_rng(12))
        np.testing.assert_allclose(reports.mean(axis=0),v,atol=.025)

    def test_lock_integrity(self):
        from paper_suite.frozen import verify
        self.assertEqual(verify(),'SDP_TABLE2_TABLE5_ALIGNED_20261021')
