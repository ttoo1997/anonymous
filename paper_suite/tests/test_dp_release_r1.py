import math
import json
import tempfile
import unittest
from decimal import Decimal, localcontext
from pathlib import Path
from dataclasses import replace
from unittest.mock import patch
import numpy as np
from dp_safety.rr import rr_keep_probability,rr_debias_factor
from dp_safety.aggregation import binomial_epsilon_bound,certify_existing_noise,ensure_local_trials
from lasso_threshold_v1.mechanism.appendix import aggregation_parameters
from lasso_threshold_v1.mechanism.core import StageBudget,RandomStreams,distributed_private_mean_ikos,local_ols_coefficients_private_totalized
from paper_suite.algorithms.sample_sdp_core import sdp2_table_parameters,prepare_exact_sample_statistics,fit_sdp2_from_sum,Config
from paper_suite.file_lock import file_lock

class ReleasePrivacyTests(unittest.TestCase):
    def test_rr_exact_dyadic_budget(self):
        with localcontext() as ctx:
            ctx.prec=100
            for e in [.5,1.,2.,4.,8.,16.,32.,80.]:
                p=rr_keep_probability(e); P=Decimal.from_float(p)
                self.assertLessEqual(P/(1-P),Decimal.from_float(e).exp())
                self.assertEqual(p*(1<<53),int(p*(1<<53)))
                self.assertGreater(p,.5);self.assertLess(p,1.)
                self.assertEqual(rr_debias_factor(e),1/(2*p-1))
    def test_old_rr_counterexample_is_repaired(self):
        with localcontext() as ctx:
            ctx.prec=100;E=Decimal(32).exp()
            old=Decimal.from_float(1/(1+math.exp(-32)))
            self.assertGreater(old-E*(1-old),Decimal('1e-6'))
            new=Decimal.from_float(rr_keep_probability(32.))
            self.assertLessEqual(new-E*(1-new),0)
    def test_fail_closed_bad_privacy_parameters(self):
        for e in [0.,-1.,float('nan'),float('inf')]:
            with self.assertRaises(ValueError):rr_keep_probability(e)
        with self.assertRaises(ValueError):
            certify_existing_noise(n_users=100,dimension=10,modulus=10001,
                total_trials=100,sensitivity_l2=100.,epsilon=1.,delta=1e-6)
    def test_public_only_noise_enlargement(self):
        n0=ensure_local_trials(20,5,10.,4.,1e-6,2)
        self.assertEqual(n0%2,0);self.assertGreater(n0,2)
        self.assertLessEqual(binomial_epsilon_bound(5,10.,20*n0,5e-7),4.)
        self.assertEqual(ensure_local_trials(20,5,10.,4.,1e-6,n0),n0)
    def test_parameters_carry_a_real_certificate(self):
        for eps in [.5,4.,16.,32.]:
            a=aggregation_parameters(3750,8,.7,15000,StageBudget(eps,1e-6),1.)
            cert=a.privacy_certificate
            self.assertLessEqual(cert['epsilon_noise_certified'],eps)
            self.assertLessEqual(cert['delta_total_bound'],1e-6)
            self.assertEqual(cert['shuffled_shares'],a.shares)
            self.assertEqual(a.delta_noise,5e-7)
    def test_sample_formal_parameters_keep_original_noise(self):
        for d in [64,128,256,512]:
            for e in [.5,1,2,4,8,16,32]:
                t=sdp2_table_parameters(750000,d,e,1e-6)
                self.assertFalse(t['noise_recalibrated'])
                self.assertLessEqual(t['privacy_certificate']['delta_total_bound'],1e-6)
    def test_transport_counts_do_not_change_outputs(self):
        p=aggregation_parameters(24,3,1.,96,StageBudget(32.,1e-6),1.)
        vals=np.random.default_rng(19).normal(size=(24,3))
        a,_=distributed_private_mean_ikos(vals,np.zeros(3),p,RandomStreams(9),'t',materialize_ikos=False)
        b,_=distributed_private_mean_ikos(vals,np.zeros(3),p,RandomStreams(9),'t',materialize_ikos=True)
        c,_=distributed_private_mean_ikos(vals,np.zeros(3),replace(p,shares=p.shares+3),RandomStreams(9),'t',materialize_ikos=True)
        np.testing.assert_array_equal(a,b);np.testing.assert_array_equal(b,c)
    def test_local_singular_totalization(self):
        x=np.ones((24,5,3));y=np.ones((24,5))
        beta,info=local_ols_coefficients_private_totalized(x,y,[0,1],return_diagnostics=True)
        np.testing.assert_array_equal(beta,np.zeros((24,2)))
        self.assertEqual(info['local_ols_fallback_count'],24)
    def test_sample_extreme_local_statistics_are_bounded(self):
        x=np.full((32,2),1e200);y=np.full(32,1e200)
        t=prepare_exact_sample_statistics(x,y,3.,2,np.random.default_rng(1))
        self.assertEqual(t['meta']['local_statistic_totalization_count'],32)
        np.testing.assert_array_equal(t['rounded'],np.zeros(5))
    def test_postprivate_inverse_failure_is_totalized(self):
        n,d=200,3;c=Config(sdp2_dimension=d,sdp2_clip=.1*d*math.log(n))
        t=sdp2_table_parameters(n,d,4.,c.delta)
        f=prepare_exact_sample_statistics(np.zeros((n,d)),np.zeros(n),t['clip'],t['precision'],np.random.default_rng(1))
        with patch('paper_suite.algorithms.sample_sdp_core.np.linalg.solve',side_effect=np.linalg.LinAlgError('singular')):
            b,m=fit_sdp2_from_sum(np.zeros(t['D']),f['meta'],4.,c,np.random.default_rng(2),f['rounded'])
        np.testing.assert_array_equal(b,np.zeros(d));self.assertTrue(m['postprivate_solver_fallback'])
    def test_lock_releases_and_nonblocking_refuses(self):
        with tempfile.TemporaryDirectory() as d:
            p=Path(d)/'a.lock'
            with file_lock(p):
                with self.assertRaises(RuntimeError):
                    with file_lock(p,blocking=False):pass
            with file_lock(p,blocking=False):pass
    def test_rng_call_counts_and_parameters_still_frozen(self):
        from paper_suite.algorithms.iht_selected_profile import selected_parameters
        from paper_suite.algorithms.two_slr_paper import reference_profile
        p=selected_parameters(8,750000)
        self.assertEqual((p['iterations'],p['step_size'],p['sparsity']),(2,.01,10))
        t=reference_profile(8,256)
        self.assertEqual((t.local_selected,t.B,t.num_bins,t.hh_alpha),(2,1.,4,.05))
