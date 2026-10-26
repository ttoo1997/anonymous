'Regression checks for private release and Ridge postprocessing.'
import math
import tempfile
import unittest
from unittest.mock import patch
import numpy as np
from exp5 import common
from exp5 import sample_ridge as ridge
from exp5.algorithms import fit_sample
from paper_suite.algorithms import sample_sdp_core as core
from paper_suite.frozen import verify
from threadpoolctl import threadpool_limits


class RidgeChecks(unittest.TestCase):
    def test_release_matches_frozen_query_and_random_stream(self):
        rng = np.random.default_rng(9)
        X = rng.normal(size=(200, 7)); y = X[:, 0] * .3 + rng.normal(size=200)
        for eps, delta in ((1., 1e-6), (4., 1e-6), (.04, 1e-9)):
            table = core.sdp2_table_parameters(200, 7, eps, delta, clip_constant=.003, threshold_constant=1.)
            cfg = core.Config(delta=delta, sdp2_dimension=7, sdp2_clip=table['clip'],
                sdp2_clip_constant=.003, sdp2_threshold_constant=1.)
            prep = core.prepare_exact_sample_statistics(X, y, table['clip'], table['precision'], rng)

            captured = {}; solve = np.linalg.solve
            def capture(G, h):
                captured['G'], captured['h'] = G.copy(), h.copy()
                return solve(G, h)
            r1, r2 = np.random.default_rng(83), np.random.default_rng(83)
            with patch.object(core.np.linalg, 'solve', side_effect=capture):
                old, meta = core.fit_sdp2_from_sum(np.zeros(prep['meta']['D']), prep['meta'], eps, cfg, r1, prep['rounded'])
            G, h, new = ridge.release_statistics(prep['rounded'], prep['meta'], eps, cfg, r2)
            np.testing.assert_array_equal(G, captured['G']); np.testing.assert_array_equal(h, captured['h'])
            self.assertEqual(r1.bit_generator.state, r2.bit_generator.state)
            for k in ('privacy_certificate','integer_sum_l2_sensitivity','precision','binomial_trials',
                      'local_binomial_trials','modulus','coordinate_noise_sd','soft_threshold'):
                self.assertEqual(meta[k], new[k], k)

    def test_positive_curvature_and_objective_stationarity(self):
        rng=np.random.default_rng(52); A=rng.normal(size=(6,6)); Q,_=np.linalg.qr(A)
        values=np.array([-3., -.1, 0., .01, 2., 9.]); G=(Q*values)@Q.T
        h=rng.normal(size=6)*.02; multiplier=.7; sd=.1
        beta, meta=ridge.ridge_from_eigh(values,Q,h,0.,sd,multiplier)
        M=G+meta['diagonal_shift_total']*np.eye(6)
        self.assertGreater(np.linalg.eigvalsh(M).min(),0)
        self.assertFalse(meta['l1_projection_active'])
        np.testing.assert_allclose(M@beta,h,atol=1e-14)
        np.testing.assert_allclose(beta,np.linalg.solve(M,h),atol=1e-14)

    def test_threshold_projection_retained_and_invalid_rejected(self):
        vals=np.array([0.,1.,2.]); Q=np.eye(3); h=np.array([100.,-40.,.1])
        beta,meta=ridge.ridge_from_eigh(vals,Q,h,.02,.1,1.)
        lam=math.sqrt(3)*.1; raw=h/(vals+lam)
        expected=core.project_l1_ball(np.sign(raw)*np.maximum(abs(raw)-.02,0.),1.)
        np.testing.assert_array_equal(beta,expected)
        self.assertLessEqual(np.linalg.norm(beta,1),1+1e-12)
        for bad in (0.,-1.,float('nan'),float('inf')):
            with self.assertRaises(ValueError):ridge.ridge_from_eigh(vals,Q,h,.02,.1,bad)

    def test_adapter_preserves_default_and_explicit_variant(self):
        rng=np.random.default_rng(63); X=rng.normal(size=(400,4)); y=X[:,0]*.3+rng.normal(size=400)*.1
        meta=dict(numeric_dim=4,categorical_low=[],categorical_high=[])
        p=dict(clip_constant=.1,threshold_constant=1.)
        with tempfile.TemporaryDirectory() as tmp:
            a,old=fit_sample(X,y,4.,1e-6,p,41,tmp,'small',meta)
            b,explicit=fit_sample(X,y,4.,1e-6,{**p,'solver':'appendix_direct'},41,tmp,'small',meta)
            c,new=fit_sample(X,y,4.,1e-6,{**p,'solver':'ridge','ridge_noise_multiplier':1.},41,tmp,'small',meta)
        np.testing.assert_array_equal(a,b)
        self.assertEqual(old['privacy_certificate'],new['privacy_certificate'])
        self.assertEqual(new['estimator_variant'],'SampleSDP-Ridge')
        self.assertTrue(np.isfinite(c).all())
        self.assertLessEqual(np.linalg.norm(c,1),1+1e-12)
        self.assertEqual(verify(),'SDP_TABLE2_TABLE5_ALIGNED_20261021')


if __name__=='__main__':
    with threadpool_limits(limits=2): unittest.main()
