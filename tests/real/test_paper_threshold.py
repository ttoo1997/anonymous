import math
import tempfile
import unittest
import numpy as np
from exp5 import common
from exp5.algorithms import fit_sample, default_profile
from exp5 import sample_ridge
from paper_suite.algorithms import sample_sdp_core as core


class RealThresholdCorrection(unittest.TestCase):
    def test_defaults_preserve_accepted_final_profiles(self):
        self.assertEqual(default_profile('u2r'),dict(k=12,rho=.05,norm=1.,support_cap_policy='min_k_m_minus_2'))
        p=default_profile('sample_sdp')
        self.assertEqual((p['clip_constant'],p['threshold_constant'],p['solver'],p['ridge_noise_multiplier']),(.03,1.,'ridge',4.))
        self.assertEqual((p['threshold_noise_scale'],p['threshold_kappa_infinity']),(.25,1.))
        self.assertEqual(default_profile('2slr')['num_bins'],16)
        self.assertEqual(default_profile('ldpiht')['gradient_clip'],4.)

    def test_real_direct_and_ridge_use_explicit_public_scales(self):
        rng=np.random.default_rng(51)
        X=rng.normal(size=(400,4));y=X[:,0]*.3+rng.normal(size=400)*.1
        p={'clip_constant':.03,'threshold_constant':1.,
           'threshold_noise_scale':.7,'threshold_kappa_infinity':.4,
           'threshold_scale_source':'public_reference_test'}
        data={'numeric_dim':4,'categorical_low':[],'categorical_high':[]}
        t=core.sdp2_table_parameters(400,4,4.,1e-5,clip_constant=.03,threshold_constant=1.,
                                   noise_scale=.7,kappa_infinity=.4)
        with tempfile.TemporaryDirectory() as tmp:
            _,direct=fit_sample(X,y,4.,1e-5,p,9,tmp,'same-data',data)
            _,ridge=fit_sample(X,y,4.,1e-5,{**p,'solver':'ridge','ridge_noise_multiplier':4.},9,tmp,'same-data',data)
        for result in (direct,ridge):
            self.assertEqual(result['soft_threshold'],t['soft_threshold'])
            self.assertEqual(result['threshold_scale_source'],'public_reference_test')
            self.assertFalse(result['threshold_curvature_bound_verified'])
            self.assertEqual(result['threshold_statistical_component'],t['threshold_statistical_component'])
        self.assertEqual(direct['privacy_certificate'],ridge['privacy_certificate'])

    def test_threshold_scales_do_not_change_private_statistics(self):
        n,d=200,7
        t=core.sdp2_table_parameters(n,d,4.,1e-5)
        rounded=np.zeros(t['D'],dtype=np.int64)
        meta={'n':n,'d':d,'D':t['D']}
        a=core.Config(sdp2_dimension=d,sdp2_clip=t['clip'])
        b=core.Config(sdp2_dimension=d,sdp2_clip=t['clip'],sdp2_noise_scale=.7,sdp2_kappa_infinity=.4)
        G1,h1,m1=sample_ridge.release_statistics(rounded,meta,4.,a,np.random.default_rng(9))
        G2,h2,m2=sample_ridge.release_statistics(rounded,meta,4.,b,np.random.default_rng(9))
        np.testing.assert_array_equal(G1,G2)
        np.testing.assert_array_equal(h1,h2)
        self.assertEqual(m1['privacy_certificate'],m2['privacy_certificate'])
        self.assertNotEqual(m1['soft_threshold'],m2['soft_threshold'])


if __name__=='__main__':unittest.main()
