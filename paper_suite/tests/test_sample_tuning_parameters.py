import unittest
import numpy as np
from paper_suite.algorithms.sample_sdp_core import sdp2_table_parameters
from paper_suite.algorithms.sample_sdp import fit_sample_sdp

class SampleTuningTests(unittest.TestCase):
    def test_clipping_change_recalibrates_noise_and_threshold(self):
        old=sdp2_table_parameters(40000,256,4,1e-6,clip_constant=2.0)
        new=sdp2_table_parameters(40000,256,4,1e-6,clip_constant=.1)
        self.assertAlmostEqual(new['clip']/old['clip'],.05)
        self.assertAlmostEqual(new['threshold_private_component']/old['threshold_private_component'],.05)
        self.assertEqual(new['threshold_statistical_component'],old['threshold_statistical_component'])
        self.assertGreater(new['soft_threshold']/old['soft_threshold'],.05)
        self.assertLess(new['integer_sensitivity'],old['integer_sensitivity'])
        self.assertEqual(new['local_trials']%2,0)
        self.assertGreaterEqual(new['total_trials'],new['requested_trials'])
        self.assertGreater(new['total_trials'],0)
    def test_revised_default_is_validated_c0_without_stabilization(self):
        table=sdp2_table_parameters(40000,256,4,1e-6)
        self.assertAlmostEqual(table['clip'],.1*256*np.log(40000))

    def test_stabilization_is_disabled(self):
        rng=np.random.default_rng(11);x=rng.normal(size=(40,5,4));y=rng.normal(size=(40,5))
        with self.assertRaises(TypeError):
            fit_sample_sdp(x,y,4.,7,clip_constant=.05,eigen_floor=.5)
    def test_nondefault_clip_uses_direct_solver_and_private_noise(self):
        rng=np.random.default_rng(11);x=rng.normal(size=(40,5,4));y=rng.normal(size=(40,5))
        beta,meta=fit_sample_sdp(x,y,4.,7,clip_constant=.1)
        self.assertTrue(np.isfinite(beta).all())
        self.assertEqual(meta['solver'],'appendix_direct_solve')
        self.assertEqual(meta['clip_constant'],.1)
        self.assertGreater(meta['coordinate_noise_sd'],0)
        self.assertLessEqual(np.linalg.norm(beta,1),1+1e-10)
        self.assertIn('raw_beta_l2',meta)
        self.assertIn('l1_projection_active',meta)

if __name__=='__main__':unittest.main()
