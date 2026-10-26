import unittest
import numpy as np
from scipy.stats import binom
from paper_suite.algorithms.sample_sdp_core import (exact_rounded_statistic_sums,
    clipped_sufficient_statistic_sum,prepare_exact_sample_statistics,
    exact_aggregate_binomial_noise_sum)


class SampleFastTests(unittest.TestCase):
    def test_fused_pass_preserves_seeded_rounding_and_clipping(self):
        rng=np.random.default_rng(341)
        x=rng.normal(size=(137,9));y=rng.normal(size=137)
        for clip in [.5,100.]:
            for precision in [1,7]:
                old=exact_rounded_statistic_sums(x,y,clip,precision,[np.random.default_rng(43)],chunk_size=13)[0]
                new=prepare_exact_sample_statistics(x,y,clip,precision,np.random.default_rng(43),chunk_size=13)
                np.testing.assert_array_equal(old,new['rounded'])
                _,meta=clipped_sufficient_statistic_sum(x,y,clip,chunk_size=13)
                self.assertEqual(meta['clip_rate'],new['meta']['clip_rate'])
                np.testing.assert_allclose(meta['raw_norm_percentiles'],new['meta']['raw_norm_percentiles'],rtol=1e-14)

    def test_exact_convolution_law(self):
        local=binom.pmf(np.arange(7),6,.5)
        combined=np.convolve(np.convolve(local,local),local)
        np.testing.assert_allclose(combined,binom.pmf(np.arange(19),18,.5),atol=1e-15)

    def test_sampler_center_and_variance(self):
        draw=exact_aggregate_binomial_noise_sum(31,50000,14,np.random.default_rng(41))
        self.assertLess(abs(draw.mean()),.2)
        self.assertLess(abs(draw.var()/(31*14/4)-1),.035)
        self.assertTrue(np.issubdtype(draw.dtype,np.integer))

    def test_total_trial_overflow_guard(self):
        with self.assertRaises(ValueError):
            exact_aggregate_binomial_noise_sum(2**52,1,4,np.random.default_rng(1))


if __name__=='__main__':unittest.main()
