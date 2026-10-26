import unittest
import numpy as np

from paper_suite.algorithms.two_slr_author import fit as fit_author
from paper_suite.algorithms.two_slr_paper import (
    PAPER_B_GRID,
    PAPER_BIN_GRID,
    PAPER_LOCAL_SELECTED_GRID,
    FROZEN_HH_ALPHA,
    PROTOCOL,
    _mean_private,
    paper_parameter_grid,
    select_support,
)
from paper_suite.vendor.ma2024.ULDPFS import ULDPFS


class Ma24SourceAndPaperAdapterTests(unittest.TestCase):
    def test_vendored_author_source_remains_reproducible(self):
        rng=np.random.default_rng(8)
        x=rng.normal(size=(120,30,8));y=x[:,:,0]*.2+rng.normal(size=(120,30))
        np.random.seed(6);expected_next=np.random.random()
        np.random.seed(6);beta,meta=fit_author(x,y,4,2,17)
        self.assertEqual(np.random.random(),expected_next)
        np.random.seed(17);model=ULDPFS(**meta['parameters']).fit(x,y)
        np.testing.assert_array_equal(beta,model.coef_)

    def test_main_entry_uses_paper_aligned_adapter(self):
        from paper_suite.runner import run_algorithm
        from paper_suite.config import SyntheticSetting
        from paper_suite.data import make_synthetic
        setting=SyntheticSetting(n=80,m=10,d=8,k=2)
        data=make_synthetic(setting,8,test_samples=5)
        _,meta=run_algorithm('2slr',data,setting,17)
        self.assertEqual(meta['protocol_version'],PROTOCOL)
        self.assertTrue(meta['hh_one_report_per_user'])
        self.assertTrue(meta['stage2_groups_disjoint'])
        self.assertEqual(meta['hh_report_draws_max'],1)

    def test_appendix_e_grid_is_exposed_without_hidden_validation_grid(self):
        grid=paper_parameter_grid(8,256)
        self.assertEqual(len(grid),len(PAPER_LOCAL_SELECTED_GRID)*len(PAPER_B_GRID)*len(PAPER_BIN_GRID))
        self.assertEqual({p.local_selected for p in grid},set(PAPER_LOCAL_SELECTED_GRID))
        self.assertEqual({p.B for p in grid},set(PAPER_B_GRID))
        self.assertEqual({p.num_bins for p in grid},set(PAPER_BIN_GRID))
        self.assertEqual({p.screening_features for p in grid},{64})
        self.assertEqual({p.hh_target for p in grid},{8})
        self.assertEqual({p.hh_alpha for p in grid},{FROZEN_HH_ALPHA})

    def test_algorithm8_projects_before_laplace(self):
        rng=np.random.default_rng(123)
        result=_mean_private(np.array([100.,100.,100.]),(-1.,1.),1e9,rng)
        self.assertAlmostEqual(result,1.0,places=6)

    def test_stage1_is_deterministic_for_seed_and_uses_one_report(self):
        rng=np.random.default_rng(4)
        x=rng.normal(size=(80,10,8));y=.2*x[:,:,0]+.2*x[:,:,1]+rng.normal(size=(80,10))
        a,ma=select_support(x,y,4,2,91)
        b,mb=select_support(x,y,4,2,91)
        np.testing.assert_array_equal(a,b)
        self.assertEqual(ma['hh_report_draws_max'],1)
        self.assertAlmostEqual(ma['hh_alpha'],FROZEN_HH_ALPHA)
        self.assertTrue(ma['hh_one_report_per_user'])
        self.assertEqual(ma['hh_report_draws_mean'],mb['hh_report_draws_mean'])


if __name__=='__main__':unittest.main()
