import unittest
import numpy as np
from paper_suite.algorithms.two_slr_paper import _screen_indices
from paper_suite.algorithms.iht_selected_profile import selected_parameters,profile_name
from paper_suite.algorithms.ldp_iht import paper_parameter_grid,ma24_empirical_parameter_grid


class SourceAlignedBaselineTests(unittest.TestCase):
    def test_two_slr_screening_uses_positive_marginal_correlation(self):
        x=np.eye(8);y=np.array([-1.,.8,.01,.02,.03,.04,.05,.06])
        indexes=_screen_indices(x,y,2)
        np.testing.assert_array_equal(indexes,[1,7])

    def test_iht_selected_profile_is_fixed_ma24_grid_member(self):
        p=selected_parameters(8,750000)
        self.assertEqual(p,dict(iterations=2,step_size=.01,x_clip=2.,y_clip=2.,sparsity=10,vectorized=True))
        self.assertIn('ma24_grid_fixed',profile_name(8))
        self.assertIn(p,ma24_empirical_parameter_grid())

    def test_ma24_empirical_grid_matches_reported_cartesian_product(self):
        grid=ma24_empirical_parameter_grid()
        self.assertEqual(len(grid),5*3*3*3*4)
        self.assertEqual({g['iterations'] for g in grid},{2,5,10,20,50})
        self.assertEqual({g['step_size'] for g in grid},{.01,.1,1.})
        self.assertEqual({g['x_clip'] for g in grid},{2.,4.,8.})
        self.assertEqual({g['y_clip'] for g in grid},{2.,4.,8.})
        self.assertEqual({g['sparsity'] for g in grid},{5,10,20,50})

    def test_theorem_reference_validation_grid_still_available(self):
        grid=paper_parameter_grid(40000,8)
        self.assertTrue(grid)
        self.assertEqual({g['sparsity'] for g in grid},{64})
        self.assertTrue(all(g['x_clip']==g['y_clip'] for g in grid))

if __name__=='__main__':unittest.main()
