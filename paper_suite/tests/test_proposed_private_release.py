import inspect
import unittest
from unittest.mock import patch

import numpy as np

from paper_suite.config import SyntheticSetting
from paper_suite.data import make_synthetic
from lasso_threshold_v1.dataset import public_parameter_l2_norm, public_signal_lower_bound
from lasso_threshold_v1.mechanism.proposed import (
    PRIVATE_FORBIDDEN_METADATA_KEYS,
    _select_support_impl,
    fit_private,
)


def walk_keys(value):
    if isinstance(value, dict):
        for key, item in value.items():
            yield str(key)
            yield from walk_keys(item)
    elif isinstance(value, (list, tuple)):
        for item in value:
            yield from walk_keys(item)


class ProposedPrivateReleaseTests(unittest.TestCase):
    def _fixture(self):
        st=SyntheticSetting(n=128,m=10,d=8,k=2,epsilon=16.)
        return st,make_synthetic(st,31,test_samples=2)

    def test_private_api_accepts_no_seed_and_exports_no_secret_or_local_diagnostics(self):
        self.assertNotIn('seed',inspect.signature(fit_private).parameters)
        st,data=self._fixture()
        with patch('lasso_threshold_v1.mechanism.proposed.secrets.randbits',side_effect=[123456789,987654321]) as entropy:
            beta,meta=fit_private(
                data.user_x,data.user_y,st.epsilon,st.k,
                rho=public_signal_lower_bound(st.k),
                parameter_l2_norm=public_parameter_l2_norm(st.k),
            )
        self.assertEqual(entropy.call_count,2)
        self.assertTrue(np.isfinite(beta).all())
        self.assertTrue(meta['private_release_safe'])
        self.assertFalse(meta['seed_exported'])
        self.assertFalse(meta['transcript_exported'])
        keys=set(walk_keys(meta))
        self.assertFalse(keys & PRIVATE_FORBIDDEN_METADATA_KEYS, keys & PRIVATE_FORBIDDEN_METADATA_KEYS)

    def test_private_transport_is_materialized(self):
        st,data=self._fixture()
        with patch('lasso_threshold_v1.mechanism.proposed.secrets.randbits',side_effect=[111,222]):
            _,meta=fit_private(
                data.user_x,data.user_y,st.epsilon,st.k,
                rho=public_signal_lower_bound(st.k),
                parameter_l2_norm=public_parameter_l2_norm(st.k),
            )
        self.assertEqual(meta['selection_transport'],'materialized_parallel_coordinate_shufflers')
        self.assertTrue(meta['transport_materialized'])
        if meta['algorithm_status']=='ok':
            self.assertEqual(meta['coarse']['protocol'],'materialized_parallel_ikos')
            self.assertEqual(meta['fine']['protocol'],'materialized_parallel_ikos')
            self.assertGreater(meta['coarse']['shuffled_messages'],0)
            self.assertGreater(meta['fine']['shuffled_messages'],0)

    def test_materialized_stage1_has_same_analyzer_output_as_fast_count_simulation(self):
        st,data=self._fixture()
        rho=public_signal_lower_bound(st.k)
        fast,_,_,_=_select_support_impl(
            data.user_x,data.user_y,st.epsilon,st.k,123,456,
            rho=rho,private_release=False,materialize_parallel_shuffle=False,
        )
        materialized,_,_,_=_select_support_impl(
            data.user_x,data.user_y,st.epsilon,st.k,123,456,
            rho=rho,private_release=False,materialize_parallel_shuffle=True,
        )
        np.testing.assert_array_equal(fast,materialized)


    def test_stage2_materialized_ikos_matches_fast_aggregate_for_same_local_randomness(self):
        from lasso_threshold_v1.mechanism.core import RandomStreams,StageBudget,distributed_private_mean_ikos
        from lasso_threshold_v1.mechanism.appendix import aggregation_parameters
        rng=np.random.default_rng(7)
        values=rng.normal(size=(32,2))
        center=np.array([.1,-.2])
        params=aggregation_parameters(32,2,.8,128,StageBudget(8.,1e-6),1.0)
        fast,_=distributed_private_mean_ikos(values,center,params,RandomStreams(77,88),'fixture',materialize_ikos=False)
        materialized,_=distributed_private_mean_ikos(values,center,params,RandomStreams(77,88),'fixture',materialize_ikos=True)
        np.testing.assert_array_equal(fast,materialized)

    def test_private_path_totalizes_singular_local_ols_instead_of_leaking_failure(self):


        x=np.zeros((128,10,8),dtype=float)
        y=np.ones((128,10),dtype=float)
        with patch('lasso_threshold_v1.mechanism.proposed.secrets.randbits',side_effect=[333,444]):
            beta,meta=fit_private(x,y,16.,2,rho=.5,parameter_l2_norm=2**-0.5)
        self.assertTrue(np.isfinite(beta).all())
        self.assertIn(meta['algorithm_status'],('ok','empty_global_support'))
        self.assertTrue(meta['private_release_safe'])

    def test_private_stage1_certificate_still_covers_target(self):
        st,data=self._fixture()
        with patch('lasso_threshold_v1.mechanism.proposed.secrets.randbits',side_effect=[555,666]):
            _,meta=fit_private(
                data.user_x,data.user_y,st.epsilon,st.k,
                rho=public_signal_lower_bound(st.k),
                parameter_l2_norm=public_parameter_l2_norm(st.k),
            )
        cert=meta['stage1_privacy_certificate']
        self.assertLessEqual(cert['certified_total_epsilon_bound'],st.epsilon*(1+1e-9))
        self.assertLessEqual(cert['certified_total_delta_bound'],st.delta)


if __name__=='__main__':
    unittest.main()
