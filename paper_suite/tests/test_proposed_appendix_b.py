import math
import unittest
import numpy as np
from paper_suite.config import SyntheticSetting
from paper_suite.data import make_synthetic
from paper_suite.runner import run_algorithm,run_recall
from lasso_threshold_v1.dataset import coefficient_values,public_signal_lower_bound,public_parameter_l2_norm
from lasso_threshold_v1.mechanism.core import local_ols_coefficients_appendix_b, StageBudget
from lasso_threshold_v1.mechanism.appendix import selection_parameters, aggregation_parameters, C_GAMMA
from lasso_threshold_v1.mechanism.privacy_accountant import certify_stage1_gamma

class ProposedAppendixBTests(unittest.TestCase):
    def test_synthetic_public_constants_do_not_read_support(self):
        k=8
        self.assertEqual(public_signal_lower_bound(k),1/k)
        self.assertAlmostEqual(public_parameter_l2_norm(k),np.linalg.norm(coefficient_values(k)))

    def test_runner_uses_public_rho_and_r0_formula(self):
        st=SyntheticSetting(n=512,m=50,d=16,k=2,epsilon=4.)
        data=make_synthetic(st,123,test_samples=10)
        beta,meta=run_algorithm('proposed',data,st,456)
        self.assertEqual(meta['rho'],public_signal_lower_bound(2))
        self.assertEqual(meta['coefficient_threshold'],meta['rho']/2)
        self.assertAlmostEqual(meta['coarse_radius'],public_parameter_l2_norm(2)+meta['tau_beta'])
        self.assertIn('Appendix-B exact normal-equation inverse',meta['local_solver'])

    def test_stage1_uses_certified_gamma_not_unproved_constant_alone(self):
        st=SyntheticSetting(n=512,m=20,d=16,k=2,epsilon=4.)
        data=make_synthetic(st,123,test_samples=10)
        _,meta=run_recall('proposed',data,st,456)
        cert=certify_stage1_gamma(st.n//2,st.epsilon,st.delta)
        self.assertAlmostEqual(meta['gamma'],cert.gamma)
        self.assertLessEqual(meta['stage1_privacy_certificate']['certified_total_epsilon_bound'],st.epsilon)
        self.assertLessEqual(meta['stage1_privacy_certificate']['certified_total_delta_bound'],st.delta)


    def test_frozen_main_protocol_constants(self):
        self.assertEqual(C_GAMMA,32.0)
        st=SyntheticSetting(epsilon=.5)
        self.assertEqual(st.n,15000)
        table=selection_parameters(st.n,st.m,st.d,st.k,st.epsilon,st.delta,n1=7500,rho=1/st.k)
        self.assertLess(table['gamma'],.5)
        self.assertGreaterEqual(table['gamma'],32*math.log(1/st.delta)/(7500*st.epsilon**2))
        self.assertIn('stage1_privacy_certificate',table)

    def test_rho_is_required_not_silently_fallback(self):
        from lasso_threshold_v1.mechanism.proposed import select_support
        x=np.zeros((60,2,3)); y=np.zeros((60,2))
        with self.assertRaisesRegex(ValueError,'explicit public signal-strength'):
            select_support(x,y,4,1,7)


    def test_stage1_count_optimization_matches_message_reference(self):
        from lasso_threshold_v1.mechanism.proposed import select_support,partition,PROTOCOL
        from lasso_threshold_v1.mechanism.local import candidates
        from lasso_threshold_v1.mechanism.core import RandomStreams,selection_client,aggregate_selection_messages
        st=SyntheticSetting(n=128,m=10,d=8,k=2,epsilon=4.)
        data=make_synthetic(st,31,test_samples=2)
        seed=91; rho=public_signal_lower_bound(st.k)
        selected,meta=select_support(data.user_x,data.user_y,st.epsilon,st.k,seed,rho=rho)
        first,_,_=partition(st.n,seed)
        supports,_=candidates(data.user_x[first],data.user_y[first],st.k,total_users=st.n,rho=rho,
                              c_lambda=PROTOCOL.c_lambda,sigma=PROTOCOL.sigma)
        streams=RandomStreams(seed); pick=streams.rng('selection/pick'); rr=streams.rng('selection/rr')
        messages=[]
        for support in supports:
            reports,_,_=selection_client(support,st.d,meta['gamma'],pick,rr)
            messages.extend(reports)
        counts=aggregate_selection_messages(messages,len(first),st.d)
        np.testing.assert_array_equal(meta['rr_counts'],counts)
        expected=np.flatnonzero((counts-len(first)*meta['gamma'])/(1-2*meta['gamma']) >= len(first)/(4*st.k))
        np.testing.assert_array_equal(selected,expected)


    def test_all_formal_stage2_table5_parameters_cover_requested_trials(self):
        from paper_suite.config import synthetic_grid
        for experiment in ('exp1_epsilon','exp2_dimension','exp3_users','exp4a_recall_m','exp4b_recall_n'):
            for st in synthetic_grid(experiment):
                n1=st.n//2; n2=st.n-n1; n3=n2//2; n4=n2-n3; s=st.k
                tau_beta=st.noise_std*math.sqrt(s*math.log(st.n)/st.m)
                r0=public_parameter_l2_norm(s)+tau_beta
                budget=StageBudget(st.epsilon,st.delta)
                coarse=aggregation_parameters(n3,s,r0,st.n,budget,1.0)
                radius=(tau_beta + st.noise_std*math.sqrt(s*math.log(st.n)/(n3*st.m)) +
                        r0*math.sqrt(s*math.log(st.n)*math.log(4*s/st.delta))/(n3*st.epsilon))
                fine=aggregation_parameters(n4,s,radius,st.n,budget,1.0)
                self.assertGreaterEqual(coarse.total_trials,coarse.requested_trials)
                self.assertGreaterEqual(fine.total_trials,fine.requested_trials)
                self.assertEqual(coarse.local_trials%2,0)
                self.assertEqual(fine.local_trials%2,0)

    def test_algorithm4_has_no_pseudoinverse_fallback(self):
        x=np.ones((1,2,3));y=np.ones((1,2))
        with self.assertRaisesRegex(ValueError,'support size'):
            local_ols_coefficients_appendix_b(x,y,[0,1,2])

if __name__=='__main__':unittest.main()
