'Independent paper-table regressions, including parameter plumbing.'
import math
import unittest
from unittest.mock import patch
from dataclasses import asdict

import numpy as np

from lasso_threshold_v1.mechanism import proposed
from lasso_threshold_v1.mechanism.appendix import aggregation_parameters
from lasso_threshold_v1.mechanism.core import StageBudget, RandomStreams
from paper_suite.algorithms.sample_sdp_core import (
    Config, sdp2_table_parameters, fit_sdp2_from_sum)
from paper_suite.algorithms.sample_sdp import fit_sample_sdp


class PaperFormulaCorrections(unittest.TestCase):
    def test_table5_default_anchor_matches_independent_paper_calculation(self):

        radius = 0.31665925835169445
        p = aggregation_parameters(3750, 8, radius, 15000, StageBudget(4., 1e-5), 1.)
        self.assertEqual(p.precision, 707)
        self.assertEqual(p.local_trials, 1700)
        self.assertAlmostEqual(p.integer_sensitivity, 461.0698998082807)
        self.assertAlmostEqual(math.sqrt(p.total_trials)/(2*3750*p.precision),
                               0.0004761671352363073)
        self.assertLessEqual(p.privacy_certificate['epsilon_noise_certified'], 4.)
        self.assertLessEqual(p.privacy_certificate['delta_total_bound'], 1e-5)

    def test_second_precision_bound_is_enforced_when_it_dominates(self):
        for epsilon in (.5, 4., 32.):
            with self.subTest(epsilon=epsilon):
                s, radius, ng = 64, .7, 19
                p=aggregation_parameters(ng,s,radius,76,StageBudget(epsilon,1e-5),1.)
                lower=(math.sqrt(s)+2)*max(1,epsilon)/radius
                self.assertGreaterEqual(p.precision,lower)
                self.assertLessEqual(p.integer_sensitivity,3*p.precision*radius+1e-12)
                expected=2*p.precision*radius+max(1,epsilon)*math.sqrt(s)+2
                self.assertAlmostEqual(p.integer_sensitivity,expected)
                expected_M=math.ceil(32*expected**2*math.log(4*s/1e-5)/epsilon**2)
                self.assertEqual(p.requested_trials,expected_M)
                self.assertGreaterEqual(p.local_trials,2*math.ceil(expected_M/(2*ng)))
                paper_shares=math.ceil((2*(math.log2(8*s/1e-5)+epsilon/math.log(2))+
                                       math.log2(p.modulus))/(math.log2(ng)-math.log2(math.e)))+1
                self.assertGreaterEqual(p.shares,paper_shares)

    def test_actual_u2r_fit_uses_table5_fine_radius(self):
        n,m,s=15000,50,8
        x=np.broadcast_to(np.zeros((1,1,s)),(n,m,s))
        y=np.broadcast_to(np.zeros((1,1)),(n,m))
        groups=proposed.partition(n)
        captured=[]
        def local(a,b,selected,return_diagnostics=True):
            return np.zeros((len(a),len(selected))),{'local_ols_fallback_count':0}
        def mean(values,center,params,streams,stage,**kwargs):
            captured.append(params)
            return np.zeros(s),{'parameters':asdict(params)}
        with patch.object(proposed,'local_ols_coefficients_private_totalized',side_effect=local), \
             patch.object(proposed,'distributed_private_mean_ikos',side_effect=mean):
            beta,meta=proposed._fit_after_selection(x,y,4.,np.arange(s),groups,RandomStreams(7),
                parameter_l2_norm=1/math.sqrt(s),private_release=False,meta={})
        self.assertEqual(len(captured),2)
        self.assertAlmostEqual(meta['fine_radius'],0.31665925835169445)
        self.assertEqual(captured[1].precision,707)
        self.assertTrue(np.isfinite(beta).all())

    def test_table2_default_threshold_and_statistical_floor(self):
        t=sdp2_table_parameters(750000,256,4.,1e-5)
        self.assertAlmostEqual(t['soft_threshold'],0.004955654463979503)
        self.assertAlmostEqual(t['threshold_statistical_component'],
                               2*.25*math.sqrt(math.log(256)/750000))
        self.assertAlmostEqual(t['soft_threshold'],
                               t['threshold_statistical_component']+t['threshold_private_component'])
        zero=sdp2_table_parameters(750000,256,4.,1e-5,noise_scale=0.)
        half=sdp2_table_parameters(750000,256,4.,1e-5,kappa_infinity=2.)
        self.assertAlmostEqual(t['soft_threshold']-zero['soft_threshold'],
                               t['threshold_statistical_component'])
        self.assertAlmostEqual(half['soft_threshold'],t['soft_threshold']/2)
        for key in ('precision','local_trials','modulus','shuffled_shares','privacy_certificate'):
            self.assertEqual(t[key],zero[key])
            self.assertEqual(t[key],half[key])

    def test_table2_uses_correct_dimension_and_sample_count(self):
        n,d,e,delta,c0,c1,sigma,kappa=1000,7,2.,1e-5,.03,1.5,.7,.4
        D=d*(d+1)//2+d
        expected=c1/kappa*(sigma*math.sqrt(math.log(d)/n)+
            c0*d*math.log(n)*math.sqrt(math.log(4*D/delta)*math.log(D))/(n*e))
        t=sdp2_table_parameters(n,d,e,delta,clip_constant=c0,threshold_constant=c1,
                               noise_scale=sigma,kappa_infinity=kappa)
        self.assertAlmostEqual(t['soft_threshold'],expected)

    def test_public_threshold_inputs_reject_invalid_values(self):
        for sigma in (-1.,float('nan'),float('inf')):
            with self.assertRaises(ValueError):sdp2_table_parameters(1000,8,4.,1e-5,noise_scale=sigma)
        for kappa in (0.,-1.,float('nan'),float('inf')):
            with self.assertRaises(ValueError):sdp2_table_parameters(1000,8,4.,1e-5,kappa_infinity=kappa)

    def test_config_and_wrapper_forward_nondefault_public_scales(self):
        n,d=200,4
        t=sdp2_table_parameters(n,d,4.,1e-5,noise_scale=.7,kappa_infinity=.4)
        config=Config(sdp2_dimension=d,sdp2_clip=t['clip'],sdp2_noise_scale=.7,sdp2_kappa_infinity=.4)
        meta={'n':n,'d':d,'D':t['D']}
        _,details=fit_sdp2_from_sum(np.zeros(t['D']),meta,4.,config,
                                   np.random.default_rng(2),np.zeros(t['D'],dtype=np.int64))
        self.assertEqual(details['soft_threshold'],t['soft_threshold'])
        x=np.zeros((40,5,d));y=np.zeros((40,5))
        _,details=fit_sample_sdp(x,y,4.,9,noise_scale=.7,kappa_infinity=.4)
        self.assertEqual(details['soft_threshold'],t['soft_threshold'])
        self.assertEqual(details['threshold_noise_scale'],.7)
        self.assertEqual(details['threshold_kappa_infinity'],.4)


if __name__=='__main__':unittest.main()
