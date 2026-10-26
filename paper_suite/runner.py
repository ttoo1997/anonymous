from __future__ import annotations

import json
from .checksum import checksum_file
import math
from pathlib import Path

import numpy as np
import pandas as pd

from .algorithms.sample_sdp import fit_sample_sdp
from .algorithms.nonprivate_lasso import fit_lasso
from .algorithms.iht_stabilized_profile import stabilized_parameters, PROFILE as IHT_PROFILE
from .algorithms.ldp_iht import fit_ldp_iht, paper_reference_parameters
from .algorithms.two_slr_paper import fit as fit_paper_two_slr, select_support as select_paper_two_slr
from lasso_threshold_v1.mechanism.proposed import fit as fit_pdf_proposed,select_support as select_pdf_proposed,PROTOCOL as PROPOSED_PROTOCOL
from lasso_threshold_v1.dataset import public_signal_lower_bound, public_parameter_l2_norm, public_parameter_l1_norm
from .config import (SyntheticSetting, RealSetting, SAMPLE_SDP_CLIP_CONSTANT,
    SAMPLE_SDP_THRESHOLD_CONSTANT)
from .data import RegressionData
from .metrics import mse, relative_coefficient_error


DEFAULT_ALGORITHMS = ("proposed", "2slr", "ldpiht", "sample_sdp", "lasso")
ALGORITHMS = DEFAULT_ALGORITHMS


def run_algorithm(name: str, data: RegressionData, setting: SyntheticSetting | RealSetting,
                  seed: int, sample_preparation=None, ldpiht_parameters=None) -> tuple[np.ndarray, dict]:
    setting.validate()
    if data.user_x.shape != (setting.n, setting.m, setting.d) or data.user_y.shape != (setting.n, setting.m):
        raise ValueError("data dimensions disagree with setting")
    if setting.delta != PROPOSED_PROTOCOL.delta:
        raise ValueError("setting delta disagrees with the active mechanism protocol")
    if name == "proposed":
        if (data.metadata or {}).get("coefficient_protocol") == "equal_positive_l1_v1":
            if not isinstance(setting, SyntheticSetting):
                raise AssertionError("synthetic Proposed data require SyntheticSetting")
            if not math.isclose(setting.noise_std, PROPOSED_PROTOCOL.sigma, rel_tol=0.0, abs_tol=0.0):
                raise ValueError('Proposed protocol requires sigma to equal the frozen synthetic noise_std=0.25')
            rho=public_signal_lower_bound(setting.k)
            l2norm=public_parameter_l2_norm(setting.k)
            beta,meta=fit_pdf_proposed(data.user_x,data.user_y,setting.epsilon,setting.k,seed,
                                       rho=rho,parameter_l2_norm=l2norm)
            return beta,{**meta,'public_parameter_source':'equal_positive_l1_v1 coefficient protocol',
                         'public_signal_lower_bound':rho}
        raise NotImplementedError('Appendix-B Proposed requires public rho and ||theta*||_2; real-data constants are not specified by the current source')
    if name == "2slr":
        regime = "synthetic_positive" if (data.metadata or {}).get("coefficient_protocol") == "equal_positive_l1_v1" else "real_standardized"
        return fit_paper_two_slr(data.user_x,data.user_y,setting.epsilon,setting.k,seed, data_regime=regime)
    if name == "ldpiht":
        x = data.user_x.reshape(-1, setting.d)
        y = data.user_y.ravel()
        is_frozen_synthetic=(data.metadata or {}).get("coefficient_protocol")=="equal_positive_l1_v1"
        parameters=(stabilized_parameters(setting.k, len(x), setting.epsilon) if is_frozen_synthetic else paper_reference_parameters(len(x), setting.k)) if ldpiht_parameters is None else dict(ldpiht_parameters)
        if parameters.get("private",True) is not True:
            raise ValueError("nonprivate IHT is diagnostic only; not a formal baseline")
        beta, meta = fit_ldp_iht(x, y, setting.epsilon, setting.k, seed, **parameters)
        return beta, {**meta,
                      "parameter_profile":(IHT_PROFILE if is_frozen_synthetic else "zhu24_theorem7_reference_real_standardized_exploratory") if ldpiht_parameters is None else "explicit_override",
                      "scale_adaptation":(
                          "none; synthetic target already lies in Zhu24 unit-L2 ball"
                          if is_frozen_synthetic else
                          "Exp5 loader z-scores features and target using the training split; no private-data adaptive rescaling is performed inside LDP-IHT"
                      ),
                      "study_mapping": "all n*m rows as sample-level LDP participants",
                      "theory_scope":(
                          "Zhu24 IHT framework with explicit gradient clipping; conditional sample-LDP channel retained; original utility theorem not claimed"
                          if is_frozen_synthetic else
                          "Zhu24 Algorithm 2 on standardized real rows; exploratory real-data mapping rather than a theorem-assumption claim"
                      ),
                      "data_regime":"synthetic_equal_positive_l1" if is_frozen_synthetic else "real_standardized",
                      "unused_rows_per_user": 0}
    if name == "sample_sdp":
        if (data.metadata or {}).get("coefficient_protocol") != "equal_positive_l1_v1":
            raise NotImplementedError("Formal Sample SDP requires a public unit-L1 parameter bound; no arbitrary real-data response rescaling is used in the frozen suite")
        if not math.isclose(public_parameter_l1_norm(setting.k),1.0,rel_tol=0.0,abs_tol=1e-12):
            raise AssertionError("frozen synthetic protocol must lie in the unit-L1 ball")
        beta,meta=fit_sample_sdp(data.user_x, data.user_y, setting.epsilon, seed,
            prepared=sample_preparation, delta=setting.delta, clip_constant=SAMPLE_SDP_CLIP_CONSTANT,
            threshold_constant=SAMPLE_SDP_THRESHOLD_CONSTANT,
            noise_scale=setting.noise_std,kappa_infinity=1.0)
        return beta,{**meta,'public_response_scale':1.0,
            'response_scale_rule':'none; equal_positive_l1_v1 has ||beta*||_1=1',
            'output_l1_bound':1.0,
            'implementation':'Appendix-B Algorithm 2 direct solve; no Ridge/PSD flooring/response rescaling'}
    if name == "lasso":
        return fit_lasso(data)
    raise ValueError(f"unknown algorithm: {name}")


def metric_row(beta: np.ndarray, data: RegressionData) -> dict:
    if beta.shape != (data.user_x.shape[-1],) or not np.isfinite(beta).all():
        raise ValueError("invalid coefficient vector")
    row = {"test_mse": mse(beta, data.x_test, data.y_test),
           "estimated_nonzero": int(np.sum(np.abs(beta) > 1e-8))}
    if data.beta_true is not None:
        row["relative_coefficient_error"] = relative_coefficient_error(beta, data.beta_true)
        actual = np.flatnonzero(data.beta_true)
        estimated = np.flatnonzero(np.abs(beta) > 1e-8)
        tp = len(np.intersect1d(actual, estimated))
        row.update(final_support_recall=tp/len(actual) if len(actual) else 0.,
                   final_support_precision=tp/len(estimated) if len(estimated) else 0.,
                   final_support_f1=2*tp/(len(actual)+len(estimated)) if len(actual)+len(estimated) else 0.)


        topk=len(actual)
        if topk:
            chosen=np.argpartition(np.abs(beta), -topk)[-topk:] if topk < len(beta) else np.arange(len(beta))
            top_tp=len(np.intersect1d(actual, chosen))
            row['topk_support_recall']=top_tp/topk
            row['topk_support_precision']=top_tp/len(chosen) if len(chosen) else 0.
    return row


def metadata_row(meta: dict) -> dict:
    support = meta.get("support")
    return {"selected_count": len(support) if support is not None else None,
            "privacy_validation": meta.get("privacy_validation", "not_end_to_end_certified"),
            "privacy_unit": meta.get("privacy_unit"),
            "private_release_safe": bool(meta.get("private_release_safe", False)),
            "implementation_variant": meta.get("implementation") or meta.get("mechanism")}


def run_recall(name: str, data: RegressionData, setting: SyntheticSetting,
               seed: int) -> tuple[np.ndarray, dict]:
    setting.validate()
    if setting.delta != PROPOSED_PROTOCOL.delta:
        raise ValueError("setting delta disagrees with the active mechanism protocol")


    if name == "proposed":
        if (data.metadata or {}).get("coefficient_protocol") == "equal_positive_l1_v1":
            if not isinstance(setting, SyntheticSetting):
                raise AssertionError("synthetic Proposed data require SyntheticSetting")
            if not math.isclose(setting.noise_std, PROPOSED_PROTOCOL.sigma, rel_tol=0.0, abs_tol=0.0):
                raise ValueError('Proposed protocol requires sigma to equal the frozen synthetic noise_std=0.25')
            rho=public_signal_lower_bound(setting.k)
            return select_pdf_proposed(data.user_x,data.user_y,setting.epsilon,setting.k,seed,rho=rho)
        raise NotImplementedError('Appendix-B Proposed Stage 1 requires a public signal lower bound rho')
    if name == "2slr":
        return select_paper_two_slr(data.user_x,data.user_y,setting.epsilon,setting.k,seed)
    raise ValueError("unsupported Stage-1 recall algorithm")


def write_outputs(out: Path, rows: list[dict], protocol: dict) -> None:
    frame = pd.DataFrame(rows)
    if frame.empty or frame.duplicated([c for c in ("experiment", "dataset", "algorithm", "repeat", "n", "m", "d", "epsilon") if c in frame]).any():
        raise AssertionError("empty results or duplicate result keys")
    numeric = [c for c in ("relative_coefficient_error", "support_recall", "test_mse", "mse_ratio") if c in frame]
    for column in numeric:

        values = frame[column].dropna().to_numpy(dtype=float)
        if not np.isfinite(values).all():
            raise AssertionError(f"non-finite metric: {column}")
    if "status" in frame:
        success = frame[frame.status == "ok"]
        for _, row in success.iterrows():
            metric = "support_recall" if str(row["experiment"]).startswith("exp4") else "test_mse"
            if metric not in row or not np.isfinite(row[metric]):
                raise AssertionError(f"successful row lacks {metric}")
    source_root = Path(__file__).resolve().parent
    sources = list(source_root.rglob("*.py")) + list((source_root.parent/'lasso_threshold_v1').rglob('*.py'))
    protocol = {**protocol, "source_checksums": {
        str(p.relative_to(source_root.parent)): checksum_file(p)
        for p in sources}}
    payload = json.dumps(protocol, ensure_ascii=False, indent=2, allow_nan=False)
    out.mkdir(parents=True, exist_ok=False)
    frame.to_csv(out / "results.csv", index=False)
    (out / "protocol.json").write_text(payload, encoding="utf-8")


def validate_request(algorithms: list[str], repeats: int, out: Path) -> None:
    from .frozen import verify
    verify()
    if repeats < 1 or not algorithms or len(set(algorithms)) != len(algorithms):
        raise ValueError("positive repeats and distinct algorithms are required")
    if any(a not in ALGORITHMS for a in algorithms):
        raise ValueError("unknown algorithm")
    if out.exists():
        raise FileExistsError(out)


def evaluate(name: str, data: RegressionData, setting: SyntheticSetting | RealSetting, seed: int,
             recall_only: bool = False, sample_preparation=None) -> dict:
    'Preserve missing/infeasible cells without inventing numeric scores.'
    from .metrics import support_recall
    try:
        if recall_only:
            support, meta = run_recall(name, data, setting, seed)
            truth=np.flatnonzero(data.beta_true)
            tp=len(np.intersect1d(support,truth)); ss=len(support); kk=len(truth)
            measured = {"support_precision":tp/ss if ss else 0.,
                        "support_f1":2*tp/(ss+kk) if ss+kk else 0.,
                        "exact_support":bool(np.array_equal(np.sort(support),truth)),
                        "support_recall": support_recall(support, data.beta_true),
                        "stage1_support_recall": support_recall(support, data.beta_true),
                        "support_recall_definition": "stage1_selected_set",
                        "privacy_validation":meta.get("privacy_validation", "not_end_to_end_certified"),
                        "selected_count": len(support)}
        else:
            beta, meta = run_algorithm(name, data, setting, seed,sample_preparation=sample_preparation)
            measured = {**metric_row(beta, data), **metadata_row(meta)}

            if data.beta_true is not None:
                measured['support_recall_definition'] = 'not_a_primary_metric; see final_support_* / topk_support_* diagnostics'
                if name in ('proposed','2slr'):
                    selected = np.asarray(meta['support'], dtype=int)
                    measured['stage1_support_recall'] = support_recall(selected, data.beta_true)
                    tp=len(np.intersect1d(selected,np.flatnonzero(data.beta_true)))
                    truth_size=int(np.count_nonzero(data.beta_true))
                    measured['stage1_support_precision']=tp/len(selected) if len(selected) else 0.
                    measured['stage1_support_f1']=2*tp/(len(selected)+truth_size) if len(selected)+truth_size else 0.
                    measured['support_recall'] = measured['stage1_support_recall']
                    measured['support_precision']=measured['stage1_support_precision']
                    measured['support_f1']=measured['stage1_support_f1']
                    measured['support_recall_definition'] = 'stage1_selected_set'
        detail = json.dumps(meta, default=lambda v: v.tolist() if isinstance(v, np.ndarray) else v.item(),
                            allow_nan=False, ensure_ascii=False)
        return {"status": "ok", **measured, "details_json": detail}
    except NotImplementedError as exc:
        return {"status": "not_implemented", "reason": str(exc)}
