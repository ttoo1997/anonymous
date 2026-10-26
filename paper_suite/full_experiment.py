'Full experiment dispatcher: bounded processes, per-cell checkpoints, no plots.'
from __future__ import annotations
import os
for _key in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS','NUMEXPR_NUM_THREADS'):
    os.environ[_key]='1'
import argparse
from concurrent.futures import ProcessPoolExecutor,as_completed
from dataclasses import asdict
from .checksum import checksum_file,checksum_json
from .frozen import VERSION, verify
import json
import shutil
from pathlib import Path
import time
import traceback
import numpy as np
import pandas as pd
from threadpoolctl import threadpool_limits
from .config import (SyntheticSetting, RealSetting, synthetic_grid, REAL_LOCAL_SAMPLE_GRID, REAL_EPSILON_GRID,
    REAL_FORMAL_ALGORITHMS, SAMPLE_SDP_CLIP_CONSTANT,SAMPLE_SDP_THRESHOLD_CONSTANT)
from .data import make_synthetic
from .datasets import load_registered,DATASETS,digest
from .runner import DEFAULT_ALGORITHMS,evaluate,fit_lasso,metric_row
from .run_synthetic import EXPERIMENTS
from .storage import atomic_json,json_default


def plan(repeats=10,profile='pdf_v1'):
    if profile!='pdf_v1':raise ValueError('unknown experiment profile')
    jobs=[]
    for experiment in EXPERIMENTS:
        for setting in synthetic_grid(experiment):
            for repeat in range(repeats):
                jobs.append({'kind':'synthetic','experiment':experiment,'setting':asdict(setting),'repeat':repeat})
    for dataset in sorted(DATASETS):
        for m in REAL_LOCAL_SAMPLE_GRID:
            for repeat in range(repeats):
                jobs.append({'kind':'real','dataset':dataset,'m':m,'repeat':repeat})
    for i,job in enumerate(jobs):job['job_id']=i
    return jobs


def identity(payload):
    return checksum_json(payload)


def validate_saved_row(row, rowbase):
    if any(row.get(k)!=v for k,v in rowbase.items()) or row.get('status')!='ok':
        raise ValueError('saved result does not match requested cell or success status')
    required=(['support_recall'] if rowbase['experiment'].startswith('exp4') else
              ['test_mse','mse_ratio'] if rowbase['experiment']=='exp5_real' else
              ['test_mse','relative_coefficient_error'])
    if any(not isinstance(row.get(k),(int,float)) or not np.isfinite(row[k]) for k in required):
        raise ValueError('saved result lacks finite required metrics')


def job_is_complete(out, job):
    'A finished status alone is not evidence that result files survived copying.'
    directory=Path(out)/'jobs'/f"{job['job_id']:04d}"
    try:
        status=json.loads((directory/'status.json').read_text(encoding='utf-8'))
        if status.get('state')!='finished':return False
        recall=job.get('experiment','').startswith('exp4')
        names=['proposed','2slr'] if recall else (list(REAL_FORMAL_ALGORITHMS) if job['kind']=='real' else list(DEFAULT_ALGORITHMS))
        epsilons=REAL_EPSILON_GRID if job['kind']=='real' else [job['setting']['epsilon']]
        expected={(a,e) for a in names for e in epsilons}
        seen=set()
        for path in directory.glob('*.row.json'):
            row=json.loads(path.read_text(encoding='utf-8'))
            base={'experiment':job.get('experiment','exp5_real'),'repeat':job['repeat'],
                  'seed':job.get('seed',20260915+job['repeat'])}
            if job['kind']=='real':base.update(dataset=job['dataset'],m=job['m'])
            else:base.update(job['setting'])
            validate_saved_row(row,base)
            key=(row['algorithm'],row['epsilon'])
            if key not in expected or key in seen:return False
            seen.add(key)
        return seen==expected
    except (OSError,ValueError,KeyError,TypeError):
        return False


def memo(out,payload,fn):

    from .file_lock import file_lock
    key=identity({'release':'SDP919L_DP_FIXED_R1_DELTA1E5_IHT_STAB',**payload})
    directory=out/'cache';directory.mkdir(exist_ok=True)
    dest=directory/(key+'.json')
    with file_lock(directory/(key+'.lock')):
        if dest.exists():return json.loads(dest.read_text(encoding='utf-8'))
        value=fn()
        atomic_json(dest,value)
        return value


def work(job,out_string):
    out=Path(out_string)
    directory=out/'jobs'/f"{job['job_id']:04d}"
    directory.mkdir(parents=True,exist_ok=True)
    status_path=directory/'status.json'
    atomic_json(status_path,{'state':'running','job':job,'pid':os.getpid(),'started':time.time()})
    completed=0;errors=[]
    try:
        seed=job.get('seed',20260915+job['repeat'])
        with threadpool_limits(limits=1):
            if job['kind']=='synthetic':
                setting=SyntheticSetting(**job['setting'])
                data=make_synthetic(setting,seed)
                settings=[setting]
                recall=job['experiment'].startswith('exp4')
                names=['proposed','2slr'] if recall else (list(REAL_FORMAL_ALGORITHMS) if job['kind']=='real' else list(DEFAULT_ALGORITHMS))
                base={'experiment':job['experiment'],'repeat':job['repeat'],'seed':seed}
                dataset_key={'kind':'synthetic','coefficient_protocol':'equal_positive_l1_v1','seed':seed,**asdict(setting)}
                dataset_key.pop('epsilon');dataset_key.pop('delta')
            else:
                data=load_registered(job['dataset'],job['m'],seed)
                settings=[RealSetting(n=len(data.user_x),m=job['m'],d=data.user_x.shape[-1],
                           k=min(8,data.user_x.shape[-1]),epsilon=e) for e in REAL_EPSILON_GRID]
                recall=False;names=list(REAL_FORMAL_ALGORITHMS)
                base={'experiment':'exp5_real','dataset':job['dataset'],'repeat':job['repeat'],'seed':seed}
                dataset_key={'kind':'real','dataset':job['dataset'],'m':job['m'],'seed':seed}
                atomic_json(directory/'data_preparation.json',data.metadata or {})
            def lasso_value():
                beta,meta=fit_lasso(data)
                return {'beta':beta,'meta':meta,**metric_row(beta,data)}

            lasso_key={**dataset_key,'method':'lasso_full_training'}
            if job['kind']=='real':lasso_key.pop('m')
            baseline=memo(out,lasso_key,lasso_value) if not recall else None
            if job['kind']=='real' and (not np.isfinite(baseline['test_mse']) or baseline['test_mse']<=0):
                raise ValueError('positive finite real Lasso denominator required')
            for setting in settings:
                for name in names:
                    rowbase={**base,**asdict(setting),'algorithm':name}
                    cell=identity(rowbase)
                    destination=directory/(cell+'.row.json')
                    if destination.exists():
                        validate_saved_row(json.loads(destination.read_text(encoding='utf-8')),rowbase)
                        completed+=1;continue
                    try:
                        if name=='lasso':
                            metrics={'status':'ok',**metric_row(np.asarray(baseline['beta']),data),
                                     'details_json':json.dumps(baseline['meta'])}
                        else:
                            protocol_tag={
                                'proposed':'appendix_b_table5_aligned_20261021',
                                '2slr':'ma24_algorithm1_alg7to9_paper_aligned_v4_hh_alpha0p05_exp5_context',
                                'ldpiht':('ldpiht_gradient_clipped_stabilization_v1'
                                           if job['kind']=='synthetic' else
                                           'zhu24_algorithm2_theorem7_reference_real_standardized_exploratory'),
                            }.get(name,'baseline_unchanged')
                            key={**dataset_key,'method':name,'selection_protocol':protocol_tag,'epsilon':setting.epsilon,
                                 'delta':setting.delta,'recall':recall,'algorithm_seed':seed+100000}
                            prepared=None
                            if name=='sample_sdp':
                                from .algorithms.sample_sdp_core import sdp2_table_parameters,prepare_exact_sample_statistics
                                x=data.user_x.reshape(-1,setting.d)
                                table=sdp2_table_parameters(len(x),setting.d,setting.epsilon,setting.delta,len(x),
                                    clip_constant=SAMPLE_SDP_CLIP_CONSTANT,threshold_constant=SAMPLE_SDP_THRESHOLD_CONSTANT,
                                    noise_scale=setting.noise_std,kappa_infinity=1.0)
                                prep_key={**dataset_key,'method':'sample_exact_preparation_v2_c0p1','precision':table['precision'],
                                          'clip':table['clip'],'clip_constant':SAMPLE_SDP_CLIP_CONSTANT,
                                          'threshold_constant':SAMPLE_SDP_THRESHOLD_CONSTANT,
                                          'response_scale':1.0,'response_scale_rule':'none_unit_l1_protocol',
                                          'rounding_seed':seed+100001}
                                prepared=memo(out,prep_key,lambda:prepare_exact_sample_statistics(x,
                                    data.user_y.ravel(),table['clip'],table['precision'],
                                    np.random.default_rng(seed+100001),chunk_size=max(1,min(64,2_000_000//(setting.d**2)))))
                                key['simulation_version']='sample_sdp_table2_threshold_20261021'
                                key['clip_constant']=SAMPLE_SDP_CLIP_CONSTANT
                                key['threshold_constant']=SAMPLE_SDP_THRESHOLD_CONSTANT
                                key['threshold_noise_scale']=setting.noise_std
                                key['threshold_kappa_infinity']=1.0
                                key['response_scale']=1.0
                                key['response_scale_rule']='none_unit_l1_protocol'
                            metrics=memo(out,key,lambda:evaluate(name,data,setting,seed+100000,recall,sample_preparation=prepared))
                        row={**rowbase,**metrics,
                             'comparison_delta':setting.delta,
                             'mechanism_delta':(None if name=='lasso' else 0.0 if name in ('2slr','ldpiht') else setting.delta),
                             'display_algorithm':('U2R' if name=='proposed' else 'LDPIHT (gradient-clipped)' if name=='ldpiht' and job['kind']=='synthetic' else name)}
                        if job['kind']=='real':
                            row['lasso_test_mse']=baseline['test_mse']
                            if metrics['status']=='ok':row['mse_ratio']=metrics['test_mse']/baseline['test_mse']
                        if row['status']!='ok':raise ValueError('non-success algorithm status: '+str(row))
                        required=['support_recall'] if recall else ['test_mse','mse_ratio' if job['kind']=='real' else 'relative_coefficient_error']
                        if any(not np.isfinite(row[k]) for k in required):raise ValueError('nonfinite required metric')
                        atomic_json(destination,row)
                        old_error=directory/(cell+'.error.json')
                        if old_error.exists():old_error.unlink()
                        completed+=1
                    except Exception:
                        failure={'cell':rowbase,'traceback':traceback.format_exc()}
                        errors.append(failure)
                        atomic_json(directory/(cell+'.error.json'),failure)
                    atomic_json(status_path,{'state':'running','job':job,'pid':os.getpid(),
                                'completed':completed,'errors':errors,'updated':time.time()})
    except Exception:
        errors.append({'traceback':traceback.format_exc()})
    status={'state':'failed' if errors else 'finished','job':job,'completed':completed,
            'errors':errors,'finished':time.time()}
    atomic_json(status_path,status)
    return status


def collect(out,jobs):
    rows=[]
    for path in sorted((out/'jobs').glob('*/*.row.json')):
        rows.append(json.loads(path.read_text(encoding='utf-8')))
    expected=sum(2 if j['kind']=='synthetic' and j['experiment'].startswith('exp4') else
                 len(DEFAULT_ALGORITHMS) if j['kind']=='synthetic' else len(REAL_FORMAL_ALGORITHMS)*len(REAL_EPSILON_GRID)
                 for j in jobs)
    statuses=[]
    for path in (out/'jobs').glob('*/status.json'):statuses.append(json.loads(path.read_text(encoding='utf-8')))
    result={'expected_cells':expected,'completed_cells':len(rows),'expected_jobs':len(jobs),
            'finished_jobs':sum(s['state']=='finished' for s in statuses),
            'failed_jobs':sum(s['state']=='failed' for s in statuses),
            'running_jobs':sum(s['state']=='running' for s in statuses),
            'all_requested_cells_ok':len(rows)==expected and all(r.get('status')=='ok' for r in rows)
                and sum(s['state']=='finished' for s in statuses)==len(jobs)
                and not any(s['state']!='finished' for s in statuses)
                and all(job_is_complete(out,j) for j in jobs),
            'updated':time.time()}
    if rows:
        frame=pd.DataFrame(rows)
        keys=['experiment','algorithm','repeat','n','m','d','k','epsilon']
        if 'dataset' in frame:keys.append('dataset')
        if frame.duplicated(keys).any():raise AssertionError('duplicate experiment rows')
        temp=out/'results.csv.tmp';frame.to_csv(temp,index=False);temp.replace(out/'results.csv')
        metric=[k for k in ['relative_coefficient_error','support_recall','stage1_support_recall','stage1_support_precision','stage1_support_f1','final_support_recall','final_support_precision','final_support_f1','topk_support_recall','topk_support_precision','support_precision','support_f1','estimated_nonzero','selected_count','test_mse','mse_ratio'] if k in frame]
        group=[k for k in keys if k!='repeat']
        summary=frame.groupby(group,dropna=False)[metric].agg(['mean','std','count'])
        summary.columns=['_'.join(c) for c in summary.columns]
        temp=out/'summary.csv.tmp';summary.to_csv(temp);temp.replace(out/'summary.csv')
        result['counts_by_experiment']=frame.groupby('experiment').size().to_dict()
        result['counts_by_algorithm']=frame.groupby('algorithm').size().to_dict()
    atomic_json(out/'progress.json',result)
    return result


def validate_resume(previous, current):
    if previous['jobs'] != current['jobs'] or previous['data_checksums'] != current['data_checksums']:
        raise ValueError('resume inputs differ from frozen manifest')
    if previous.get('source_checksums') != current['source_checksums']:
        raise ValueError('resume source code differs from frozen manifest; start a separate run')


def main():
    verify()
    from .frozen import verify
    verify()
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--out',required=True,type=Path)
    parser.add_argument('--workers',type=int)
    parser.add_argument('--profile',choices=['pdf_v1'],default='pdf_v1')
    parser.add_argument('--plan-only',action='store_true',help='freeze and inspect manifest without launching jobs')
    parser.add_argument('--repeats',type=int,default=10)
    parser.add_argument('--resume',action='store_true')
    parser.add_argument('--scope',choices=['synthetic','all'],default='synthetic')
    parser.add_argument('--experiment',choices=EXPERIMENTS,action='append')
    args=parser.parse_args()
    if args.workers is None:args.workers=1
    if args.workers<1 or args.repeats<1:raise ValueError('positive worker and repeat counts required')
    if args.out.exists() and not args.resume:raise FileExistsError(args.out)
    args.out.mkdir(parents=True,exist_ok=True)
    (args.out/'jobs').mkdir(exist_ok=True)
    jobs=plan(args.repeats,args.profile)
    if args.scope=='synthetic':jobs=[j for j in jobs if j['kind']=='synthetic']
    if args.experiment:jobs=[j for j in jobs if j.get('experiment') in args.experiment]
    for index,job in enumerate(jobs):job['job_id']=index
    code_root=Path(__file__).resolve().parent.parent
    sources=list((code_root/'paper_suite').rglob('*.py'))+list((code_root/'lasso_threshold_v1').rglob('*.py'))+list((code_root/'dp_safety').rglob('*.py'))
    source_checksums={p.relative_to(code_root).as_posix():checksum_file(p) for p in sources}
    manifest={'jobs':jobs,'workers':args.workers,'repeats':args.repeats,
              'metric_schema':'pdf_stage1_recall_v2',
              'metric_definitions':{'relative_coefficient_error':'L2(beta-beta_true)/L2(beta_true)',
                'support_recall':'Stage-1 selected-set recall only; reported for Proposed/2-SLR support experiments, not used as the LDP-IHT primary metric',
                'stage1_support_recall':'PDF stage1 selected-set recall',
                'final_support_recall':'post-hoc final nonzero coefficient recall (diagnostic)',
                'topk_support_recall':'post-hoc recall among the |S*| largest-magnitude final coefficients (diagnostic; especially for LDP-IHT)',
                'mse_ratio':'test MSE / same-split full-training Lasso test MSE'},
              'protocol':'equal_positive_l1_v1_sigma025_anchor_n15000_cross_panel_v4_n8000to20000_exp5warnsafe','coefficient_protocol':'equal_positive_l1_v1','plot_generation':False,
              'profile':args.profile,'source_checksums':source_checksums,
              'seed_rule':'20260915'+'+repeat; algorithm seed +100000',
              'data_checksums':({name:digest(spec['path']) for name,spec in DATASETS.items()} if any(j['kind']=='real' for j in jobs) else {}),
              'scope':args.scope, 'release_version':VERSION,
              'comparison_delta':SyntheticSetting().delta,
              'delta_rule':'U2R/SampleSDP use delta=1e-5; 2-SLR/LDPIHT remain pure DP with actual delta=0; no epsilon/m conversion',
              'privacy_scope':'conditional numerical utility study; not full logs DP certification',
              'exp5_interface':{'setting':'RealSetting without synthetic noise_std',
                                'formal_algorithms':list(REAL_FORMAL_ALGORITHMS),
                                'epsilon_grid':list(REAL_EPSILON_GRID),
                                'expected_cells_per_job':len(REAL_FORMAL_ALGORITHMS)*len(REAL_EPSILON_GRID)}}
    manifest['expected_cells']=sum(2 if j['kind']=='synthetic' and j['experiment'].startswith('exp4') else
                                  5 if j['kind']=='synthetic' else len(REAL_FORMAL_ALGORITHMS)*len(REAL_EPSILON_GRID) for j in jobs)
    manifest['largest_synthetic_x_gib']=max(j['setting']['n']*j['setting']['m']*j['setting']['d']*8/2**30
                                        for j in jobs if j['kind']=='synthetic')
    if args.resume:
        previous=json.loads((args.out/'manifest.json').read_text(encoding='utf-8'))
        validate_resume(previous, manifest)
    else:
        for path in sources:
            destination=args.out/'source_snapshot'/path.relative_to(code_root)
            destination.parent.mkdir(parents=True,exist_ok=True)
            shutil.copy2(path,destination)
        atomic_json(args.out/'manifest.json',manifest)
    if args.plan_only:
        print(json.dumps({'planned_jobs':len(jobs),'expected_cells':manifest['expected_cells'],
                         'largest_x_gib':manifest['largest_synthetic_x_gib'],'execution':'not_started'}),flush=True)
        return
    from .file_lock import file_lock
    with file_lock(args.out/'controller.lock',blocking=False):
        todo=[]
        for job in jobs:
            path=args.out/'jobs'/f"{job['job_id']:04d}"/'status.json'
            if job_is_complete(args.out,job):continue
            todo.append(job)
        print(json.dumps({'queued_jobs':len(todo),'workers':args.workers}),flush=True)
        with ProcessPoolExecutor(max_workers=args.workers,mp_context=__import__('multiprocessing').get_context('spawn')) as pool:
            futures={pool.submit(work,job,str(args.out)):job for job in todo}
            for future in as_completed(futures):
                job=futures[future]
                try:status=future.result()
                except Exception:
                    directory=args.out/'jobs'/f"{job['job_id']:04d}";directory.mkdir(exist_ok=True)
                    status={'state':'failed','job':job,'errors':[{'traceback':traceback.format_exc()}]}
                    atomic_json(directory/'status.json',status)
                progress=collect(args.out,jobs)
                print(json.dumps({'job_id':job['job_id'],'state':status['state'],**progress}),flush=True)
        final=collect(args.out,jobs)
        final['execution_status']='finished' if final['all_requested_cells_ok'] else 'incomplete'
        atomic_json(args.out/'completion.json',final)
        print(json.dumps(final),flush=True)
        if not final['all_requested_cells_ok']:raise SystemExit(2)

if __name__=='__main__':main()
