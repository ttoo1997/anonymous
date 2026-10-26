"""Run the frozen synthetic studies and retain every requested seed."""
import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed
import json
from pathlib import Path

from paper_suite.full_experiment import work, collect, job_is_complete
from paper_suite.frozen import verify
from paper_suite.checksum import checksum_file
from paper_suite.storage import atomic_json

ROOT = Path(__file__).resolve().parents[1]


def build_jobs(config, studies, seeds=None):
    jobs = []
    for name in studies:
        spec = config['studies'][name]
        for value in spec['values']:
            for repeat, seed in enumerate(spec['seeds'] if seeds is None else seeds):
                setting = dict(config['default'])
                setting[spec['variable']] = value
                jobs.append(dict(kind='synthetic', experiment=spec['experiment'],
                                 setting=setting, repeat=repeat, seed=seed, job_id=len(jobs)))
    return jobs


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--study', nargs='+', choices=['privacy','users','dimension','support_m','support_n','all'], default=['all'])
    p.add_argument('--out', type=Path, required=True)
    p.add_argument('--workers', type=int, default=1)
    p.add_argument('--seeds', type=int, nargs='+')
    p.add_argument('--plan-only', action='store_true')
    p.add_argument('--resume', action='store_true')
    args = p.parse_args()
    if args.workers < 1 or (args.seeds and len(set(args.seeds)) != len(args.seeds)):
        raise ValueError('Require positive workers and unique seeds')
    verify()
    config = json.loads((ROOT / 'configs/experiments.json').read_text())
    studies = list(config['studies']) if 'all' in args.study else list(dict.fromkeys(args.study))
    jobs = build_jobs(config, studies, args.seeds)
    out = args.out.resolve()
    out.mkdir(parents=True, exist_ok=True)
    manifest = {'version':verify(), 'jobs':jobs,
                'source_lock_checksum':checksum_file(ROOT/'VERSION_LOCK.json'),
                'selection':'all specified seeds; no outcome-dependent filtering'}
    path = out / 'manifest.json'
    if path.exists():
        if not args.resume or json.loads(path.read_text()) != manifest:
            raise ValueError('Existing output needs --resume with an identical manifest')
    else:
        atomic_json(path, manifest)
    if args.plan_only:
        print(json.dumps({'jobs':len(jobs), 'studies':studies, 'fitting_started':False}))
        return
    pending = [j for j in jobs if not job_is_complete(out, j)]
    with ProcessPoolExecutor(max_workers=args.workers) as pool:
        futures = {pool.submit(work, j, str(out)):j for j in pending}
        for future in as_completed(futures):
            result = future.result()
            print(json.dumps({'job_id':result['job']['job_id'], 'state':result['state']}), flush=True)
    summary = collect(out, jobs)
    print(json.dumps(summary))
    if not summary['all_requested_cells_ok']:
        raise SystemExit(1)


if __name__ == '__main__':
    main()
