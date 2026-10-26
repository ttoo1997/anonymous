'Verify immutable release files, without treating output folders/.venv as source changes.'
from .checksum import checksum_file
import json
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
VERSION='SDP_TABLE2_TABLE5_ALIGNED_20261021'
def verify():
    manifest=json.loads((ROOT/'VERSION_LOCK.json').read_text(encoding='utf-8'))
    expected=manifest['files'];differences=[]
    for name,checks in expected.items():
        p=ROOT/name
        if not p.is_file() or checksum_file(p)!=checks:
            differences.append(name)
    for directory in ('paper_suite','lasso_threshold_v1','dp_safety'):
        for p in (ROOT/directory).rglob('*.py'):
            rel=p.relative_to(ROOT).as_posix()
            if '__pycache__' not in p.parts and rel not in expected:differences.append(rel)
    if differences:raise RuntimeError('Frozen version changed: '+', '.join(sorted(set(differences))))
    if manifest['version']!=VERSION:raise RuntimeError('unexpected release version')
    return manifest['version']
if __name__=='__main__':print(verify())
