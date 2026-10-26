from pathlib import Path
import json,sys,os
import numpy as np
from paper_suite.checksum import checksum_file,checksum_json
ROOT=Path(__file__).resolve().parents[1]
VENDOR=ROOT
sys.path.insert(0,str(VENDOR))
def serial(x):
    if isinstance(x,np.ndarray):return x.tolist()
    if isinstance(x,np.generic):return x.item()
    if isinstance(x,Path):return str(x)
    raise TypeError(type(x).__name__)
def dump(path,obj):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    temp=path.with_suffix(path.suffix+'.tmp')
    temp.write_text(json.dumps(obj,ensure_ascii=False,indent=2,default=serial,allow_nan=False),encoding='utf-8')
    os.replace(temp,path)
def load(path):return json.loads(Path(path).read_text(encoding='utf-8'))
def digest(obj):return checksum_json(obj,default=serial)
def filecheck(path):return checksum_file(path)
def source_checksum():
    return digest({'backend_lock':filecheck(VENDOR/'VERSION_LOCK.json'),'adapter':{p.relative_to(ROOT).as_posix():filecheck(p) for p in sorted((ROOT/'exp5').glob('*.py'))}})
def algorithm_checksum():
    return digest({'backend_lock':filecheck(VENDOR/'VERSION_LOCK.json'),'adapter':{n:filecheck(ROOT/'exp5'/n) for n in ('algorithms.py','sample_fast.py','sample_ridge.py','privacy.py','lasso_gram.py','two_local.py','two_solver.py','names.py','parameter_policy.py')}})
