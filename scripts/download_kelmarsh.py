"""Download publisher files and perform basic name, size, and CSV header checks."""
import argparse
import json
from pathlib import Path
import shutil
import urllib.request

RECORD='16807551'


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--raw-dir',type=Path,required=True)
    args=p.parse_args(); args.raw_dir.mkdir(parents=True,exist_ok=True)
    with urllib.request.urlopen('https://zenodo.org/api/records/'+RECORD,timeout=60) as response:
        record=json.load(response)
    selected=[f for f in record['files'] if f['key']=='Kelmarsh_WT_dataSignalMapping.csv' or
              any(f['key'].startswith('Kelmarsh_SCADA_'+str(year)+'_') for year in range(2017,2023))]
    if len(selected)!=7:raise ValueError('Unexpected publisher file list')
    checked=[]
    for item in selected:
        name=item['key']
        if Path(name).name!=name:raise ValueError('Invalid publisher filename')
        target=args.raw_dir/name
        if not target.exists():
            temporary=target.with_suffix(target.suffix+'.part')
            url='https://zenodo.org/records/'+RECORD+'/files/'+name+'?download=1'
            with urllib.request.urlopen(url,timeout=60) as source,temporary.open('wb') as output:
                shutil.copyfileobj(source,output,length=1024*1024)
            temporary.replace(target)
        size=target.stat().st_size
        published_size=item.get('size')
        if size<=0 or (published_size is not None and size!=int(published_size)):
            raise ValueError('Invalid download size: '+name)
        with target.open('rb') as handle:
            header=handle.readline(8192).decode('utf-8-sig',errors='replace').strip()
        if not header or (',' not in header and ';' not in header):
            raise ValueError('Invalid CSV header: '+name)
        checked.append({'name':name,'size_bytes':size})
        print('Size and header checked',name,flush=True)
    (args.raw_dir/'publisher_manifest.json').write_text(json.dumps({'publisher_record':RECORD,'files':checked},indent=2),encoding='utf-8')


if __name__=='__main__':
    main()
