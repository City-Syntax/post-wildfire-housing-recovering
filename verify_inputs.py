"""Verify the packaged numerical inputs against their recorded hashes."""
from pathlib import Path
import json,hashlib
ROOT=Path(__file__).resolve().parent

def verify():
    records=json.loads((ROOT/'reference/data_provenance.json').read_text(encoding='utf8'));checked=[]
    for row in records:
        if not row['package_path'].startswith('data/'):continue
        p=ROOT/row['package_path'];actual=hashlib.sha256(p.read_bytes()).hexdigest();assert actual==row['sha256'],'Input changed: '+row['package_path'];checked.append(row['package_path'])
    for row in json.loads((ROOT/'data/geography/source_manifest.json').read_text()):
        p=ROOT/'data/geography'/(row['name']+'.zip');assert hashlib.sha256(p.read_bytes()).hexdigest()==row['sha256'];checked.append(str(p.relative_to(ROOT)))
    return {'all_inputs_match':True,'files_checked':len(checked),'files':checked}

if __name__=='__main__':print(json.dumps(verify(),indent=2))
