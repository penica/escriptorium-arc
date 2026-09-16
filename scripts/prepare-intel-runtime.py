"""Download checksum-pinned Intel packages; extract libraries and redistribution notices."""
import hashlib
import json
import shutil
import subprocess
import tempfile
import urllib.request
from pathlib import Path

root = Path(__file__).resolve().parents[1]
output = root / '.build' / 'intel-runtime'
if output.exists():
    raise SystemExit('Use a clean .build/intel-runtime directory')
output.mkdir(parents=True)
packages = json.loads((root / 'runtime/intel-packages.json').read_text())
with tempfile.TemporaryDirectory() as temp:
    temp = Path(temp)
    for package in packages:
        deb = temp / (package['name'] + '.deb')
        with urllib.request.urlopen(package['url'], timeout=120) as response, deb.open('wb') as dest:
            shutil.copyfileobj(response, dest)
        with deb.open('rb') as stream:
            if hashlib.file_digest(stream, 'sha256').hexdigest() != package['sha256']:
                raise SystemExit('Checksum mismatch: ' + package['name'])
        extracted = temp / package['name']
        subprocess.run(['dpkg-deb', '-x', str(deb), str(extracted)], check=True)
        libraries = list((extracted / 'usr/lib/x86_64-linux-gnu').rglob('*.so*'))
        if not libraries:
            raise SystemExit('No libraries: ' + package['name'])
        for lib in libraries:
            shutil.copy2(lib, output / lib.name, follow_symlinks=False)
        docs = extracted / 'usr/share/doc' / package['name']
        if not (docs / 'copyright').is_file():
            raise SystemExit('Missing redistribution notice: ' + package['name'])
        shutil.copytree(docs, output / 'licenses' / package['name'])
        print('Verified ' + package['name'] + '=' + package['version'], flush=True)
(output / 'vendors').mkdir()
(output / 'vendors/intel.icd').write_text('/opt/intel-runtime/libigdrcl.so\n')
shutil.copyfile(root / 'runtime/intel-packages.json', output / 'package-provenance.json')
print('Intel runtime prepared with original package notices')
