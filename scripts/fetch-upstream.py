"""Fetch and archive exactly one official upstream commit into the build context."""
import argparse, json, os, re, subprocess, tarfile, tempfile
from pathlib import Path

parser = argparse.ArgumentParser()
parser.add_argument('--commit', help='Optional immutable commit; default is current develop HEAD')
args = parser.parse_args()
root = Path(__file__).resolve().parents[1]
config = json.loads((root / 'upstream.json').read_text())
source = root / 'source'
if source.exists():
    raise SystemExit('source/ already exists; use a clean checkout for each build')
commit = args.commit
if commit is None:
    commit = subprocess.check_output(['git', 'ls-remote', config['repository'], 'refs/heads/' + config['branch']], text=True).split()[0]
if not re.fullmatch(r'[0-9a-f]{40}', commit):
    raise SystemExit('Expected a full 40-character commit SHA')
with tempfile.TemporaryDirectory() as temp:
    checkout = Path(temp) / 'checkout'
    subprocess.run(['git', 'init', str(checkout)], check=True, stdout=subprocess.DEVNULL)
    git = ['git', '-C', str(checkout)]
    subprocess.run(git + ['fetch', '--depth=1', config['repository'], commit], check=True)
    actual = subprocess.check_output(git + ['rev-parse', 'FETCH_HEAD'], text=True).strip()
    if actual != commit:
        raise SystemExit('Fetched commit does not match the requested SHA')
    archive = Path(temp) / 'source.tar'
    subprocess.run(git + ['archive', '--format=tar', '--output=' + str(archive), commit], check=True)
    with tarfile.open(archive) as tar:
        tar.extractall(source, filter='data')
print(commit)
if os.getenv('GITHUB_OUTPUT'):
    with open(os.environ['GITHUB_OUTPUT'], 'a') as output:
        output.write('commit=' + commit + '\n')
