"""Check exact upstream patch applicability and compile the patched application."""
import compileall, os, subprocess, sys
from pathlib import Path
root = Path(__file__).resolve().parents[1]
app = root / 'source' / 'app'
if not app.is_dir():
    raise SystemExit('Fetch upstream first')
import tempfile, shutil
temp = tempfile.TemporaryDirectory()
app_copy = Path(temp.name) / 'app'
shutil.copytree(app, app_copy)
app = app_copy
env = dict(os.environ, APP_ROOT=str(app))
subprocess.run([sys.executable, str(root / 'patches' / 'app.py')], env=env, check=True)
if not compileall.compile_dir(str(app), quiet=1) or not compileall.compile_dir(str(root / 'extensions'), quiet=1):
    raise SystemExit('Python compilation failed')
print('Exact application patches and Python syntax passed')
