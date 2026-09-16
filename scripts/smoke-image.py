"""CPU-only import validation; real XPU inference/training remains a staging gate."""
from importlib.metadata import version
for package, expected in [('kraken', '7.1.1'), ('lightning', '2.6.1')]:
    if version(package) != expected:
        raise SystemExit(f'{package} must remain {expected} until revalidated')
import torch
import arc_adapter
import videm_inference
import entry_extraction
assert torch.version.xpu is not None, 'PyTorch is not an XPU build'
print('CPU import checks passed; this is not an ARC hardware validation')
