"""Opt-in single-Intel-GPU, float32 adapter for Kraken 7.1.1 and Lightning 2.6.1.

Registers an accelerator through Lightning's extension API and adapts Kraken's
device parser for this process only. Installed Kraken/CPU files are not patched.
"""
import sys
from importlib.metadata import version
for package, expected in [('kraken','7.1.1'),('lightning','2.6.1')]:
    if version(package) != expected:
        raise RuntimeError(f'This Arc adapter was verified against {package} {expected}; revalidate before upgrading.')
import torch
from lightning.pytorch.accelerators import Accelerator, AcceleratorRegistry

class IntelXPUAccelerator(Accelerator):
    def setup_device(self, device):
        if device.type != 'xpu': raise ValueError('Expected an Intel XPU device')
        torch.xpu.set_device(device)
        print(f'Kraken compute device: {device}; {torch.xpu.get_device_name(device)}',flush=True)
    def get_device_stats(self, device):
        return {'allocated_bytes':torch.xpu.memory_allocated(device),'reserved_bytes':torch.xpu.memory_reserved(device)}
    def teardown(self):
        torch.xpu.synchronize(); torch.xpu.empty_cache()
    @staticmethod
    def parse_devices(devices):
        if devices == 'auto': return [0]
        if isinstance(devices,int): devices=list(range(devices))
        if isinstance(devices,str): devices=[int(x) for x in devices.split(',')]
        if devices != [0]: raise ValueError('This adapter supports only xpu:0')
        return devices
    @staticmethod
    def get_parallel_devices(devices):
        return [torch.device('xpu',i) for i in IntelXPUAccelerator.parse_devices(devices)]
    @staticmethod
    def auto_device_count(): return 1
    @staticmethod
    def is_available(): return torch.xpu.is_available()
    @staticmethod
    def name(): return 'xpu'

AcceleratorRegistry.register('xpu',IntelXPUAccelerator,description='Local Kraken single Intel GPU adapter')
from lightning.fabric.accelerators import ACCELERATOR_REGISTRY
ACCELERATOR_REGISTRY.register('xpu',IntelXPUAccelerator,description='Local Kraken single Intel GPU adapter')
# Lightning 2.6 chooses CPU for unknown accelerator types even after registration.
# Scope this compatibility hook to explicit XPU calls in this wrapper process.
from lightning.fabric.connector import _Connector
from lightning.fabric.strategies import SingleDeviceStrategy as FabricSingleDevice
from lightning.pytorch.trainer.connectors.accelerator_connector import _AcceleratorConnector
from lightning.pytorch.strategies import SingleDeviceStrategy as TrainerSingleDevice
def choose_single_xpu(connector_class,strategy_class):
    original_choose=connector_class._choose_strategy
    def choose(self):
        if self._accelerator_flag == 'xpu':
            return strategy_class(device=torch.device('xpu:0'))
        return original_choose(self)
    connector_class._choose_strategy=choose
choose_single_xpu(_Connector,FabricSingleDevice)
choose_single_xpu(_AcceleratorConnector,TrainerSingleDevice)
import kraken.ketos
import kraken.ketos.util
original=kraken.ketos.util.to_ptl_device
def device_parser(device):
    if device.strip() in ('xpu','xpu:0'): return 'xpu',[0]
    return original(device)
for name,module in list(sys.modules.items()):
    if name.startswith('kraken.ketos') and getattr(module,'to_ptl_device',None) is original:
        module.to_ptl_device=device_parser
if __name__=='__main__':
    if not torch.xpu.is_available(): raise SystemExit('Intel XPU is unavailable; refusing silent CPU fallback')
    print('Intel XPU adapter:',torch.xpu.get_device_name(0),flush=True)
    kraken.ketos.cli()
