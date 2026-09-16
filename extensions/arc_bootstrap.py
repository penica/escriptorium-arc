import os
if os.environ.get('KRAKEN_ENABLE_XPU') == '1':
    import arc_adapter
