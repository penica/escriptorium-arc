import os
if os.environ.get('KRAKEN_ENABLE_XPU') == '1':
    import arc_adapter
# Preserve full-float32 page geometry across CUDA and XPU backends. Reduced
# precision Tensor Core convolution can move thresholded contours by pixels.
if any(os.getenv(name, '').startswith('cuda') for name in ('KRAKEN_TRAINING_DEVICE', 'KRAKEN_INFERENCE_DEVICE')):
    import torch
    torch.set_float32_matmul_precision('highest')
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
