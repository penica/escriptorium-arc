"""Explicit GPU routing. Unapproved training stays on the existing ARC queue."""
import os

TRAINING = frozenset({'core.tasks.train', 'core.tasks.segtrain'})
INFERENCE = frozenset({'core.tasks.segment', 'core.tasks.transcribe'})

class DeviceRouter:
    def __init__(self, training_documents=(), inference_queue='intensive-inference', training_queue='cuda-training'):
        self.training_documents = frozenset(int(pk) for pk in training_documents)
        self.inference_queue = inference_queue
        self.training_queue = training_queue

    def __call__(self, name, args, kwargs, options, task=None, **extra):
        if name in INFERENCE:
            return {'queue': self.inference_queue}
        if name in TRAINING:
            from training_policy import document_policy, task_document
            document = task_document(name, args, kwargs)
            policy = document_policy(document) if document is not None else None
            if policy is not None:
                return {'queue': self.training_queue if policy['device'] == 'cuda' else 'gpu'}
            if document is not None and int(document) in self.training_documents:
                return {'queue': self.training_queue}
            return {'queue': 'gpu'}
        # Collection training and unrelated tasks retain their existing routes.
        return None

def install(app):
    """Invoke after the application has loaded its normal Django configuration.

    Preserve settings.CELERY_TASK_ROUTES for upstream GPU accounting. The app
    router adds an explicit device decision ahead of those default routes.
    """
    if os.getenv('DISTRIBUTED_GPU_ENABLED') != '1':
        return
    if getattr(app, '_distributed_gpu_router_installed', False):
        return
    docs = filter(None, os.getenv('CUDA_TRAINING_DOCUMENT_IDS', '').split(','))
    router = DeviceRouter(docs)
    original = app.conf.task_routes or {}
    existing = list(original) if isinstance(original, (list, tuple)) else [original]
    app.conf.update(CELERY_TASK_ROUTES=[router, *existing])
    app._distributed_gpu_router_installed = True
