"""Verify routing does not accidentally send unrestricted training to CUDA."""
import sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parent.parent/'extensions'))
from distributed_routing import DeviceRouter
router=DeviceRouter([123])
assert router('core.tasks.segtrain',(),{'document_pk':123},{}) == {'queue':'cuda-training'}
for kwargs in ({}, {'document_pk':10}, {'document_pk':None}):
 assert router('core.tasks.segtrain',(),kwargs,{}) == {'queue':'gpu'}
assert router('core.tasks.transcribe',(),{}, {}) == {'queue':'intensive-inference'}
assert router('core.tasks.segment',(),{}, {}) == {'queue':'intensive-inference'}
assert router('core.tasks.train_from_collection',(),{}, {}) is None
assert router('imports.tasks.import_document',(),{}, {}) is None
print('Explicit device routing checks passed')
if '--celery' in sys.argv:
 import os
 from celery import Celery
 from distributed_routing import install
 app=Celery('routing-check',set_as_current=False)
 app.config_from_object({'CELERY_TASK_ROUTES':{'core.tasks.segtrain':{'queue':'gpu'}}},namespace='CELERY')
 os.environ['DISTRIBUTED_GPU_ENABLED']='1'
 os.environ['CUDA_TRAINING_DOCUMENT_IDS']='123'
 install(app)
 assert app.amqp.router.route({},'core.tasks.segtrain',kwargs={'document_pk':123})['queue'].name=='cuda-training'
 assert app.amqp.router.route({},'core.tasks.segtrain',kwargs={'document_pk':10})['queue'].name=='gpu'
 print('Namespaced Celery routing checks passed')
