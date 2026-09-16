from pathlib import Path
import os
root=Path(os.environ.get('APP_ROOT', '/usr/src/app'))
def edit(path,old,new,count=1):
 p=root/path;s=p.read_text();assert s.count(old)==count,(path,old[:90],s.count(old));p.write_text(s.replace(old,new))
edit('apps/core/tasks.py',"    if device in ['cpu', 'mps']:","    if device in ('xpu', 'xpu:0'):\n        return 'xpu', [0]\n    if device in ['cpu', 'mps']:")
edit('apps/core/tasks.py',"        self.es_model.new_version(file=f'{relpath}/checkpoint_epoch={trainer.current_epoch}.ckpt')", "        checkpoint_path = trainer.checkpoint_callback.best_model_path\n        if checkpoint_path and os.path.isfile(checkpoint_path):\n            self.es_model.new_version(file=os.path.relpath(checkpoint_path, settings.MEDIA_ROOT))")
edit('apps/core/tasks.py',"        rec_train_config = train_config_class(\n            batch_size=BATCH_SIZE,\n            resize='union',\n        )", "        extra = {'optimizer': 'AdamW', 'schedule': 'constant'} if accelerator == 'xpu' and architecture == 'PPOCRv6Model' else {}\n        rec_train_config = train_config_class(\n            batch_size=BATCH_SIZE, resize='union', **extra,\n        )")
edit('apps/core/tasks.py',"def segtrain(model_pk=None, part_pks=[], document_pk=None, task_group_pk=None, user_pk=None, **kwargs):", "def segtrain(model_pk=None, part_pks=[], document_pk=None, task_group_pk=None, user_pk=None, **kwargs):\n    if os.getenv('VIDEM_V4_ENABLED') == '1':\n        from videm_seg_v4 import CONFIG\n        if document_pk == CONFIG['document']:\n            from videm_training import _videm_v4_segtrain\n            return _videm_v4_segtrain(model_pk=model_pk, part_pks=part_pks, document_pk=document_pk, task_group_pk=task_group_pk, user_pk=user_pk, **kwargs)")
p=root/'apps/core/tasks.py';p.write_text(p.read_text()+"\nfrom videm_inference import install as _install_videm_inference\n_install_videm_inference()\n")
edit('apps/core/models.py','SegmentationInferenceConfig(accelerator=accelerator, device=device','SegmentationInferenceConfig(precision="32-true", raise_on_error=True, accelerator=accelerator, device=device',2)
edit('apps/core/models.py','RecognitionInferenceConfig(accelerator=accelerator, device=device','RecognitionInferenceConfig(precision="32-true", raise_on_error=True, accelerator=accelerator, device=device')
p=root/'escriptorium/settings.py';p.write_text(p.read_text()+'''
# Pinned local ARC compatibility configuration.
import arc_bootstrap
KRAKEN_TRAINING_PRECISION = '32-true'
KRAKEN_TRAINING_BATCH_SIZE = int(os.getenv('KRAKEN_TRAINING_BATCH_SIZE', '1'))
SESSION_COOKIE_NAME = os.getenv('SESSION_COOKIE_NAME', 'sessionid')
CSRF_COOKIE_NAME = os.getenv('CSRF_COOKIE_NAME', 'csrftoken')
EMAIL_BACKEND = os.getenv('EMAIL_BACKEND', 'django.core.mail.backends.console.EmailBackend')
CELERY_TASK_ROUTES.update({'core.tasks.segment': {'queue': 'gpu'}, 'core.tasks.transcribe': {'queue': 'gpu'}})
''')
edit('escriptorium/celery.py',"app.autodiscover_tasks()", """# Preserve protection for long running training jobs.
app.conf.broker_transport_options = {**(app.conf.broker_transport_options or {}), 'visibility_timeout': 43200}
app.conf.result_backend_transport_options = {**(app.conf.result_backend_transport_options or {}), 'visibility_timeout': 43200}
app.conf.visibility_timeout = 43200
app.conf.worker_deduplicate_successful_tasks = True
_annotations = dict(app.conf.task_annotations or {})
_annotations['core.tasks.segtrain'] = {**_annotations.get('core.tasks.segtrain', {}), 'acks_late': True}
app.conf.task_annotations = _annotations
app.autodiscover_tasks()""")
# Optional device routing is kept separate from upstream's accounting route map.
p=root/'escriptorium/celery.py'
p.write_text(p.read_text()+"\nfrom distributed_routing import install as _install_distributed_routing\n_install_distributed_routing(app)\n")
# Lock each model across hosts while it is being trained. Opt-in at runtime.
p=root/'apps/core/tasks.py'
s=p.read_text()
s='from training_safety import exclusive_training, assert_training_lock\n'+s
for name in ('train', 'segtrain', 'train_from_collection', 'segtrain_from_collection'):
    marker='def '+name+'('
    assert s.count(marker)==1,(name,s.count(marker))
    s=s.replace(marker,'@exclusive_training\n'+marker)
import re
s=re.sub(r'(?m)^( +)convert_models\(\[best_path\], model.file.path\)', r'\1assert_training_lock()\n\1convert_models([best_path], model.file.path)', s)
p.write_text(s)
