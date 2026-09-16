"""Bounded standard Kraken training with full, independently refreshed checkpoints."""
import hashlib
from importlib.metadata import version
import json
import os
from pathlib import Path
import random
import shutil

import numpy as np
import torch
from kraken.train import KrakenTrainer
from lightning.pytorch.callbacks import Callback, ModelCheckpoint
from training_checkpoints import AtomicCheckpointIO
from training_policy import current_policy, should_stop, ensure_shared_directory
from training_safety import assert_training_lock


def sha(path):
    with Path(path).open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def write_json(path, data):
    path = Path(path)
    ensure_shared_directory(path.parent)
    temporary = path.with_suffix(path.suffix + '.pending')
    with temporary.open('w') as stream:
        json.dump(data, stream, indent=2, sort_keys=True)
        stream.flush()
        os.fsync(stream.fileno())
    os.replace(temporary, path)


def dataset_identity(policy):
    from django.apps import apps
    parts = apps.get_model('core', 'DocumentPart').objects.filter(pk__in=policy['part_ids']).order_by('pk')
    rows = []
    for part in parts:
        rows.append({'part':part.pk,'image_sha256':sha(part.image.path),
                     'lines':list(part.lines.order_by('pk').values('pk','baseline','mask','typology__name')),
                     'regions':list(part.blocks.order_by('pk').values('pk','box','typology__name'))})
    texts = []
    if policy.get('transcription_id'):
        texts = list(apps.get_model('core','LineTranscription').objects.filter(
            transcription_id=policy['transcription_id'],line__document_part_id__in=policy['part_ids']
        ).order_by('pk').values('line_id','content'))
    payload = {'pages':rows,'texts':texts}
    return hashlib.sha256(json.dumps(payload,sort_keys=True,separators=(',',':')).encode()).hexdigest()


class RunState(Callback):
    def __init__(self, policy, identity):
        self.policy, self.identity = policy, identity
        self.device = 'xpu' if policy['device']=='arc' else 'cuda'
        self.pending_rng = None

    def on_save_checkpoint(self, trainer, module, checkpoint):
        assert_training_lock()
        checkpoint['standard_identity'] = self.identity
        numpy_state = np.random.get_state()
        numpy_state = (numpy_state[0], numpy_state[1].tolist(), *numpy_state[2:])
        checkpoint['standard_rng'] = {'python':random.getstate(),'numpy':numpy_state,
            'cpu':torch.get_rng_state(),'device_type':self.device,
            'device':getattr(torch,self.device).get_rng_state_all()}

    def on_load_checkpoint(self, trainer, module, checkpoint):
        if checkpoint.get('standard_identity') != self.identity:
            raise ValueError('Checkpoint does not match this standard training run')
        self.pending_rng = checkpoint.get('standard_rng')
        if not self.pending_rng:
            raise ValueError('Checkpoint is missing RNG state')

    def on_fit_start(self, trainer, module):
        # Lightning creates the data iterator before on_train_start; restore
        # before that iterator consumes the next random seed.
        if self.policy['resume_from'] and not self.pending_rng:
            raise ValueError('Resume RNG state was not loaded before fit setup')
        if self.pending_rng:
            state = self.pending_rng
            random.setstate(state['python'])
            numpy_state = state['numpy']
            np.random.set_state((numpy_state[0], np.asarray(numpy_state[1], dtype=np.uint32), *numpy_state[2:]))
            torch.set_rng_state(state['cpu'].cpu())
            if state['device_type']==self.device:
                getattr(torch,self.device).set_rng_state_all([s.cpu() for s in state['device']])

    def on_train_start(self, trainer, module):
        actual = next(module.parameters()).device.type
        if actual != self.device:
            raise ValueError(f'Training expected {self.device}, got {actual}')
        write_json(Path(self.policy['attempt_dir'])/'started.json',{
            'device':str(next(module.parameters()).device),'global_step':trainer.global_step,
            'epoch':trainer.current_epoch,'resumed':bool(self.pending_rng),
            'cross_device':bool(self.pending_rng and self.pending_rng['device_type']!=self.device)})

    def on_train_epoch_end(self, trainer, module):
        assert_training_lock()
        if should_stop(self.policy,trainer.current_epoch+1):
            trainer.should_stop = True


class PolicyTrainer(KrakenTrainer):
    def __init__(self, *args, **kwargs):
        self.training_policy = current_policy()
        policy = self.training_policy
        if policy is None or policy['recipe']!='standard':
            super().__init__(*args, **kwargs)
            return
        run = Path(policy['run_dir']); attempt=Path(policy['attempt_dir'])
        ensure_shared_directory(run)
        ensure_shared_directory(attempt.parent)
        attempt.mkdir(exist_ok=False)
        from django.apps import apps
        model=apps.get_model('core','OcrModel').objects.get(pk=policy['model_id'])
        identity={'document':policy['document_id'],'model':policy['model_id'],'parts':policy['part_ids'],
                  'transcription':policy.get('transcription_id'),'max_epochs':policy['max_epochs'],
                  'dataset_sha256':dataset_identity(policy),'standard_code_sha256':sha(__file__),
                  'kraken':version('kraken'),'lightning':version('lightning'),'torch':version('torch').split('+')[0]}
        manifest_path=run/'manifest.json'
        if policy['resume_from']:
            manifest=json.loads(manifest_path.read_text())
            if manifest['identity']!=identity:
                raise ValueError('Dataset, software or training policy changed since checkpoint')
            initial=run/'initial.safetensors'
            if manifest.get('initial_sha256') and sha(initial)!=manifest['initial_sha256']:
                raise ValueError('Original starting weights changed')
            saved=torch.load(policy['resume_from'],map_location='cpu',weights_only=False)
            if saved.get('standard_identity')!=identity or not saved.get('optimizer_states'):
                raise ValueError('A matching full optimizer checkpoint is required')
            completed=saved['epoch']+1
            if completed>=policy['max_epochs'] or (policy.get('stop_after_epoch') is not None and policy['stop_after_epoch']<=completed):
                raise ValueError('No training epochs remain for this resume request')
            del saved
        else:
            if manifest_path.exists():
                raise ValueError('Existing run requires explicit resume')
            initial_sha=None
            if model.file and Path(model.file.path).is_file():
                shutil.copyfile(model.file.path,run/'initial.safetensors');initial_sha=sha(run/'initial.safetensors')
            write_json(manifest_path,{'identity':identity,'initial_sha256':initial_sha})
        write_json(attempt/'policy.json',policy)
        callbacks=list(kwargs.get('callbacks',[]))
        previous=[c for c in callbacks if isinstance(c,ModelCheckpoint)]
        if len(previous)!=1:
            raise ValueError('Standard training requires one original selection callback')
        selector=previous[0]
        best=ModelCheckpoint(dirpath=str(run),filename='best-{epoch:04d}',monitor=selector.monitor,
            mode=selector.mode,save_top_k=1,save_last=False,save_weights_only=False,
            save_on_train_epoch_end=False,enable_version_counter=False)
        latest=ModelCheckpoint(dirpath=str(run),filename='latest',monitor=None,save_top_k=1,
            every_n_epochs=1,save_last=False,save_weights_only=False,
            save_on_train_epoch_end=True,enable_version_counter=False)
        callbacks=[c for c in callbacks if not isinstance(c,ModelCheckpoint)]
        callbacks.extend([best,latest,RunState(policy,identity)])
        plugins=kwargs.pop('plugins',[])
        if not isinstance(plugins,list):plugins=[plugins]
        kwargs.update(callbacks=callbacks,max_epochs=policy['max_epochs'],min_epochs=1,
                      plugins=[*plugins,AtomicCheckpointIO()])
        if policy['resume_from']:kwargs['num_sanity_val_steps']=0
        super().__init__(*args,**kwargs)

    def fit(self,*args,**kwargs):
        policy=self.training_policy
        if policy is None or policy['recipe']!='standard':
            return super().fit(*args,**kwargs)
        if policy['resume_from']:
            kwargs.update(ckpt_path=policy['resume_from'],weights_only=False)
        try:
            result=super().fit(*args,**kwargs)
            assert_training_lock()
            latest=Path(policy['run_dir'])/'latest.ckpt'
            saved=torch.load(latest,map_location='cpu',weights_only=False)
            if saved['global_step']!=self.global_step or not saved.get('optimizer_states'):
                raise RuntimeError('Latest checkpoint is missing current full optimizer state')
            write_json(Path(policy['attempt_dir'])/'fit-result.json',{
                'status':'paused' if saved['epoch']+1 < policy['max_epochs'] else 'complete',
                'global_step':self.global_step,'completed_epochs':saved['epoch']+1,
                'latest_checkpoint':str(latest),'best_checkpoint':self.checkpoint_callback.best_model_path,
                'optimizer_saved':True,'scheduler_count':len(saved.get('lr_schedulers',[])),
                'rng_saved':bool(saved.get('standard_rng')),'device':policy['device']})
            return result
        except Exception as exc:
            write_json(Path(policy['attempt_dir'])/'error.json',{'error':repr(exc)})
            raise
