"""Trusted, per-document training controls shared by producers and workers.

The router reads current policy; execution takes a private snapshot and refuses
to run on a different device if policy changed while the task was queued.
"""
from contextvars import ContextVar
from copy import deepcopy
from functools import wraps
from inspect import signature
import json
import os
from pathlib import Path
import re
import uuid
import datetime
import time


def ensure_shared_directory(path):
    """Tolerate a short shared-filesystem metadata visibility delay."""
    path = Path(path)
    for attempt in range(7):
        try:
            path.mkdir(parents=True, exist_ok=True)
            if path.is_dir():
                return path
        except (FileExistsError, FileNotFoundError):
            if path.is_dir():
                return path
            if path.exists():
                raise
        if attempt == 6:
            raise OSError(f'Shared directory did not become visible: {path}')
        time.sleep(0.25 * (attempt + 1))

_current = ContextVar('training_policy', default=None)


def read_registry():
    filename = os.getenv('TRAINING_POLICY_FILE')
    if not filename:
        return None
    data = json.loads(Path(filename).read_text())
    if data.get('schema_version') != 1 or not isinstance(data.get('documents'), dict):
        raise ValueError('Invalid training policy registry')
    return data


def document_policy(document_id):
    registry = read_registry()
    if registry is None:
        return None
    row = deepcopy(registry.get('defaults', {}))
    row.update(deepcopy(registry['documents'].get(str(document_id), {})))
    if row.get('device') not in ('arc', 'cuda'):
        raise ValueError('Training device must be arc or cuda')
    if row.get('recipe', 'standard') not in ('standard', 'videm', 'legacy'):
        raise ValueError('Unknown training recipe')
    row.setdefault('recipe', 'standard')
    if type(row.get('max_epochs')) is not int or row['max_epochs'] < 1:
        raise ValueError('max_epochs must be a positive integer')
    stop = row.get('stop_after_epoch')
    if stop is not None and (type(stop) is not int or not 1 <= stop <= row['max_epochs']):
        raise ValueError('stop_after_epoch must be within the planned training duration')
    if row['recipe'] == 'legacy' and row['device'] != 'arc':
        raise ValueError('Legacy private training remains on ARC')
    return row


def task_document(name, args=(), kwargs=None):
    kwargs = kwargs or {}
    if kwargs.get('document_pk') is not None:
        return int(kwargs['document_pk'])
    if name == 'core.tasks.segtrain' and len(args or ()) > 2:
        return int(args[2])
    if name == 'core.tasks.train':
        transcription = kwargs.get('transcription_pk', (args or [None])[0])
        if transcription is not None:
            from django.apps import apps
            return apps.get_model('core', 'Transcription').objects.values_list('document_id', flat=True).get(pk=transcription)
    return None


def current_policy():
    return _current.get()


def should_stop(policy, completed_epochs):
    bound=policy.get('stop_after_epoch')
    if bound is not None and completed_epochs >= bound:
        return True
    marker=Path(policy['attempt_dir'])/'stop-request.json'
    if not marker.exists():
        return False
    request=json.loads(marker.read_text())
    if request.get('task_id')!=policy['task_id'] or request.get('model_id')!=policy['model_id']:
        raise ValueError('Stop request does not match the current training attempt')
    return True


def atomic_json(path, data):
    path=Path(path);ensure_shared_directory(path.parent)
    temporary=path.with_name('.'+path.name+'.'+uuid.uuid4().hex+'.pending')
    try:
        with temporary.open('w') as stream:
            json.dump(data,stream,indent=2,sort_keys=True);stream.flush();os.fsync(stream.fileno())
        os.replace(temporary,path)
    finally:
        temporary.unlink(missing_ok=True)


def record_status(policy, state, error=None):
    from django.conf import settings
    receipt={'state':state,'updated_at':datetime.datetime.now(datetime.timezone.utc).isoformat(),
             'policy':policy,'error':error}
    atomic_json(Path(settings.MEDIA_ROOT)/'training-control/models'/f"{policy['model_id']}.json",receipt)


def confined_path(root, relative, required_prefix):
    relative = Path(relative)
    if relative.is_absolute():
        raise ValueError('Policy paths must be relative to MEDIA_ROOT')
    path = (root / relative).resolve()
    allowed = (root / required_prefix).resolve()
    if not path.is_relative_to(allowed) or path == allowed:
        raise ValueError('Training policy path escapes its allowed directory')
    return path


def snapshot_policy(document_id, model_id, parts, task_id):
    from django.conf import settings
    from django.apps import apps
    row = document_policy(document_id)
    if row is None or row['recipe'] == 'legacy':
        return None
    models = row.get('models')
    if models is not None and str(model_id) not in models:
        raise ValueError('This model is not approved by the document training policy')
    parts = sorted(set(int(p) for p in parts))
    if not parts or apps.get_model('core','DocumentPart').objects.filter(pk__in=parts, document_id=document_id).count() != len(parts):
        raise ValueError('Training pages do not belong to the configured document')
    configured_device = str(getattr(settings, 'KRAKEN_TRAINING_DEVICE', 'cpu'))
    expected = 'xpu' if row['device'] == 'arc' else 'cuda'
    if not configured_device.startswith(expected):
        raise ValueError('Task reached the wrong training device; refusing fallback')
    if not re.fullmatch(r'[A-Za-z0-9_-]+', task_id):
        raise ValueError('Invalid training task identifier')
    media = Path(settings.MEDIA_ROOT).resolve()
    ensure_shared_directory(media / 'training-runs' / str(model_id))
    model_options = (models or {}).get(str(model_id), {})
    resume = model_options.get('resume_from')
    if resume:
        checkpoint = confined_path(media, resume, f'training-runs/{model_id}')
        if checkpoint.name != 'latest.ckpt' or not checkpoint.is_file():
            raise ValueError('Resume requires an existing trusted latest.ckpt')
        run = checkpoint.parent
    else:
        checkpoint = None
        run = media / 'training-runs' / str(model_id) / task_id
        # A redelivered job must not silently overwrite an earlier attempt.
        if run.exists():
            raise ValueError('Run already exists; inspect it and request explicit resume')
    recipe_config = row.get('recipe_config')
    if row['recipe'] == 'videm' and not recipe_config:
        raise ValueError('Portable Videm training requires an explicit recipe_config')
    config_path = confined_path(media, recipe_config, 'training-control/recipes') if recipe_config else None
    if config_path is not None and not config_path.is_file():
        raise ValueError('Recipe configuration file is missing')
    result = {**row, 'document_id':int(document_id),'model_id':int(model_id),'part_ids':parts,
              'task_id':task_id,'run_dir':str(run),'attempt_dir':str(run/'attempts'/task_id),
              'resume_from':str(checkpoint) if checkpoint else None,
              'recipe_config_path':str(config_path) if config_path else None}
    return result


def managed_training(function):
    params = signature(function)
    @wraps(function)
    def wrapped(*args, **kwargs):
        bound = params.bind_partial(*args, **kwargs).arguments
        values = {**bound.get('kwargs', {}), **{k:v for k,v in bound.items() if k != 'kwargs'}}
        name = 'core.tasks.' + function.__name__
        token = None
        policy = None
        try:
            document = task_document(name, kwargs=values)
            if document is None or read_registry() is None:
                return function(*args, **kwargs)
            from celery import current_task
            task_id = getattr(getattr(current_task, 'request', None), 'id', None) or str(uuid.uuid4())
            policy = snapshot_policy(document, values['model_pk'], values.get('part_pks') or [], task_id)
            if policy is not None:
                policy['transcription_id'] = values.get('transcription_pk')
                record_status(policy,'running')
            token = _current.set(policy)
            result=function(*args, **kwargs)
            if policy is not None:
                receipt=Path(policy['attempt_dir'])/('result.json' if policy['recipe']=='videm' else 'fit-result.json')
                completed=json.loads(receipt.read_text())['completed_epochs']
                record_status(policy,'paused' if completed < policy['max_epochs'] else 'complete')
            return result
        except Exception as exc:
            # The outer model lease ensures we cannot clear another writer's flag.
            from django.apps import apps
            try:
                apps.get_model('core','OcrModel').objects.filter(pk=values['model_pk']).update(training=False)
            except Exception:
                import logging
                logging.getLogger(__name__).exception('Could not clear training flag after failure')
            if policy is not None:
                record_status(policy,'failed',repr(exc))
            raise
        finally:
            if token is not None:
                _current.reset(token)
    return wrapped


def dispatch_videm(**kwargs):
    policy = current_policy()
    if policy is None or policy['recipe'] != 'videm':
        return False, None
    from training_extensions.videm_recipe import train_videm
    return True, train_videm(policy=deepcopy(policy), **kwargs)
