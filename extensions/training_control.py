"""Operator controls. Run inside the web container; submission stays in the API/UI."""
import argparse
from contextlib import contextmanager
import fcntl
import hashlib
import json
import os
from pathlib import Path
import sys

sys.path.insert(0, '/usr/src/app')
from training_policy import atomic_json, confined_path, document_policy


@contextmanager
def registry_lock():
    path = Path(os.environ['TRAINING_POLICY_FILE'])
    with path.with_suffix('.lock').open('a') as stream:
        fcntl.flock(stream, fcntl.LOCK_EX)
        yield path, json.loads(path.read_text())


def idle_model(model_id):
    from core.models import OcrModel
    model = OcrModel.objects.get(pk=model_id)
    if model.training:
        raise ValueError('Model is marked training; inspect its current task first')
    return model


def receipt(model_id):
    from django.conf import settings
    path = Path(settings.MEDIA_ROOT)/'training-control/models'/f'{model_id}.json'
    return json.loads(path.read_text()) if path.exists() else None


def idle_document(document_id, row):
    """Document defaults affect every approved model, including queued work."""
    from core.models import OcrModel
    from django.apps import apps
    if OcrModel.objects.filter(pk__in=row.get('models',{}),training=True).exists():
        raise ValueError('Another approved model on this document is training')
    # Task reports also cover jobs queued before model.training becomes true.
    Report = apps.get_model('reporting','TaskReport')
    if Report.objects.filter(document_id=document_id,workflow_state__in=[0,1]).exists():
        raise ValueError('Document has queued/running tasks; wait before changing shared policy')


def save_registry(path, data):
    # Validate candidate before publishing; avoid changing the process-wide
    # policy path while another operation could be using it.
    candidate = path.with_name('policies.validation.json')
    atomic_json(candidate, data)
    previous = os.environ['TRAINING_POLICY_FILE']
    try:
        os.environ['TRAINING_POLICY_FILE'] = str(candidate)
        for doc in data['documents']:
            document_policy(doc)
    finally:
        os.environ['TRAINING_POLICY_FILE'] = previous
        candidate.unlink(missing_ok=True)
    old = path.read_bytes()
    backup = path.with_name('policies.before-'+hashlib.sha256(old).hexdigest()+'.json')
    if not backup.exists():
        backup.write_bytes(old)
    atomic_json(path, data)
    return {'policy_sha256': hashlib.sha256(path.read_bytes()).hexdigest()}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest='command', required=True)
    status = commands.add_parser('status')
    status.add_argument('--model',type=int,required=True)
    clone = commands.add_parser('clone')
    clone.add_argument('--source-model',type=int,required=True)
    clone.add_argument('--document',type=int,required=True)
    clone.add_argument('--name',required=True)
    configure = commands.add_parser('configure')
    configure.add_argument('--document',type=int,required=True)
    configure.add_argument('--model',type=int,required=True)
    configure.add_argument('--device',choices=['arc','cuda'],required=True)
    configure.add_argument('--epochs',type=int,required=True)
    configure.add_argument('--stop-after',type=int)
    resume = commands.add_parser('prepare-resume')
    resume.add_argument('--model',type=int,required=True)
    resume.add_argument('--device',choices=['arc','cuda'])
    resume.add_argument('--stop-after',type=int)
    stop = commands.add_parser('request-stop')
    stop.add_argument('--model',type=int,required=True)
    args = parser.parse_args()
    os.environ.setdefault('DJANGO_SETTINGS_MODULE','escriptorium.settings')
    import django
    django.setup()
    from core.models import Document, OcrModel
    from django.conf import settings
    if args.command == 'status':
        model = OcrModel.objects.get(pk=args.model)
        result = {'model':model.pk,'database_training':model.training,'receipt':receipt(model.pk),
                  'note':'A stored running receipt does not prove a live worker; inspect task reports after a crash.'}
    elif args.command == 'clone':
        source = idle_model(args.source_model)
        document = Document.objects.get(pk=args.document)
        model = source.clone_for_training(document.owner,name=args.name)
        result = {'new_model':model.pk,'source_model':source.pk,'submitted':False}
    elif args.command == 'request-stop':
        state = receipt(args.model)
        if not state or state['state']!='running':
            raise ValueError('No recorded running attempt to stop')
        policy = state['policy']
        root = Path(settings.MEDIA_ROOT).resolve()
        relative = Path(policy['attempt_dir']).relative_to(root)
        attempt = confined_path(root, relative, f'training-runs/{args.model}')
        atomic_json(attempt/'stop-request.json',{'task_id':policy['task_id'],'model_id':args.model})
        result = {'stop_requested':True,'task_id':policy['task_id'],'effective':'next completed epoch'}
    else:
        idle_model(args.model)
        with registry_lock() as (path,data):
            if args.command == 'configure':
                Document.objects.get(pk=args.document)
                row = data['documents'].get(str(args.document),{})
                idle_document(args.document,row)
                if row.get('recipe') in ('legacy','videm'):
                    raise ValueError('Private recipe policies require their reviewed deployment workflow')
                if receipt(args.model):
                    raise ValueError('Existing run: use prepare-resume or clone a fresh model')
                if row.get('models') and row.get('max_epochs')!=args.epochs:
                    raise ValueError('Cannot change planned epochs for existing sibling models; use a separate document')
                row.update(device=args.device,recipe='standard',max_epochs=args.epochs,stop_after_epoch=args.stop_after)
                row.setdefault('models',{})[str(args.model)]={'resume_from':None}
                data['documents'][str(args.document)]=row
            else:
                state = receipt(args.model)
                if not state or state['state'] not in ('paused','failed'):
                    raise ValueError('Resume requires a paused or failed run; inspect stale running tasks manually')
                policy = state['policy']
                root = Path(settings.MEDIA_ROOT).resolve()
                relative = (Path(policy['run_dir'])/'latest.ckpt').relative_to(root)
                checkpoint = confined_path(root,relative,f'training-runs/{args.model}')
                if not checkpoint.is_file():
                    raise ValueError('No full latest checkpoint exists')
                row = data['documents'][str(policy['document_id'])]
                idle_document(policy['document_id'],row)
                if row['max_epochs']!=policy['max_epochs'] or row['recipe']!=policy['recipe']:
                    raise ValueError('Planned epochs or recipe changed since the original attempt')
                row['models'][str(args.model)]['resume_from']=str(relative)
                row['stop_after_epoch']=args.stop_after
                if args.device:
                    row['device']=args.device
            result = save_registry(path,data)
            result.update(submitted=False,next='Verify worker policy hash, then submit the same model/pages through API with override=true.')
    print(json.dumps(result,indent=2))


if __name__ == '__main__':
    main()
