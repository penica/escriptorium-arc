"""Exercise live registry reload, routing and confinement without a database."""
import json, os, sys, tempfile
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'extensions'))
from training_policy import document_policy, confined_path
from distributed_routing import DeviceRouter

with tempfile.TemporaryDirectory() as directory:
    root=Path(directory);registry=root/'policies.json'
    os.environ['TRAINING_POLICY_FILE']=str(registry)
    data={'schema_version':1,'defaults':{'device':'arc','max_epochs':40},
          'documents':{'19':{'device':'cuda','max_epochs':3,'stop_after_epoch':1}}}
    registry.write_text(json.dumps(data))
    router=DeviceRouter()
    assert router('core.tasks.segtrain',(),{'document_pk':19},{})=={'queue':'cuda-training'}
    assert router('core.tasks.segtrain',(),{'document_pk':15},{})=={'queue':'gpu'}
    data['documents']['19']['device']='arc';registry.write_text(json.dumps(data))
    assert router('core.tasks.segtrain',(),{'document_pk':19},{})=={'queue':'gpu'}
    for relative in ['../escape','training-runs/19/../../escape','/tmp/outside']:
        try:confined_path(root,relative,'training-runs/19')
        except ValueError:pass
        else:raise AssertionError('Accepted unsafe path '+relative)
    (root/'training-runs/19').mkdir(parents=True)
    (root/'training-runs/19/link').symlink_to(root,target_is_directory=True)
    try:confined_path(root,'training-runs/19/link/outside','training-runs/19')
    except ValueError:pass
    else:raise AssertionError('Accepted symlink escape')
    for invalid in [0,-1,True,'3']:
        data['documents']['19']['max_epochs']=invalid;registry.write_text(json.dumps(data))
        try:document_policy(19)
        except ValueError:pass
        else:raise AssertionError('Accepted invalid epoch bound')
    data['documents']['19']={'device':'cuda','recipe':'legacy','max_epochs':40}
    registry.write_text(json.dumps(data))
    try:document_policy(19)
    except ValueError:pass
    else:raise AssertionError('Legacy recipe was allowed on CUDA')
print('Policy routing reload, bounds and path confinement passed')
