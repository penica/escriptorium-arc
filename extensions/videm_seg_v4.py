"""Scoped Videm v4 region-only training; identical extractor for validation and deployment."""
from pathlib import Path
import json,time
import numpy as np
import torch
from torch.nn import functional as F
from lightning.pytorch import seed_everything
from lightning.pytorch.callbacks import Callback
from torchvision.transforms import v2,InterpolationMode
from PIL import Image
from shapely.geometry import Polygon
from scipy.optimize import linear_sum_assignment
from entry_extraction import extract_entries,CONFIG as EXTRACTOR_CONFIG
ROOT=Path('/usr/src/app/videm-seg-v4')
CONFIG=json.loads((ROOT/'training-config.json').read_text())
def seed():seed_everything(241960,workers=True)
def configure(model,dm):
 assert model.net.input[2]==1000,model.net.input
 assert model.net.user_metadata['class_mapping']['baselines']=={}
 assert model.net.user_metadata['class_mapping']['regions']=={'BaptismEntry':2}
 dm.train_set.dataset.aug._augment=v2.RandomApply([
  v2.RandomAffine(degrees=3,translate=(0.02,0.02),scale=(0.95,1.05),shear=(-2.,2.),interpolation=InterpolationMode.BILINEAR,fill=0.0),
  v2.ColorJitter(brightness=.1,contrast=.1)],p=.5)
 # Mark only this region branch. Deployment must explicitly support this metadata.
 model.net.user_metadata['videm_entry_extraction']={**EXTRACTOR_CONFIG,'version':3}
 model.net.user_metadata['videm_training_round']=4
class EntryMetrics(Callback):
 def on_fit_start(self,trainer,module):
  def capture(net,args,result):
   if not net.training:self.raw=result[0].detach()
  self.hook=module.net.register_forward_hook(capture)
 def on_validation_epoch_start(self,trainer,module):self.pages=[]
 def on_validation_batch_end(self,trainer,module,outputs,batch,batch_idx,dataloader_idx=0):
  assert tuple(module.hparams.config.padding)==(0,0)
  # Same nearest-neighbour upsampling and sigmoid order as Kraken's production _compute_segmentation_map.
  idx=module.net.user_metadata['class_mapping']['regions']['BaptismEntry']
  prob=torch.sigmoid(F.interpolate(self.raw,size=batch['image'].shape[-2:]))[0,idx].cpu().float().numpy();ph,pw=prob.shape
  polygons=extract_entries(prob);ds=trainer.datamodule.val_set.dataset;target=ds.targets[batch_idx]
  with Image.open(ds.imgs[batch_idx]) as im:
   w,h=im.size
   if trainer.sanity_checking:
    expected=ds.transforms(im)
    torch.testing.assert_close(batch['image'][0].cpu(),expected.cpu(),rtol=0,atol=0)
  gold=[Polygon([(x*pw/w,y*ph/h) for x,y in r.boundary]) for regs in target['regions'].values() for r in regs]
  mat=np.asarray([[a.intersection(b).area/a.union(b).area for b in gold] for a in polygons]);tp=0;overlap=0.;merged=0;split=0
  if polygons and gold:
   ii,jj=linear_sum_assignment(1-mat);scores=mat[ii,jj];tp=int((scores>=.5).sum());overlap=float(scores.sum())
   coverage=np.array([[a.intersection(b).area/b.area for b in gold] for a in polygons])
   merged=int(((coverage>.25).sum(axis=1)>1).sum())
   split=int(((coverage>.25).sum(axis=0)>1).sum())
  self.pages.append({'file':str(ds.imgs[batch_idx]),'tp':tp,'pred':len(polygons),'gt':len(gold),'iou_sum':overlap,'merged':merged,'split':split,'map_size':[pw,ph]})
 def on_validation_epoch_end(self,trainer,module):
  tp=sum(x['tp'] for x in self.pages);pred=sum(x['pred'] for x in self.pages);gt=sum(x['gt'] for x in self.pages)
  f1=2*tp/(pred+gt) if pred+gt else 1.;overlap=sum(x['iou_sum'] for x in self.pages)/gt;score=f1+.001*overlap
  module.log('val_entry_f1',f1,on_epoch=True);module.log('val_entry_score',score,on_epoch=True)
  row={'epoch':trainer.current_epoch+1,'step':trainer.global_step,'sanity':trainer.sanity_checking,'f1':f1,'assigned_iou':overlap,'score':score,'tp':tp,'pred':pred,'gt':gt,'merged':sum(x['merged'] for x in self.pages),'split':sum(x['split'] for x in self.pages),'pages':self.pages,'time':time.time()}
  if trainer.sanity_checking:(ROOT/'initial-validation.json').write_text(json.dumps(row,indent=2))
  else:
   with (ROOT/'history.jsonl').open('a') as f:f.write(json.dumps(row)+'\n')
  print('VIDEM_ENTRY_VALIDATION '+json.dumps(row),flush=True)
 def on_fit_end(self,trainer,module):self.hook.remove()
