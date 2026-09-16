"""Opt-in Videm inference adapter. No behavior change for unmarked Kraken models."""
from dataclasses import replace
import uuid
import torch
import numpy as np
from kraken.containers import Segmentation,Region
from kraken.lib.vgsl.spred import VGSLSegmentationInference
from entry_extraction import extract_entries,CONFIG
_INSTALLED=False

def install():
 global _INSTALLED
 if _INSTALLED:return
 original=VGSLSegmentationInference._segmentation_pred
 @torch.inference_mode()
 def scoped(self,im):
  settings=self.user_metadata.get('videm_entry_extraction')
  if settings:
   assert settings=={**CONFIG,'version':3},'Unknown Videm extractor version/configuration'
   assert not self.user_metadata['class_mapping']['baselines'],'Entry branch must not contain text lines'
   with self._fabric.init_tensor():result=self._compute_segmentation_map(im)
   idx=result['cls_map']['regions']['BaptismEntry'];polys=extract_entries(result['heatmap'][idx]);scale=result['scale']
   regs=[Region(id='_'+str(uuid.uuid4()),boundary=np.rint(np.asarray(poly.exterior.coords[:-1])*scale).astype(int).tolist(),tags={'type':[{'type':'BaptismEntry'}]}) for poly in polys]
   return Segmentation(imagename=getattr(im,'filename',None),type='baselines',text_direction=self._inf_config.text_direction,lines=[],regions={'BaptismEntry':regs},script_detection=False,line_orders=[])
  seg=original(self,im)
  if self.user_metadata.get('videm_preserve_original_line_crops'):
   # Compute original generic regions and crops first. Hide generic regions only
   # after crop geometry is complete; the task merger assigns custom entries.
   return replace(seg,regions={},lines=[replace(line,regions=[]) for line in seg.lines])
  return seg
 VGSLSegmentationInference._segmentation_pred=scoped
 _INSTALLED=True
