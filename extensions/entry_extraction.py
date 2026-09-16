"""Videm v3 candidate instance extraction, shared by validation and inference."""
import numpy as np
from skimage.measure import label, regionprops, find_contours
from skimage.segmentation import watershed
from shapely.geometry import Polygon
CONFIG={'method':'confidence_seeded_watershed','high':0.85,'low':0.5,'min_area_fraction':0.005,'simplify_pixels':2.0}
def extract_entries(prob):
 prob=np.asarray(prob);assert prob.ndim==2 and np.isfinite(prob).all()
 seeds=label(prob>CONFIG['high'])
 for c in regionprops(seeds):
  if c.area<CONFIG['min_area_fraction']*prob.size:seeds[seeds==c.label]=0
 seeds=label(seeds>0)
 instances=watershed(-prob,seeds,mask=prob>CONFIG['low'],watershed_line=True)
 result=[]
 for c in regionprops(instances):
  if c.area<CONFIG['min_area_fraction']*prob.size:continue
  contours=find_contours(np.pad(c.image,1),.5)
  if not contours:continue
  contour=max(contours,key=len)+np.asarray(c.bbox[:2])-1
  poly=Polygon(contour[:,::-1]).simplify(CONFIG['simplify_pixels'],preserve_topology=True)
  if not poly.is_valid:poly=poly.buffer(0)
  pieces=list(poly.geoms) if poly.geom_type=='MultiPolygon' else [poly]
  result.extend(x for x in pieces if x.area>=CONFIG['min_area_fraction']*prob.size)
 return sorted(result,key=lambda x:(x.centroid.y,x.centroid.x))
