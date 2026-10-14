"""Verify and time sampling optimization against the original algorithm."""
import json
import time
import numpy as np
from sqlalchemy import select
from ..config import settings
from ..models import Study
from .imaging import read_dicom_series,sample_25d,normalize,physical_crop


def original(image,plane,laterality,samples=16,size=224,field_mm=130):
    data=normalize(image.get_fdata(dtype=np.float32));spacing=image.header.get_zooms()
    if laterality.upper() in ('R','RIGHT'):data=np.flip(data,axis=0)
    axis={'Sagittal':0,'Coronal':1,'Axial':2}[plane];depth=data.shape[axis]
    indices=np.rint(np.linspace(.1*(depth-1),.9*(depth-1),samples)).astype(int)
    pixel_spacing=[spacing[i] for i in range(3) if i!=axis]
    return np.stack([np.stack([physical_crop(np.take(data,np.clip(center+offset,0,depth-1),axis=axis),pixel_spacing,field_mm,size)
                     for offset in (-1,0,1)]) for center in indices]).astype(np.float32)


def benchmark(db):
    studies=db.scalars(select(Study).where(Study.split=='train').order_by(Study.id).limit(3)).all()
    rows=[]
    for study in studies:
        for plane in ['Sagittal','Coronal','Axial']:
            series=next(s for s in sorted(study.series,key=lambda s:s.id) if s.plane==plane)
            image,meta=read_dicom_series(sorted((settings.data_root/series.relative_path).glob('*.dcm')))
            start=time.perf_counter();a=original(image,plane,meta['laterality']);old=time.perf_counter()-start
            start=time.perf_counter();b=sample_25d(image,plane,meta['laterality']);new=time.perf_counter()-start
            row=dict(study_id=study.id,plane=plane,old_seconds=old,new_seconds=new,speedup=old/new,
                     max_pixel_difference=float(np.max(np.abs(a-b))),identical=bool(np.array_equal(a,b)))
            rows.append(row);print(json.dumps(row),flush=True)
            if not row['identical']:raise ValueError('Optimized sampler changed input pixels')
    directory=settings.artifact_root/'benchmarks';directory.mkdir(parents=True,exist_ok=True)
    (directory/'sampling.json').write_text(json.dumps(rows,indent=2))
