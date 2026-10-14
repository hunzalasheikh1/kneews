"""Compare encoder precision on one real source series; emit numbers, never pixels."""
import json
import time
import numpy as np
import torch
from sqlalchemy import select
from ..config import ROOT,settings
from ..models import Study
from .features import FrozenEncoder
from .imaging import read_dicom_series,sample_25d


def benchmark(db):
    study=db.scalar(select(Study).where(Study.split=='train').order_by(Study.id))
    series=sorted(study.series,key=lambda s:s.id)[0]
    start=time.perf_counter()
    image,metadata=read_dicom_series(sorted((settings.data_root/series.relative_path).glob('*.dcm')))
    decoded=time.perf_counter()
    triplets=sample_25d(image,series.plane,metadata['laterality'])
    sampled=time.perf_counter()
    result=dict(study_id=study.id,series_id=series.id,decode_seconds=decoded-start,sample_seconds=sampled-decoded,
                acceptance_gate='median cosine >=0.999 and relative L2 <=0.01 on this engineering check; no outcome metrics inspected',measurements=[])
    reference=None
    for precision in ['float32','float16']:
        encoder=FrozenEncoder(ROOT/'models/dinov2-small',precision=precision)
        encoder.encode(triplets[:2])
        times=[]
        for _ in range(3):
            start=time.perf_counter();features=encoder.encode(triplets);times.append(time.perf_counter()-start)
        if reference is None:reference=features
        cosine=float(np.dot(reference,features)/(np.linalg.norm(reference)*np.linalg.norm(features)))
        relative=float(np.linalg.norm(reference-features)/np.linalg.norm(reference))
        result['measurements'].append(dict(precision=precision,seconds=times,median_seconds=float(np.median(times)),
                                            cosine_to_float32=cosine,relative_l2_to_float32=relative,signature=encoder.signature))
        del encoder
        if torch.backends.mps.is_available():torch.mps.empty_cache()
    directory=settings.artifact_root/'benchmarks';directory.mkdir(parents=True,exist_ok=True)
    (directory/'encoder-precision.json').write_text(json.dumps(result,indent=2))
    print(json.dumps(result,indent=2),flush=True)
