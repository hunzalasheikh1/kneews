"""Strict feature-cache contract shared by preflight, resume and training."""
import hashlib
import json
import numpy as np


def read_cache(path,study_id,signature,dimension=768):
    with np.load(path,allow_pickle=False) as saved:
        features=saved['features'];mask=saved['mask'];metadata=json.loads(str(saved['metadata']))
    if metadata.get('study_id')!=study_id or metadata.get('signature')!=signature:
        raise ValueError('Feature cache study/signature provenance mismatch')
    encoder=metadata.get('encoder')
    if not isinstance(encoder,dict) or hashlib.sha256(json.dumps(encoder,sort_keys=True).encode()).hexdigest()!=signature:
        raise ValueError('Feature cache encoder configuration hash mismatch')
    if features.shape!=(6,dimension) or features.dtype!=np.float32 or not np.isfinite(features).all():
        raise ValueError('Feature cache must contain finite float32 six-slot embeddings')
    if mask.shape!=(6,) or mask.dtype!=np.bool_ or not mask.any():
        raise ValueError('Feature cache requires a boolean mask with an available slot')
    if features[~mask].any():raise ValueError('Unavailable feature slots must be zero')
    slots=[source.get('slot') for source in metadata.get('sources',[])]
    if sorted(slots)!=np.flatnonzero(mask).tolist():
        raise ValueError('Feature cache sources do not match available slots')
    return features,mask,metadata
