import hashlib
import json
from types import SimpleNamespace
import numpy as np
import pytest
from app.research.cache import read_cache
from app.research.readiness import validate_label
from app.research.constants import FINDINGS


def test_cache_rejects_corruption_wrong_study_and_mask_provenance(tmp_path):
    encoder={'version':'fixture'};signature=hashlib.sha256(json.dumps(encoder,sort_keys=True).encode()).hexdigest()
    path=tmp_path/'study.npz';features=np.zeros((6,8),np.float32);features[0]=1
    mask=np.array([1,0,0,0,0,0],bool)
    metadata=dict(study_id='study',signature=signature,encoder=encoder,sources=[{'slot':0}])
    def save():np.savez(path,features=features,mask=mask,metadata=json.dumps(metadata))
    save();read_cache(path,'study',signature,8)
    with pytest.raises(ValueError,match='provenance'):read_cache(path,'other',signature,8)
    features[0,0]=np.nan;save()
    with pytest.raises(ValueError,match='finite'):read_cache(path,'study',signature,8)
    features[0,0]=1;metadata['sources']=[];save()
    with pytest.raises(ValueError,match='sources'):read_cache(path,'study',signature,8)
    metadata['sources']=[{'slot':0}];encoder['version']='altered';save()
    with pytest.raises(ValueError,match='configuration hash'):read_cache(path,'study',signature,8)
    path.write_bytes(b'truncated')
    with pytest.raises(ValueError):read_cache(path,'study',signature,8)


def test_training_labels_must_match_report_and_version():
    report='No tear.'
    provenance=dict(model='test:local',digest='a'*64,prompt_sha256='b'*64,schema_version='s3',local_only=True,
                    report_sha256=hashlib.sha256(report.encode()).hexdigest())
    label=SimpleNamespace(version='test:local:'+('a'*12)+':'+('b'*12)+':s3',provenance=provenance,
        findings={f:dict(state='unmentioned',probability=None,evidence=[]) for f in FINDINGS})
    study=SimpleNamespace(report=report)
    validate_label(label,study)
    study.report='Changed report.'
    with pytest.raises(ValueError,match='report hash'):validate_label(label,study)
    study.report=report;label.version='another-version'
    with pytest.raises(ValueError,match='provenance'):validate_label(label,study)
