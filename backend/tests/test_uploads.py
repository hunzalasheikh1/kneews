import io
import numpy as np
import nibabel as nib
import pytest
from starlette.datastructures import UploadFile
from app.uploads import ingest_upload
from app.config import settings

class FakeDB:
    def __init__(self):self.rows=[]
    def add(self,row):self.rows.append(row)
    def flush(self):pass
    def commit(self):pass


def test_nifti_upload_preserves_shape_and_source_partition(tmp_path,monkeypatch):
    monkeypatch.setattr(settings,'data_root',tmp_path/'data')
    source=tmp_path/'scan.nii'
    nib.save(nib.Nifti1Image(np.ones((5,7,9),np.float32),np.eye(4)),source)
    db=FakeDB()
    with source.open('rb') as f:
        result=ingest_upload(UploadFile(filename='scan.nii',file=f),db)
    assert len(result['study_ids'])==1
    assert db.rows[0].split=='external'
    assert db.rows[1].metadata_json['shape']==[5,7,9]


def test_upload_rejects_non_imaging_extension(tmp_path,monkeypatch):
    monkeypatch.setattr(settings,'data_root',tmp_path)
    with pytest.raises(ValueError,match='DICOM ZIP'):
        ingest_upload(UploadFile(filename='script.py',file=io.BytesIO(b'print(1)')),FakeDB())
