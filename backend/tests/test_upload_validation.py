"""Upload validation edge cases (module M1). Uses only synthetic files."""
import io
import zipfile
import numpy as np
import nibabel as nib
import pytest
from pydicom.dataset import Dataset,FileMetaDataset
from pydicom.uid import ExplicitVRLittleEndian,generate_uid
from starlette.datastructures import UploadFile
from app.uploads import ingest_upload
from app.config import settings


class FakeDB:
    def __init__(self):self.rows=[]
    def add(self,row):self.rows.append(row)
    def flush(self):pass
    def commit(self):pass
    def get(self,model,key):return None


def dicom_bytes(**fields):
    ds=Dataset();ds.file_meta=FileMetaDataset();ds.file_meta.TransferSyntaxUID=ExplicitVRLittleEndian
    ds.file_meta.MediaStorageSOPClassUID='1.2.840.10008.5.1.4.1.1.4';ds.file_meta.MediaStorageSOPInstanceUID=generate_uid()
    ds.SOPClassUID=ds.file_meta.MediaStorageSOPClassUID;ds.SOPInstanceUID=ds.file_meta.MediaStorageSOPInstanceUID
    for key,value in fields.items():setattr(ds,key,value)
    buffer=io.BytesIO();ds.save_as(buffer,enforce_file_format=True);return buffer.getvalue()


def zip_upload(members):
    buffer=io.BytesIO()
    with zipfile.ZipFile(buffer,'w') as archive:
        for name,data in members.items():archive.writestr(name,data)
    buffer.seek(0);return UploadFile(filename='study.zip',file=buffer)


@pytest.fixture
def data_root(tmp_path,monkeypatch):
    monkeypatch.setattr(settings,'data_root',tmp_path);return tmp_path


def test_zip_without_dicom_files_is_rejected(data_root):
    with pytest.raises(ValueError,match='DICOM files'):
        ingest_upload(zip_upload({'notes.txt':b'hello'}),FakeDB())


def test_dicom_without_study_identifiers_is_rejected(data_root):
    with pytest.raises(ValueError,match='identifiers'):
        ingest_upload(zip_upload({'a.dcm':dicom_bytes(Modality='MR')}),FakeDB())


def test_dicom_without_orientation_is_rejected(data_root):
    data=dicom_bytes(StudyInstanceUID=generate_uid(),SeriesInstanceUID=generate_uid())
    with pytest.raises(ValueError,match='orientation'):
        ingest_upload(zip_upload({'a.dcm':data}),FakeDB())


def test_valid_dicom_plane_comes_from_orientation_and_paths_are_not_trusted(data_root):
    study,series=generate_uid(),generate_uid()
    data=dicom_bytes(StudyInstanceUID=study,SeriesInstanceUID=series,ImageOrientationPatient=[0,1,0,0,0,-1])
    db=FakeDB()
    result=ingest_upload(zip_upload({'../../escape.dcm':data}),db)
    assert result=={'study_ids':[study],'series':1}
    saved=db.rows[1]
    assert saved.plane=='Sagittal' and saved.study_id==study
    assert not (data_root.parent/'escape.dcm').exists()
    assert len(list((data_root/saved.relative_path).glob('*.dcm')))==1


def test_compressed_nifti_upload_is_accepted(data_root,tmp_path):
    path=tmp_path/'scan.nii.gz'
    nib.save(nib.Nifti1Image(np.zeros((4,5,6),np.float32),np.eye(4)),path)
    db=FakeDB()
    with path.open('rb') as f:
        result=ingest_upload(UploadFile(filename='scan.nii.gz',file=f),db)
    assert result['series']==1 and db.rows[1].format=='nifti'
