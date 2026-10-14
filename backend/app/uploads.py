"""Local DICOM ZIP / NIfTI ingestion into isolated upload directories."""
import shutil
import uuid
import zipfile
from pathlib import Path
import pydicom
from fastapi import UploadFile
from .config import settings
from .models import Study,Series
from .research.imaging import load_nifti

MAX_UPLOAD=5*1024**3
MAX_EXPANDED=10*1024**3


def ingest_upload(upload:UploadFile,db):
    name=(upload.filename or '').lower()
    if not name.endswith(('.zip','.nii','.nii.gz')):
        raise ValueError('Upload a DICOM ZIP, .nii or .nii.gz volume')
    identifier=str(uuid.uuid4());directory=settings.data_root/'uploads'/identifier
    directory.mkdir(parents=True)
    extension='.nii.gz' if name.endswith('.nii.gz') else Path(name).suffix
    original=directory/('original'+extension)
    total=0
    with original.open('wb') as target:
        while chunk:=upload.file.read(4*1024*1024):
            total+=len(chunk)
            if total>MAX_UPLOAD: raise ValueError('Upload exceeds the 5 GB limit')
            target.write(chunk)
    if extension!='.zip':
        image=load_nifti(original)
        study_id='upload-'+identifier;series_id='nifti-'+identifier
        db.add(Study(id=study_id,source='upload',split='external',metadata_json={'original_name':upload.filename}))
        db.flush()
        db.add(Series(id=series_id,study_id=study_id,plane='NIfTI',format='nifti',
            relative_path=str(original.relative_to(settings.data_root)),
            metadata_json={'shape':list(image.shape),'spacing':[float(x) for x in image.header.get_zooms()]}))
        db.commit();return {'study_ids':[study_id],'series':1}
    with zipfile.ZipFile(original) as archive:
        files=[x for x in archive.infolist() if not x.is_dir() and x.filename.lower().endswith('.dcm')]
        if not files or len(files)>50000 or sum(x.file_size for x in files)>MAX_EXPANDED:
            raise ValueError('ZIP must contain 1–50,000 DICOM files, at most 10 GB expanded')
        studies={};series={}
        for info in files:
            # Never use archive paths for filesystem writes.
            with archive.open(info) as f: ds=pydicom.dcmread(f,stop_before_pixels=True)
            if not getattr(ds,'StudyInstanceUID',None) or not getattr(ds,'SeriesInstanceUID',None):
                raise ValueError('DICOM study and series identifiers are required')
            study_id=str(ds.StudyInstanceUID);series_id=str(ds.SeriesInstanceUID)
            if study_id not in studies:
                if db.get(Study,study_id): raise ValueError('Study already exists; duplicate uploads are not merged')
                studies[study_id]=Study(id=study_id,source='upload',split='external')
                db.add(studies[study_id]);db.flush()
            if series_id not in series:
                if db.get(Series,series_id): raise ValueError('Series identifier already exists')
                import numpy as np
                orientation=np.asarray(getattr(ds,'ImageOrientationPatient',[]),float)
                if len(orientation)!=6: raise ValueError('DICOM orientation is required')
                axis=int(np.argmax(np.abs(np.cross(orientation[:3],orientation[3:]))))
                folder=directory/str(uuid.uuid4());folder.mkdir()
                record=Series(id=series_id,study_id=study_id,plane=['Sagittal','Coronal','Axial'][axis],
                    relative_path=str(folder.relative_to(settings.data_root)),metadata_json={'source':'upload'})
                db.add(record);series[series_id]=(record,folder)
            record,folder=series[series_id]
            if record.study_id!=study_id: raise ValueError('Series appears under multiple studies')
            with archive.open(info) as src,(folder/(str(uuid.uuid4())+'.dcm')).open('wb') as dst:
                shutil.copyfileobj(src,dst)
        db.commit();return {'study_ids':list(studies),'series':len(series)}
