import csv
import hashlib
from pathlib import Path
from collections import Counter
import pydicom
from sqlalchemy import select
from .config import settings
from .models import Study,Series
from .research.constants import FINDINGS
from .research.splits import report_hash


def import_rsna(db,progress=lambda p,m:None):
    root=settings.data_root
    for split in ('train','test'):
        with (root/f'{split}.csv').open() as f: rows=list(csv.DictReader(f))
        for i,row in enumerate(rows):
            study=db.get(Study,row['StudyInstanceUID'])
            if study is None:
                gold={k:int(float(row[k])) for k in FINDINGS if row.get(k) not in ('',None)}
                report=row.get('Report','')
                study=Study(id=row['StudyInstanceUID'],split=split,report=report,
                            report_hash=report_hash(report),gold=gold,is_gold=len(gold)==12)
                db.add(study)
            if i%200==0: db.flush()
        db.commit()
        with (root/f'{split}_series.csv').open() as f: rows=list(csv.DictReader(f))
        existing=set(db.scalars(select(Series.id)).all())
        for i,row in enumerate(rows):
            if row['SeriesInstanceUID'] not in existing:
                db.add(Series(id=row['SeriesInstanceUID'],study_id=row['StudyInstanceUID'],
                    plane=row['Anatomical_Plane'],relative_path=f"{split}_series/{row['StudyInstanceUID']}/{row['SeriesInstanceUID']}",
                    metadata_json={'fluid_sensitive_csv':row['Fluid_Sensitive'],'fat_suppression_csv':row['Fat_Suppression']}))
            if i%500==0: db.commit(); progress(i/max(1,len(rows)),f'Importing {split} series {i}/{len(rows)}')
        db.commit()
    return {'studies':len(db.scalars(select(Study.id)).all()),'series':len(db.scalars(select(Series.id)).all())}


def profile_headers(db,progress=lambda p,m:None):
    studies=db.scalars(select(Study).order_by(Study.id)).all()
    available=missing=0
    for i,study in enumerate(studies):
        fingerprints=[]; descriptions=[]; laterality=[]
        for series in study.series:
            path=settings.data_root/series.relative_path
            file=next(path.glob('*.dcm'),None)
            if file is None: missing+=1; continue
            ds=pydicom.dcmread(file,stop_before_pixels=True)
            meta={k:str(getattr(ds,k,'')) for k in ('Manufacturer','ManufacturerModelName','MagneticFieldStrength',
                    'Laterality','ImageLaterality','SeriesDescription','RepetitionTime','EchoTime','InstitutionName','StationName')}
            meta['sample_file']=file.name
            meta['geometry_available']=all(hasattr(ds,k) for k in ('ImagePositionPatient','ImageOrientationPatient','PixelSpacing'))
            meta['pixel_spacing']=[float(x) for x in getattr(ds,'PixelSpacing',[])]
            meta['shape']=[int(getattr(ds,'Rows',0)),int(getattr(ds,'Columns',0))]
            series.metadata_json={**series.metadata_json,**meta}
            scanner='|'.join(meta[k].strip().casefold() for k in ('Manufacturer','ManufacturerModelName','MagneticFieldStrength'))
            if all(meta[k] for k in ('Manufacturer','ManufacturerModelName','MagneticFieldStrength')):
                fingerprints.append(scanner)
            descriptions.append(meta['SeriesDescription']); laterality.append(meta['ImageLaterality'] or meta['Laterality'])
            available+=1
        # Conflicting fingerprints remain unknown rather than accidentally splitting one institution.
        if fingerprints and len(set(fingerprints))==1:
            study.site_group='scanner-'+hashlib.sha256(fingerprints[0].encode()).hexdigest()[:16]
            study.group_basis='scanner_proxy'
        else:
            study.site_group='unknown'; study.group_basis='unknown'
        study.metadata_json={**study.metadata_json,'series_descriptions':descriptions,'laterality':sorted(set(laterality)),
                             'scanner_fingerprints':sorted(set(fingerprints))}
        if i%25==0:
            db.commit(); progress((i+1)/len(studies),f'Profiled {i+1}/{len(studies)} studies')
    db.commit()
    return dict(series_profiled=available,series_missing=missing,group_basis='scanner_proxy',hospital_identity_verified=False)
