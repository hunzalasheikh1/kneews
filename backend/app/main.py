import csv
import io
import json
import zipfile
from pydicom.errors import InvalidDicomError
from functools import lru_cache
from typing import Literal
from fastapi import FastAPI,Depends,HTTPException,Query,UploadFile,File
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import Response
from pydantic import BaseModel,Field
from sqlalchemy import select,func,text
from sqlalchemy.orm import Session
from .config import settings
from .db import session
from .models import Study,Series,Job,LabelSet,Review,Experiment
from .research.constants import FINDINGS,STATES
from .research.imaging import read_dicom_series,load_nifti,slice_png

app=FastAPI(title='KneeWS Research API',version='0.1.0')
app.add_middleware(CORSMiddleware,allow_origins=[settings.cors_origin,'http://localhost:3200'],allow_methods=['GET','POST'],allow_headers=['Content-Type'])

@app.get('/api/health')
def health(db:Session=Depends(session)):
    db.execute(text('SELECT 1')); return {'status':'ok','database':'postgresql'}

@app.get('/api/overview')
def overview(db:Session=Depends(session)):
    from .research.views import label_versions
    from .research.labelling import PROMPT_HASH,SCHEMA_VERSION
    versions=label_versions(db)
    local=next((v for v in versions if v['method']=='local_llm' and v['version'].startswith(settings.local_llm_model+':')
                and v['version'].endswith(':'+PROMPT_HASH[:12]+':'+SCHEMA_VERSION)),None)
    status=settings.data_root/'extraction-status.json'
    return dict(studies=db.scalar(select(func.count()).select_from(Study)),
        series=db.scalar(select(func.count()).select_from(Series)),
        gold_studies=db.scalar(select(func.count()).select_from(Study).where(Study.is_gold)),
        labelled_studies=db.scalar(select(func.count(func.distinct(LabelSet.study_id)))),
        local_labelled_studies=local['studies'] if local else 0,local_label_version=local['version'] if local else None,
        training_studies=db.scalar(select(func.count()).select_from(Study).where(Study.split=='train')),
        rule_labelled_studies=max((v['studies'] for v in versions if v['method']=='rules'),default=0),
        reviews=db.scalar(select(func.count()).select_from(Review)),
        extraction=json.loads(status.read_text()) if status.exists() else None,
        groups=[{'basis':basis,'count':count} for basis,count in db.execute(select(Study.group_basis,func.count()).group_by(Study.group_basis))],
        findings=FINDINGS,states=STATES,research_only=True)

@app.get('/api/research/status')
def research_status(db:Session=Depends(session)):
    from .research.views import label_versions
    from .research.leases import stage_active
    audit=settings.artifact_root/'audit'/'dataset.json'
    manifests=[]
    for path in (settings.artifact_root/'features').glob('*/manifest.json'):
        try: manifests.append({**json.loads(path.read_text()),'recorded_at':path.stat().st_mtime})
        except (OSError,ValueError): continue
    label_runs=[]
    for path in (settings.artifact_root/'labels').glob('*/*.json'):
        try:
            value=json.loads(path.read_text())
            if value.get('selection')=='full_corpus':
                label_runs.append({**{k:value.get(k) for k in ['version','status','total','completed','skipped','workers','started_at','updated_at']},
                                   'failed':len(value.get('failures',[]))})
        except (OSError,ValueError):continue
    latest=max(label_runs,key=lambda r:r['updated_at'] or 0) if label_runs else None
    return dict(audit=json.loads(audit.read_text()) if audit.exists() else None,feature_caches=manifests,
                label_versions=label_versions(db),latest_label_run=latest,
                local_label_stage_active=stage_active(db,'local-label-generation'))

@app.post('/api/uploads',status_code=201)
def upload(file:UploadFile=File(...),db:Session=Depends(session)):
    from .uploads import ingest_upload
    try: return ingest_upload(file,db)
    except (ValueError,OSError,zipfile.BadZipFile,InvalidDicomError) as exc:
        db.rollback();raise HTTPException(422,str(exc))

@app.get('/api/studies')
def studies(q:str='',gold:bool|None=None,offset:int=Query(0,ge=0),limit:int=Query(40,ge=1,le=100),db:Session=Depends(session)):
    query=select(Study)
    if q: query=query.where(Study.id.contains(q))
    if gold is not None: query=query.where(Study.is_gold==gold)
    total=db.scalar(select(func.count()).select_from(query.subquery()))
    items=db.scalars(query.order_by(Study.id).offset(offset).limit(limit)).all()
    return dict(total=total,items=[dict(id=s.id,split=s.split,language=s.language,site_group=s.site_group,
                                     group_basis=s.group_basis,is_gold=s.is_gold,series_count=len(s.series)) for s in items])

@app.get('/api/studies/{study_id}')
def study_detail(study_id:str,db:Session=Depends(session)):
    s=db.get(Study,study_id)
    if not s: raise HTTPException(404,'Study not found')
    labels=db.scalars(select(LabelSet).where(LabelSet.study_id==study_id).order_by(LabelSet.created_at.desc())).all()
    return dict(id=s.id,split=s.split,report=s.report,language=s.language,is_gold=s.is_gold,site_group=s.site_group,
                group_basis=s.group_basis,metadata=s.metadata_json,
                series=[dict(id=x.id,plane=x.plane,format=x.format,metadata=x.metadata_json) for x in s.series],
                labels=[dict(id=x.id,method=x.method,version=x.version,findings=x.findings,provenance=x.provenance) for x in labels])

class ProtocolRequest(BaseModel):
    plane:Literal['Sagittal','Coronal','Axial']
    fluid_sensitive:bool
    laterality:Literal['L','R']
    reviewer:str=Field(min_length=2,max_length=128)

@app.post('/api/series/{series_id}/protocol')
def set_protocol(series_id:str,body:ProtocolRequest,db:Session=Depends(session)):
    row=db.get(Series,series_id)
    if row is None:raise HTTPException(404,'Series not found')
    study=db.get(Study,row.study_id)
    if study.split!='external' or study.source!='upload':raise HTTPException(422,'Only external upload protocols can be edited')
    db.execute(text('SELECT pg_advisory_xact_lock(hashtext(:key))'),{'key':'protocol-'+study.id})
    active=db.scalar(select(Job).where(Job.kind=='infer',Job.status.in_(['queued','running']),Job.config['study_id'].astext==study.id))
    if active:raise HTTPException(409,'Wait for the active prediction job before changing its protocol')
    if row.format=='dicom' and row.plane!=body.plane:raise HTTPException(422,'DICOM plane must match the acquired orientation')
    if not body.reviewer.strip():raise HTTPException(422,'A reviewer name is required')
    row.plane=body.plane
    row.metadata_json={**row.metadata_json,**body.model_dump(exclude={'plane'}),'protocol_reviewed':True}
    db.commit();return dict(series_id=row.id,plane=row.plane,metadata=row.metadata_json)

@app.get('/api/studies/{study_id}/predictions')
def external_predictions(study_id:str,db:Session=Depends(session)):
    study=db.get(Study,study_id)
    if study is None:raise HTTPException(404,'Study not found')
    protocol=[dict(series_id=s.id,plane=s.plane,metadata=s.metadata_json) for s in study.series]
    normalize=lambda rows:sorted(rows,key=lambda r:r['series_id'])
    return [dict(id=j.id,status=j.status,progress=j.progress,error=j.error,message=j.message,config=j.config,
                 result=j.result,created_at=j.created_at,
                 protocol_matches_current=normalize(j.result.get('protocol',[]))==normalize(protocol))
            for j in db.scalars(select(Job).where(Job.kind=='infer',Job.config['study_id'].astext==study_id)
                                .order_by(Job.created_at.desc()).limit(10))]

@lru_cache(maxsize=3)
def cached_volume(relative_path,format):
    path=(settings.data_root/relative_path).resolve()
    if not path.is_relative_to(settings.data_root.resolve()): raise ValueError('Invalid volume path')
    if format=='nifti': return load_nifti(path)
    return read_dicom_series(sorted(path.glob('*.dcm')))[0]

@app.get('/api/series/{series_id}/volume')
def volume_info(series_id:str,db:Session=Depends(session)):
    s=db.get(Series,series_id)
    if not s: raise HTTPException(404,'Series not found')
    try:
        image=cached_volume(s.relative_path,s.format)
        return dict(shape=list(image.shape),spacing=[float(x) for x in image.header.get_zooms()],orientation='RAS')
    except (ValueError,FileNotFoundError) as exc: raise HTTPException(422,str(exc))

@app.get('/api/series/{series_id}/slice')
def get_slice(series_id:str,index:int=Query(0,ge=0),axis:int=Query(2,ge=0,le=2),db:Session=Depends(session)):
    s=db.get(Series,series_id)
    if not s: raise HTTPException(404,'Series not found')
    try: return Response(slice_png(cached_volume(s.relative_path,s.format),index,axis),media_type='image/png')
    except (ValueError,FileNotFoundError) as exc: raise HTTPException(422,str(exc))

class JobRequest(BaseModel):
    kind:Literal['import','profile','label_rules','label_local','infer']
    limit:int|None=Field(None,ge=1,le=4407)
    study_id:str|None=None
    experiment_id:str|None=None
    arm:Literal['binary','masked','pooled','llm_probability','reserve']='pooled'
    model:Literal['experiment','final']='experiment'

@app.post('/api/jobs',status_code=201)
def create_job(body:JobRequest,db:Session=Depends(session)):
    if body.kind=='label_local':
        available=db.scalar(text('SELECT pg_try_advisory_xact_lock(hashtextextended(:key,0))'),{'key':'kneews-stage-local-label-generation'})
        if not available:raise HTTPException(409,'Local report labelling is already running from another process')
    if body.kind=='infer' and body.model=='final':
        from .research.inference import validate_final_target,final_model_root
        db.execute(text('SELECT pg_advisory_xact_lock(hashtext(:key))'),{'key':'protocol-'+str(body.study_id)})
        try:validate_final_target(db.get(Study,body.study_id) if body.study_id else None);final_model_root()
        except ValueError as exc:raise HTTPException(422,str(exc))
    elif body.kind=='infer':
        from .research.inference import validate_external
        db.execute(text('SELECT pg_advisory_xact_lock(hashtext(:key))'),{'key':'protocol-'+str(body.study_id)})
        try:validate_external(db.get(Study,body.study_id) if body.study_id else None)
        except ValueError as exc:raise HTTPException(422,str(exc))
        experiment=db.get(Experiment,body.experiment_id) if body.experiment_id else None
        if experiment is None or experiment.status!='completed':raise HTTPException(422,'Select a completed experiment')
        if body.arm not in experiment.config.get('arms',[]):raise HTTPException(422,'Arm not available in the experiment')
    # Serialize creation by kind to make the duplicate-active-job check race-safe.
    db.execute(text('SELECT pg_advisory_xact_lock(hashtext(:kind))'),{'kind':'kneews-job-'+body.kind})
    existing=db.scalar(select(Job).where(Job.kind==body.kind,Job.status.in_(['queued','running'])))
    if existing: raise HTTPException(409,'A job of this kind is already active')
    job=Job(kind=body.kind,config=body.model_dump(exclude={'kind'},exclude_none=True)); db.add(job); db.commit(); db.refresh(job)
    return {'id':job.id,'status':job.status}

@app.get('/api/jobs')
def jobs(db:Session=Depends(session)):
    return [dict(id=j.id,kind=j.kind,status=j.status,progress=j.progress,message=j.message,error=j.error,
                 config=j.config,result=j.result,created_at=j.created_at,heartbeat_at=j.heartbeat_at) for j in db.scalars(select(Job).order_by(Job.created_at.desc()).limit(50))]

class ReviewRequest(BaseModel):
    study_id:str
    label_set_id:str|None=None
    finding:str
    state:Literal['present','absent','uncertain','unmentioned']
    evidence:str=''
    reviewer:str=Field(min_length=2,max_length=128)
    note:str=''

@app.post('/api/reviews',status_code=201)
def review(body:ReviewRequest,db:Session=Depends(session)):
    if len(body.reviewer.strip())<2:raise HTTPException(422,'A reviewer name of at least two non-whitespace characters is required')
    study=db.get(Study,body.study_id)
    if not study or body.finding not in FINDINGS: raise HTTPException(422,'Invalid study or finding')
    if body.evidence and body.evidence not in study.report: raise HTTPException(422,'Evidence must quote the report')
    if body.state!='unmentioned' and not body.evidence.strip(): raise HTTPException(422,'A mentioned state requires report evidence')
    if body.state=='unmentioned' and body.evidence: raise HTTPException(422,'Unmentioned state cannot quote evidence')
    if body.label_set_id:
        label=db.get(LabelSet,body.label_set_id)
        if not label or label.study_id!=body.study_id: raise HTTPException(422,'Label set belongs to another study')
    row=Review(**body.model_dump()); db.add(row); db.commit(); return {'id':row.id}

@app.get('/api/studies/{study_id}/reviews')
def review_history(study_id:str,label_set_id:str|None=None,offset:int=Query(0,ge=0),limit:int=Query(30,ge=1,le=100),db:Session=Depends(session)):
    if db.get(Study,study_id) is None:raise HTTPException(404,'Study not found')
    query=select(Review).where(Review.study_id==study_id)
    if label_set_id:query=query.where(Review.label_set_id==label_set_id)
    total=db.scalar(select(func.count()).select_from(query.subquery()))
    rows=db.scalars(query.order_by(Review.created_at.desc(),Review.id.desc()).offset(offset).limit(limit)).all()
    return dict(total=total,items=[dict(id=r.id,label_set_id=r.label_set_id,finding=r.finding,state=r.state,
        evidence=r.evidence,reviewer=r.reviewer,note=r.note,created_at=r.created_at) for r in rows],
        independent_identity_verified=False)

@app.get('/api/experiments')
def experiments(db:Session=Depends(session)):
    from .research.views import experiment_summary
    return [experiment_summary(e) for e in db.scalars(select(Experiment).order_by(Experiment.created_at.desc()))]

@app.get('/api/experiments/{experiment_id}')
def experiment_detail(experiment_id:str,db:Session=Depends(session)):
    experiment=db.get(Experiment,experiment_id)
    if experiment is None:raise HTTPException(404,'Experiment not found')
    return dict(id=experiment.id,name=experiment.name,status=experiment.status,config=experiment.config,
                results=experiment.results,manifest_hash=experiment.manifest_hash)

@app.get('/api/experiments/{experiment_id}/reliability')
def reliability_table(experiment_id:str,seed:int=2026,fold:int=Query(0,ge=0,le=4),db:Session=Depends(session)):
    experiment=db.get(Experiment,experiment_id)
    if experiment is None:raise HTTPException(404,'Experiment not found')
    if experiment.status!='completed':raise HTTPException(409,'The experiment has not completed')
    run=next((r for r in experiment.results.get('runs',[]) if r['seed']==seed),None)
    estimator=next((r for r in (run or {}).get('estimators',[]) if r['fold']==fold),None)
    if estimator is None:raise HTTPException(404,'Reliability table unavailable for this seed and fold')
    return dict(fold=fold,seed=seed,table=estimator['table'],fit_studies=len(estimator['fit_ids']),
                interval_description='95% conditional beta posterior; empirical prior uncertainty is not included')

@app.get('/api/experiments/{experiment_id}/studies/{study_id}')
def study_predictions(experiment_id:str,study_id:str,arm:str='pooled',db:Session=Depends(session)):
    from .research.abstention import summarize_seed_predictions
    experiment=db.get(Experiment,experiment_id)
    if experiment is None:raise HTTPException(404,'Experiment not found')
    if experiment.status!='completed':raise HTTPException(409,'The experiment has not completed')
    predictions=[]
    for run in experiment.results.get('runs',[]):
        if study_id not in run['study_ids'] or arm not in run['predictions']:continue
        predictions.append(run['predictions'][arm][run['study_ids'].index(study_id)])
    if not predictions:raise HTTPException(404,'No held-out prediction exists for this study and arm')
    coverage=experiment.results.get('plane_coverage',{}).get(study_id,0)
    result=summarize_seed_predictions(predictions,coverage)
    for name,row in zip(FINDINGS,result['findings']):row['finding']=name
    return dict(study_id=study_id,arm=arm,seeds=len(predictions),prediction_source='held_out_outer_folds',**result)

@app.get('/api/labels/versions')
def versions(db:Session=Depends(session)):
    from .research.views import label_versions
    return label_versions(db)

@app.get('/api/labels/disagreements')
def label_disagreements(left:str,right:str,language:str|None=None,offset:int=Query(0,ge=0),limit:int=Query(30,ge=1,le=100),db:Session=Depends(session)):
    from .research.views import disagreements
    if left==right:raise HTTPException(422,'Select two different label versions')
    return disagreements(db,left,right,language,offset,limit)

@app.get('/api/reviews/queue')
def queue(version:str,language:str|None=None,offset:int=Query(0,ge=0),limit:int=Query(30,ge=1,le=100),db:Session=Depends(session)):
    from .research.views import review_queue
    return review_queue(db,version,language,offset,limit)

@app.get('/api/reviews/metrics')
def audit_reviews(db:Session=Depends(session)):
    from .research.review_audit import review_metrics
    reviews=db.scalars(select(Review)).all()
    identifiers={r.label_set_id for r in reviews if r.label_set_id}
    labels={r.id:r for r in db.scalars(select(LabelSet).where(LabelSet.id.in_(identifiers)))}
    return review_metrics(reviews,labels)

@app.get('/api/experiments/{experiment_id}/export')
def export_experiment(experiment_id:str,db:Session=Depends(session)):
    experiment=db.get(Experiment,experiment_id)
    if experiment is None:raise HTTPException(404,'Experiment not found')
    payload=dict(id=experiment.id,name=experiment.name,status=experiment.status,manifest_hash=experiment.manifest_hash,
                 config=experiment.config,results=experiment.results)
    return Response(json.dumps(payload,indent=2),media_type='application/json',
                    headers={'Content-Disposition':f'attachment; filename=kneews-experiment-{experiment.id}.json'})

@app.get('/api/experiments/{experiment_id}/report')
def experiment_report(experiment_id:str,db:Session=Depends(session)):
    from .research.reporting import report_html
    experiment=db.get(Experiment,experiment_id)
    if experiment is None:raise HTTPException(404,'Experiment not found')
    try:content=report_html(experiment)
    except ValueError as exc:raise HTTPException(409,str(exc))
    return Response(content,media_type='text/html',headers={'Content-Security-Policy':"default-src 'none'; style-src 'unsafe-inline'"})

@app.get('/api/experiments/{experiment_id}/tables')
def experiment_tables(experiment_id:str,db:Session=Depends(session)):
    from .research.reporting import report_bundle
    experiment=db.get(Experiment,experiment_id)
    if experiment is None:raise HTTPException(404,'Experiment not found')
    try:content=report_bundle(experiment)
    except ValueError as exc:raise HTTPException(409,str(exc))
    return Response(content,media_type='application/zip',
                    headers={'Content-Disposition':f'attachment; filename=kneews-tables-{experiment.id}.zip'})

@app.get('/api/export/labels.csv')
def export_labels(db:Session=Depends(session)):
    buffer=io.StringIO(); writer=csv.writer(buffer); writer.writerow(['study_id','method','version','finding','state','probability','evidence'])
    for labels in db.scalars(select(LabelSet).order_by(LabelSet.study_id)):
        for finding,row in labels.findings.items(): writer.writerow([labels.study_id,labels.method,labels.version,finding,row['state'],row.get('probability'),json.dumps(row['evidence'],ensure_ascii=False)])
    return Response(buffer.getvalue(),media_type='text/csv',headers={'Content-Disposition':'attachment; filename=kneews-labels.csv'})


@app.get('/api/final-model')
def final_model(db:Session=Depends(session)):
    from .research.inference import final_model_root,reference_performance
    from .research.abstention import POLICY
    try:root=final_model_root()
    except ValueError as exc:return dict(available=False,reason=str(exc))
    config=json.loads((root/'manifest.json').read_text());status=json.loads((root/'status.json').read_text())
    return dict(available=True,id=root.name,arm=config['arm'],seeds=config['seeds'],training_studies=config['training_studies'],
                expert_target_studies=config['expert_target_studies'],finished_at=status.get('finished_at'),
                parent_experiment_id=config['parent_experiment_id'],review_policy=POLICY,
                reference_performance=reference_performance(db,config['parent_experiment_id']),
                independent_evaluation_available=False,clinically_validated=False)

@app.get('/api/clinical/studies')
def clinical_studies(db:Session=Depends(session)):
    """Studies eligible for final-model prediction: held-out example tests and external uploads."""
    rows=db.scalars(select(Study).where(Study.split.in_(['test','external'])).order_by(Study.split,Study.id)).all()
    jobs=db.scalars(select(Job).where(Job.kind=='infer').order_by(Job.created_at.desc())).all()
    latest={}
    for j in jobs:
        if j.config.get('model')=='final':latest.setdefault(j.config.get('study_id'),j)
    out=[]
    for s in rows:
        j=latest.get(s.id)
        out.append(dict(id=s.id,split=s.split,source=s.source,series=[dict(id=x.id,plane=x.plane,format=x.format,metadata=x.metadata_json) for x in s.series],
                        original_name=(s.metadata_json or {}).get('original_name'),
                        latest_job=dict(id=j.id,status=j.status,progress=j.progress,message=j.message,error=j.error,result=j.result,created_at=j.created_at) if j else None))
    return out
