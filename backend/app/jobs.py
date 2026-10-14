"""PostgreSQL-backed single job worker with atomic SKIP LOCKED claims."""
import time
import traceback
from contextlib import ExitStack
from sqlalchemy import select,text
from .db import SessionLocal,engine
from .models import Job,now,Study,LabelSet
from .config import settings
from .ingestion import import_rsna,profile_headers
from .research.labelling import rule_labels,RULE_VERSION,label_local,PROMPT_HASH,SCHEMA_VERSION


def execute(job_id):
    with SessionLocal() as db,ExitStack() as leases:
        job=db.get(Job,job_id)
        def progress(value,message):
            with SessionLocal() as update:
                obj=update.get(Job,job_id); obj.progress=value; obj.message=message; obj.heartbeat_at=now(); update.commit()
        try:
            if job.kind=='import': result=import_rsna(db,progress)
            elif job.kind=='profile': result=profile_headers(db,progress)
            elif job.kind=='infer':
                if job.config.get('model')=='final':
                    from .research.inference import infer_final_study
                    result=infer_final_study(db,job.config['study_id'],progress)
                else:
                    from .research.inference import infer_study
                    result=infer_study(db,job.config['study_id'],job.config['experiment_id'],job.config.get('arm','pooled'),progress)
            elif job.kind in ('label_rules','label_local'):
                if job.kind=='label_local':
                    from .research.leases import stage_lease
                    leases.enter_context(stage_lease('local-label-generation'))
                studies=db.scalars(select(Study).where(Study.split=='train').order_by(Study.id)).all()
                if job.config.get('limit'): studies=studies[:job.config['limit']]
                method='rules' if job.kind=='label_rules' else 'local_llm'
                version=RULE_VERSION
                if method=='local_llm':
                    import httpx
                    with httpx.Client(trust_env=False) as client:
                        response=client.get(settings.local_llm_url+'/api/tags');response.raise_for_status()
                        model=next((x for x in response.json()['models'] if x['name']==settings.local_llm_model),None)
                    if model is None: raise ValueError('Configured local model is not installed')
                    version=settings.local_llm_model+':'+model['digest'][:12]+':'+PROMPT_HASH[:12]+':'+SCHEMA_VERSION
                completed=0
                for i,study in enumerate(studies):
                    existing=db.scalar(select(LabelSet).where(LabelSet.study_id==study.id,LabelSet.method==method,LabelSet.version==version))
                    if existing is None:
                        if method=='rules': labels=rule_labels(study.report); provenance=dict(version=version,audited=False)
                        else: labels,provenance=label_local(study.report,settings.local_llm_url,settings.local_llm_model)
                        db.add(LabelSet(study_id=study.id,method=method,version=version,findings=labels,provenance=provenance)); db.commit()
                    completed+=1
                    if i%10==0: progress((i+1)/len(studies),f'Labelled {i+1}/{len(studies)} reports')
                result=dict(completed=completed,method=method,version=version)
            else: raise ValueError('Unknown job kind')
            db.refresh(job); job.status='completed'; job.progress=1; job.result=result; job.message='Completed'; job.finished_at=now(); db.commit()
        except Exception as exc:
            db.rollback(); job=db.get(Job,job_id); job.status='failed'; job.error=str(exc); job.message='Failed'; job.finished_at=now(); db.commit()
            traceback.print_exc()


def main():
    # The session-held advisory lock proves that no previous worker remains alive.
    with engine.connect() as lease:
        acquired=lease.scalar(text('SELECT pg_try_advisory_lock(782914620)'));lease.commit()
        if not acquired: raise RuntimeError('Another KneeWS worker is already running')
        try:
            with SessionLocal() as db:
                for stale in db.scalars(select(Job).where(Job.status=='running')):
                    stale.status='interrupted';stale.message='Worker stopped before completion. Start the idempotent stage again.'
                    stale.finished_at=now()
                db.commit()
            while True:
                with SessionLocal() as db:
                    job=db.scalar(select(Job).where(Job.status=='queued').order_by(Job.created_at).with_for_update(skip_locked=True).limit(1))
                    if job:
                        job.status='running'; job.started_at=now(); job.heartbeat_at=now(); identifier=job.id; db.commit()
                    else: identifier=None
                if identifier: execute(identifier)
                else: time.sleep(2)
        finally:
            lease.execute(text('SELECT pg_advisory_unlock(782914620)'));lease.commit()

if __name__=='__main__': main()
