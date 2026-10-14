"""Exercise real PostgreSQL semantics; each test rolls back its outer transaction."""
import os
import uuid
import pytest
from fastapi.testclient import TestClient
from sqlalchemy.orm import Session
from app.db import engine,session
from app.main import app
from app.models import Study,LabelSet,Experiment
from app.research.constants import FINDINGS

pytestmark=pytest.mark.skipif(os.environ.get('KNEEWS_INTEGRATION')!='1',reason='Explicit local PostgreSQL integration run required')

@pytest.fixture
def client():
    with engine.connect() as connection:
        transaction=connection.begin()
        db=Session(bind=connection,join_transaction_mode='create_savepoint',expire_on_commit=False)
        def override():yield db
        app.dependency_overrides[session]=override
        try:
            with TestClient(app) as http:yield http,db
        finally:
            app.dependency_overrides.clear();db.close();transaction.rollback()


def make_case(db):
    key='qa-'+str(uuid.uuid4())
    db.add(Study(id=key,source='test',split='train',report='No ACL tear.',language='en',site_group='qa'))
    db.flush()
    labels={f:dict(state='unmentioned',evidence=[],probability=None) for f in FINDINGS}
    labels['ACL']=dict(state='absent',evidence=['No ACL tear.'],probability=.01)
    label=LabelSet(study_id=key,method='qa',version=key,findings=labels)
    db.add(label);db.commit();return key,label.id


def test_review_persists_and_rejects_invented_evidence(client):
    http,db=client;uid,label=make_case(db)
    payload=dict(study_id=uid,label_set_id=label,finding='ACL',state='absent',evidence='invented text',reviewer='QA Alice')
    assert http.post('/api/reviews',json=payload).status_code==422
    payload['evidence']='No ACL tear.'
    assert http.post('/api/reviews',json=payload).status_code==201
    payload['reviewer']='QA Bob'
    assert http.post('/api/reviews',json=payload).status_code==201
    metrics=http.get('/api/reviews/metrics').json()
    strata=[r for r in metrics['strata'] if r['version']==uid]
    assert len(strata)==1 and strata[0]['errors']==0 and strata[0]['reviewed']==1


def test_review_cannot_attach_to_another_study(client):
    http,db=client;uid,label=make_case(db);other,_=make_case(db)
    response=http.post('/api/reviews',json=dict(study_id=other,label_set_id=label,finding='ACL',state='absent',evidence='No ACL tear.',reviewer='QA'))
    assert response.status_code==422


def test_queued_jobs_are_deduplicated(client):
    http,db=client
    # Outer transaction makes these test jobs invisible to the running worker.
    response=http.post('/api/jobs',json={'kind':'label_rules'})
    assert response.status_code==201
    assert http.post('/api/jobs',json={'kind':'label_rules'}).status_code==409


def test_review_queue_uses_distinct_case_normalized_reviewers(client):
    http,db=client;uid,label=make_case(db)
    for name in ['QA Alice','qa alice']:
        response=http.post('/api/reviews',json=dict(study_id=uid,label_set_id=label,finding='ACL',state='absent',evidence='No ACL tear.',reviewer=name))
        assert response.status_code==201
    result=http.get('/api/reviews/queue',params={'version':uid}).json()
    assert result['total']==1 and result['items'][0]['reviewers']==1


def test_expert_targets_are_not_returned_to_viewer(client):
    http,db=client;uid,_=make_case(db)
    study=db.get(Study,uid);study.is_gold=True;study.gold={f:1 for f in FINDINGS};db.commit()
    response=http.get('/api/studies/'+uid)
    assert response.status_code==200 and 'gold' not in response.json()


def test_invalid_zip_returns_validation_error(client,tmp_path,monkeypatch):
    from app.config import settings
    monkeypatch.setattr(settings,'data_root',tmp_path)
    http,_=client
    response=http.post('/api/uploads',files={'file':('invalid.zip',b'not a zip','application/zip')})
    assert response.status_code==422


def test_study_predictions_require_completed_held_out_results(client):
    http,db=client;uid,_=make_case(db)
    experiment=Experiment(name='Synthetic prediction contract',status='running',config={},
                          manifest_hash='qa',results={})
    db.add(experiment);db.commit()
    endpoint=f'/api/experiments/{experiment.id}/studies/{uid}'
    assert http.get(endpoint).status_code==409
    experiment.status='completed'
    experiment.results=dict(plane_coverage={uid:2},runs=[
        dict(seed=2026,study_ids=[uid],predictions={'pooled':[[.1]*12]}),
        dict(seed=2027,study_ids=[uid],predictions={'pooled':[[.3]*12]})])
    db.commit()
    response=http.get(endpoint)
    assert response.status_code==200
    data=response.json()
    assert data['seeds']==2 and data['prediction_source']=='held_out_outer_folds'
    assert len(data['findings'])==12
    for row in data['findings']:
        assert row['probability']==pytest.approx(.2)
        assert row['seed_min']==.1 and row['seed_max']==.3
        assert row['seed_range_is_confidence_interval'] is False
        assert row['review_required'] is True
        assert 'Incomplete anatomical plane coverage' in row['reasons']
    assert http.get(endpoint,params={'arm':'unknown'}).status_code==404
    assert http.get(f'/api/experiments/{experiment.id}/studies/not-held-out').status_code==404


def test_blind_audit_import_is_atomic_and_idempotent(client,tmp_path):
    import json
    from sqlalchemy import select,func
    from app.models import Review
    from app.research.audit_packet import digest,import_packet
    _,db=client;uid,label_id=make_case(db)
    label=db.get(LabelSet,label_id);label.method='local_llm';db.commit()
    manifest=dict(schema='kneews-blind-audit-1',label_version=uid,
                  studies=[dict(study_id=uid,report_sha256=digest('No ACL tear.'))])
    packet_id=digest(json.dumps(manifest,sort_keys=True))
    worksheet=dict(packet_id=packet_id,reviewer='Independent QA',role='Test reviewer',languages_reviewed=['en'],
        studies=[dict(study_id=uid,report='No ACL tear.',confirmed_language='en',findings={
            f:dict(state='unmentioned',evidence='',note='') for f in FINDINGS})])
    worksheet['studies'][0]['findings']['ACL']=dict(state='absent',evidence='No ACL tear.',note='')
    # Error in the final cell must prevent every preceding valid cell from being inserted.
    worksheet['studies'][0]['findings'][FINDINGS[-1]]=dict(state='present',evidence='invented',note='')
    manifest_path=tmp_path/'manifest.json';form=tmp_path/'review.json'
    manifest_path.write_text(json.dumps(manifest));form.write_text(json.dumps(worksheet))
    with pytest.raises(ValueError,match='exact report quotation'):import_packet(db,manifest_path,form)
    assert db.scalar(select(func.count()).select_from(Review).where(Review.study_id==uid))==0
    worksheet['studies'][0]['findings'][FINDINGS[-1]]=dict(state='unmentioned',evidence='',note='')
    form.write_text(json.dumps(worksheet))
    assert import_packet(db,manifest_path,form)['imported']==12
    assert import_packet(db,manifest_path,form)['already_imported']==12
    worksheet['reviewer']='independent qa';form.write_text(json.dumps(worksheet))
    assert import_packet(db,manifest_path,form)['already_imported']==12
    worksheet['studies'][0]['findings']['ACL']['state']='uncertain';form.write_text(json.dumps(worksheet))
    with pytest.raises(ValueError,match='Conflicting existing review'):import_packet(db,manifest_path,form)


def test_external_inference_queue_requires_protocol_and_locks_it(client):
    from app.models import Series,Job
    http,db=client;uid,_=make_case(db)
    study=db.get(Study,uid);study.split='external';study.source='upload'
    series=Series(id='series-'+uid,study_id=uid,plane='NIfTI',format='nifti',relative_path='fixture.nii',metadata_json={})
    experiment=Experiment(name='QA completed fixture',status='completed',config={'arms':['pooled']},manifest_hash='qa')
    db.add_all([series,experiment]);db.commit()
    request=dict(kind='infer',study_id=uid,experiment_id=experiment.id,arm='pooled')
    assert http.post('/api/jobs',json=request).status_code==422
    protocol=dict(plane='Sagittal',fluid_sensitive=True,laterality='L',reviewer='QA reviewer')
    endpoint=f'/api/series/{series.id}/protocol'
    assert http.post(endpoint,json=protocol).status_code==200
    db.expire(study,['series'])
    response=http.post('/api/jobs',json=request)
    assert response.status_code==201
    assert http.post(endpoint,json=protocol).status_code==409
    job=db.get(Job,response.json()['id']);job.status='completed'
    job.result={'protocol':[dict(series_id=series.id,plane=series.plane,metadata=series.metadata_json)]};db.commit()
    assert http.get(f'/api/studies/{uid}/predictions').json()[0]['protocol_matches_current'] is True
    protocol['laterality']='R'
    assert http.post(endpoint,json=protocol).status_code==200
    assert http.get(f'/api/studies/{uid}/predictions').json()[0]['protocol_matches_current'] is False


def test_label_repair_never_replaces_human_reviewed_source(client,monkeypatch):
    from app.models import Review
    from app.research import labelling
    from app.research.readiness import repair_labels
    _,db=client;uid,label_id=make_case(db)
    label=db.get(LabelSet,label_id);label.method='local_llm'
    db.add(Review(study_id=uid,label_set_id=label_id,finding='ACL',state='absent',evidence='No ACL tear.',reviewer='QA human'))
    db.commit()
    monkeypatch.setattr(labelling,'installed_model',lambda *args:{'digest':'fixture'})
    monkeypatch.setattr(labelling,'local_label_version',lambda descriptor:uid)
    def must_not_run(*args):raise AssertionError('Reviewed source reached regeneration')
    monkeypatch.setattr(labelling,'label_local',must_not_run)
    with pytest.raises(ValueError,match='reviewed label set'):repair_labels(db,uid)
    assert db.get(LabelSet,label_id).provenance=={}


def test_review_completion_requires_twelve_cells_and_keeps_revisions(client):
    http,db=client;uid,label=make_case(db)
    def submit(finding,reviewer,state=None):
        body=dict(study_id=uid,label_set_id=label,finding=finding,reviewer=reviewer,
                  state=state or ('absent' if finding=='ACL' else 'unmentioned'),
                  evidence='No ACL tear.' if finding=='ACL' else '')
        return http.post('/api/reviews',json=body)
    assert submit('ACL','  ').status_code==422
    for name in ['Reviewer A','Reviewer B']:assert submit('ACL',name).status_code==201
    def queue():return http.get('/api/reviews/queue',params={'version':uid}).json()['items'][0]
    assert queue()['reviewers']==2 and queue()['two_reviewer_findings']==1
    assert queue()['fully_reviewed'] is False
    for finding in FINDINGS[1:]:
        for name in ['Reviewer A','Reviewer B']:assert submit(finding,name).status_code==201
    assert queue()['fully_reviewed'] is True and queue()['disputed_findings']==0
    assert submit('ACL','reviewer a',state='uncertain').status_code==201
    assert queue()['reviewers']==2 and queue()['disputed_findings']==1
    history=http.get(f'/api/studies/{uid}/reviews',params={'label_set_id':label}).json()
    assert history['total']==25 and history['items'][0]['state']=='uncertain'
    assert sum(r['finding']=='ACL' for r in history['items'])==3
    assert history['independent_identity_verified'] is False


def test_stage_lease_excludes_other_connection_and_releases_after_failure(client):
    from app.research.leases import stage_lease,stage_active
    _,db=client
    name='qa-'+str(uuid.uuid4())
    with pytest.raises(ValueError,match='test interruption'):
        with stage_lease(name):
            assert stage_active(db,name) is True
            with pytest.raises(RuntimeError,match='already running'):
                with stage_lease(name):pass
            raise ValueError('test interruption')
    with stage_lease(name):pass
    assert stage_active(db,name) is False


def test_overview_local_coverage_is_specific_to_configured_prompt_and_model(client,monkeypatch):
    from app.config import settings
    from app.research.labelling import PROMPT_HASH,SCHEMA_VERSION
    http,db=client;_,first=make_case(db);_,second=make_case(db)
    monkeypatch.setattr(settings,'local_llm_model','qa-model')
    current=db.get(LabelSet,first);current.method='local_llm';current.version=f'qa-model:123456789abc:{PROMPT_HASH[:12]}:{SCHEMA_VERSION}'
    stale=db.get(LabelSet,second);stale.method='local_llm';stale.version='qa-model:123456789abc:old-prompt:s0'
    db.commit()
    data=http.get('/api/overview').json()
    assert data['local_labelled_studies']==1 and data['local_label_version']==current.version
    assert data['labelled_studies']>data['local_labelled_studies']
