"""All-arm serialization/checkpoint exercise in a rollback-only private schema.

Synthetic outcomes never enter public research tables or persistent artifacts.
One epoch/seed and reduced resampling test orchestration, not scientific quality.
"""
import hashlib
import io
import json
import os
import uuid
import zipfile
from pathlib import Path
import numpy as np
import pytest
import torch
from sqlalchemy import text,select
from sqlalchemy.orm import Session
from app.db import engine,Base
from app.models import Study,LabelSet,Experiment
from app.config import settings
from app.research.constants import FINDINGS,STATES
from app.research import experiments
from app.research.reporting import report_bundle
from app.research.inference import ensemble

pytestmark=pytest.mark.skipif(os.environ.get('KNEEWS_INTEGRATION')!='1',reason='Explicit local PostgreSQL integration run required')


@pytest.fixture
def isolated_experiment_db(tmp_path,monkeypatch):
    monkeypatch.setattr(settings,'artifact_root',tmp_path)
    schema='qa_experiment_'+uuid.uuid4().hex
    with engine.connect() as connection:
        transaction=connection.begin()
        connection.execute(text(f'CREATE SCHEMA "{schema}"'))
        connection.execute(text(f'SET LOCAL search_path TO "{schema}"'))
        Base.metadata.create_all(connection)
        db=Session(bind=connection,join_transaction_mode='create_savepoint',expire_on_commit=False)
        try:yield db,tmp_path
        finally:db.close();transaction.rollback()


def populate(db,root):
    encoder={'version':'synthetic-fixture'}
    signature=hashlib.sha256(json.dumps(encoder,sort_keys=True).encode()).hexdigest()
    directory=root/'features'/signature;directory.mkdir(parents=True)
    version='synthetic:model:'+('a'*12)+':'+('b'*12)+':s3'
    rng=np.random.default_rng(11)
    for group in range(5):
        for index in range(10):
            uid=f'synthetic-{group}-{index}';report=f'Synthetic report {uid}.';gold=index<2
            report_hash=hashlib.sha256(report.encode()).hexdigest()
            db.add(Study(id=uid,source='synthetic',split='train',report=report,report_hash=report_hash,
                         is_gold=gold,gold={f:index%2 for f in FINDINGS} if gold else {},site_group=f'group-{group}',language='fixture'))
            db.flush()
            state=STATES[index%4];probability=[.9,.1,.5,None][index%4]
            db.add(LabelSet(study_id=uid,method='local_llm',version=version,
                findings={f:dict(state=state,probability=probability,evidence=[] if state=='unmentioned' else [report]) for f in FINDINGS},
                provenance=dict(model='synthetic:model',digest='a'*64,prompt_sha256='b'*64,schema_version='s3',local_only=True,report_sha256=report_hash)))
            metadata=dict(study_id=uid,signature=signature,encoder=encoder,sources=[dict(slot=i,series_id=f'{uid}-{i}') for i in range(6)])
            np.savez(directory/(uid+'.npz'),features=rng.normal(size=(6,768)).astype(np.float32),mask=np.ones(6,bool),metadata=json.dumps(metadata))
    db.commit();return version,signature


@pytest.mark.parametrize('training_device',['cpu','mps'])
def test_all_arms_checkpoint_postgres_export_and_inference(isolated_experiment_db,monkeypatch,training_device):
    if training_device=='mps' and not torch.backends.mps.is_available():
        pytest.skip('Apple GPU required for the MPS integration case')
    db,root=isolated_experiment_db;version,signature=populate(db,root)
    label=db.scalar(select(LabelSet).where(LabelSet.study_id=='synthetic-0-2'))
    label.provenance={**label.provenance,'evidence_repair':{'version':'synthetic-repair-policy','segment_count':1}}
    db.commit()
    bootstrap=experiments.paired_group_bootstrap
    monkeypatch.setattr(experiments,'paired_group_bootstrap',lambda *a,**kw:bootstrap(*a,**kw,iterations=10))
    threads=torch.get_num_threads();torch.set_num_threads(2)
    try:identifier=experiments.run_experiment(db,version,signature,seeds=(2026,),epochs=1,arms=(*experiments.ARMS,'reserve'),device=training_device)
    finally:torch.set_num_threads(threads)
    db.expire_all();row=db.get(Experiment,identifier)
    assert row.status=='completed' and not row.config['registered_primary_protocol_match']
    assert row.config['training_device']['resolved']==training_device
    assert row.config['label_repairs']==[dict(study_id='synthetic-0-2',version='synthetic-repair-policy',segment_count=1)]
    run=row.results['runs'][0]
    assert set(run['metrics'])==set((*experiments.ARMS,'reserve'))
    assert len(run['study_ids'])==10 and len(run['estimators'])==5 and len(run['reserve_estimators'])==5
    assert len(run['subgroups'])==6 and len(run['omission'])==24
    output=root/'experiments'/identifier
    assert len(list(output.glob('*.pt')))==25
    for path in output.glob('*-fold*.json'):
        assert json.loads(path.read_text())['training_device'].split(':')[0]==training_device
    source=row.config['source_snapshot']
    snapshot=(output/source['filename']).read_bytes()
    assert hashlib.sha256(snapshot).hexdigest()==source['sha256']
    with zipfile.ZipFile(io.BytesIO(snapshot)) as archived:
        assert set(archived.namelist())==set(source['files'])
        assert 'backend/app/research/training.py' in archived.namelist()
        assert 'docs/research/PREREGISTRATION.json' in archived.namelist()
        assert not any('.env' in name or name.startswith(('data/','artifacts/')) for name in archived.namelist())
        for name,digest in source['files'].items():
            assert hashlib.sha256(archived.read(name)).hexdigest()==digest
        for name,digest in row.config['research_source_sha256'].items():
            assert source['files']['backend/app/research/'+name]==digest
    with np.load(output/'training-inputs.npz',allow_pickle=False) as saved:
        assert len(saved['study_ids'])==50
        for name,expected in row.config['training_input_hashes'].items():
            assert experiments.array_digest(saved[name])==expected
    assert json.loads((output/'results-seed2026.json').read_text())['study_ids']==run['study_ids']
    with zipfile.ZipFile(io.BytesIO(report_bundle(row))) as archive:
        assert {'summary.csv','subgroup_findings.csv','report.html','provenance.json'}<=set(archive.namelist())
        assert json.loads(archive.read('provenance.json'))['config']['label_repairs']==row.config['label_repairs']
        assert b'synthetic-repair-policy' in archive.read('report.html')
    result=ensemble(output,row.config,np.zeros((6,768),np.float32),np.ones(6,bool),'pooled')
    assert len(result['findings'])==12 and result['folds_per_seed']==5
    assert all(r['review_required'] for r in result['findings'])  # one seed cannot characterize seed variation

    # The post-evaluation fit must consume every source row, keep the comparison
    # immutable, persist usable weights and refuse training-set accuracy claims.
    from app.research.final_fit import run_final_fit,predict_final
    parent_hashes={name:hashlib.sha256((output/name).read_bytes()).hexdigest()
                   for name in ['manifest.json','training-inputs.npz','results-seed2026.json']}
    final_root=Path(run_final_fit(row,device=training_device))
    final_config=json.loads((final_root/'manifest.json').read_text())
    final_status=json.loads((final_root/'status.json').read_text())
    assert final_status['status']=='completed' and final_status['completed_seeds']==[2026]
    assert final_config['training_studies']==50 and final_config['expert_target_studies']==10
    assert final_config['pooled_report_target_studies']==40
    assert final_config['training_device']['resolved']==training_device
    assert final_config['independent_evaluation_available'] is False
    with np.load(final_root/'targets.npz',allow_pickle=False) as final_targets, np.load(output/'training-inputs.npz',allow_pickle=False) as source_inputs:
        np.testing.assert_array_equal(final_targets['study_ids'],source_inputs['study_ids'])
        np.testing.assert_array_equal(final_targets['targets'][final_targets['expert']],source_inputs['gold'][final_targets['expert']])
    predicted=predict_final(final_root,np.zeros((6,768),np.float32),np.ones(6,bool),device=training_device)
    assert len(predicted['findings'])==12 and predicted['training_studies']==50
    assert not predicted['independent_evaluation_available'] and not predicted['clinically_validated']
    assert 'metrics' not in final_status
    for name,sha in parent_hashes.items():assert hashlib.sha256((output/name).read_bytes()).hexdigest()==sha
    with pytest.raises(ValueError,match='already exist'):run_final_fit(row,device=training_device)
    checkpoint=final_root/'pooled-seed2026.pt'
    checkpoint.write_bytes(checkpoint.read_bytes()+b'corruption')
    with pytest.raises(ValueError,match='checkpoint hash mismatch'):
        predict_final(final_root,np.zeros((6,768),np.float32),np.ones(6,bool))
    row.status='running'
    with pytest.raises(ValueError,match='completed'):run_final_fit(row,device=training_device)


def test_incomplete_gold_flag_cannot_silently_enter_image_training(isolated_experiment_db):
    db,root=isolated_experiment_db;version,signature=populate(db,root)
    gold=db.scalar(select(Study).where(Study.is_gold));gold.gold={'ACL':1};db.commit()
    with pytest.raises(ValueError,match='complete binary expert targets'):
        experiments.run_experiment(db,version,signature,seeds=(2026,),epochs=1,device='cpu')
    assert db.scalar(select(Experiment.id)) is None


def test_interruption_is_persisted_with_completed_seed_results(isolated_experiment_db,monkeypatch):
    db,root=isolated_experiment_db;version,signature=populate(db,root)
    def interrupted(*args,**kwargs):
        if kwargs['seed']==2027:raise KeyboardInterrupt('synthetic interruption')
        return dict(metrics={},predictions={},study_ids=[],estimators=[],reserve_estimators=[],omission=[])
    monkeypatch.setattr(experiments,'crossfit',interrupted)
    with pytest.raises(KeyboardInterrupt):experiments.run_experiment(db,version,signature,seeds=(2026,2027),epochs=1,arms=('binary',),device='cpu')
    db.expire_all();row=db.scalar(select(Experiment))
    assert row.status=='interrupted'
    assert row.results['completed_seeds']==1 and row.results['runs'][0]['seed']==2026
    assert (root/'experiments'/row.id/'results-seed2026.json').exists()
