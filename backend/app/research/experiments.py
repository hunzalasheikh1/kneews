import hashlib
import io
import json
from pathlib import Path
import zipfile
import numpy as np
import torch
from sqlalchemy import select
from ..config import settings,ROOT
from ..models import Study,LabelSet,Experiment
from .constants import FINDINGS,STATE_INDEX
from .splits import make_folds
from .training import crossfit,ARMS
from .metrics import paired_group_bootstrap
from .cache import read_cache
from .readiness import validate_label
from .devices import resolve_device,device_provenance


def array_digest(array):
    array=np.ascontiguousarray(array)
    digest=hashlib.sha256(json.dumps(dict(shape=array.shape,dtype=array.dtype.str),sort_keys=True).encode())
    digest.update(array.tobytes());return digest.hexdigest()


def source_snapshot():
    """Capture a restricted source allowlist; never include .env or patient data."""
    paths=set((ROOT/'backend/app').rglob('*.py'))
    paths.update([ROOT/'backend/requirements.lock.txt',ROOT/'docs/research/PREREGISTRATION.json',ROOT/'scripts/dev.py'])
    paths.update((ROOT/'docs/research').glob('AMENDMENT-*.json'))
    files={str(path.relative_to(ROOT)):path.read_bytes() for path in sorted(paths)}
    buffer=io.BytesIO()
    with zipfile.ZipFile(buffer,'w',compression=zipfile.ZIP_DEFLATED) as archive:
        for name,content in files.items():
            entry=zipfile.ZipInfo(name,date_time=(1980,1,1,0,0,0))
            entry.compress_type=zipfile.ZIP_DEFLATED
            archive.writestr(entry,content)
    return buffer.getvalue(),{name:hashlib.sha256(content).hexdigest() for name,content in files.items()}


def run_experiment(db,label_version,feature_signature,seeds=(2026,2027,2028),epochs=30,arms=ARMS,device='auto'):
    requested_device=device
    device=resolve_device(requested_device)
    if not arms or len(set(arms))!=len(arms) or any(a not in (*ARMS,'reserve') for a in arms):
        raise ValueError('Experiment arms must be unique supported arms')
    if not seeds or len(set(seeds))!=len(seeds) or epochs<1:
        raise ValueError('Experiment requires unique seeds and positive epochs')
    studies=db.scalars(select(Study).where(Study.split=='train').order_by(Study.id)).all()
    if not studies:raise ValueError('No training studies imported')
    labels={r.study_id:r for r in db.scalars(select(LabelSet).where(LabelSet.method=='local_llm',LabelSet.version==label_version))}
    if any(s.id not in labels for s in studies):
        raise ValueError('All training studies need the same validated local label version before a controlled comparison')
    root=settings.artifact_root/'features'/feature_signature
    features=[];masks=[];states=[];probabilities=[];gold=[];records=[]
    for s in studies:
        if s.is_gold:
            if set(s.gold)!=set(FINDINGS) or any(value not in (0,1) for value in s.gold.values()):
                raise ValueError('Expert cohort flags require complete binary expert targets')
        elif s.gold:
            raise ValueError('Report-only studies must not carry expert targets')
        path=root/(s.id+'.npz')
        if not path.exists(): raise ValueError(f'Missing frozen feature cache for {s.id}')
        cached,mask,_=read_cache(path,s.id,feature_signature)
        features.append(cached);masks.append(mask)
        validate_label(labels[s.id],s)
        source=labels[s.id].findings
        states.append([STATE_INDEX[source[k]['state']] for k in FINDINGS])
        probabilities.append([source[k]['probability'] if source[k]['probability'] is not None else np.nan for k in FINDINGS])
        gold.append([s.gold.get(k,np.nan) for k in FINDINGS])
        records.append(dict(id=s.id,site_group=s.site_group,report_hash=s.report_hash,language=s.language,is_gold=s.is_gold))
    records=make_folds(records)
    features=np.asarray(features);masks=np.asarray(masks);states=np.asarray(states);gold=np.asarray(gold,float);probabilities=np.asarray(probabilities,float)
    inputs=dict(features=features,masks=masks,states=states,gold=gold,llm_probabilities=probabilities)
    config=dict(label_version=label_version,feature_signature=feature_signature,seeds=list(seeds),epochs=epochs,arms=list(arms),
                preregistration=json.loads((ROOT/'docs/research/PREREGISTRATION.json').read_text()),folds=records,
                dataset_count=len(studies),hospital_identity_verified=False)
    config['label_repairs']=[dict(study_id=s.id,**labels[s.id].provenance['evidence_repair'])
        for s in studies if labels[s.id].provenance.get('evidence_repair')]
    config['training_device']=device_provenance(requested_device,device)
    config['training_input_hashes']={name:array_digest(values) for name,values in inputs.items()}
    config['exploratory_arms']=[a for a in arms if a not in ARMS]
    preregistration=config['preregistration']
    config['protocol_deviations']={key:dict(registered=preregistration[key],actual=config[key])
        for key in ['seeds','epochs'] if config[key]!=preregistration[key]}
    if [a for a in arms if a in ARMS]!=preregistration['arms']:
        config['protocol_deviations']['primary_arms']=dict(registered=preregistration['arms'],actual=list(arms))
    config['registered_primary_protocol_match']=not config['protocol_deviations']
    config['pre_outcome_amendments']=[json.loads(path.read_text())
        for path in sorted((ROOT/'docs/research').glob('AMENDMENT-*.json'))]
    snapshot,source_hashes=source_snapshot()
    config['source_snapshot']=dict(filename='source-snapshot.zip',sha256=hashlib.sha256(snapshot).hexdigest(),files=source_hashes)
    config['environment_sha256']=source_hashes['backend/requirements.lock.txt']
    config['research_source_sha256']={Path(name).name:digest for name,digest in source_hashes.items() if name.startswith('backend/app/research/')}
    manifest_hash=hashlib.sha256(json.dumps(config,sort_keys=True).encode()).hexdigest()
    name='Four-state supervision comparison'+(' + gold-free reserve' if 'reserve' in arms else '')
    experiment=Experiment(name=name,status='running',config=config,manifest_hash=manifest_hash)
    db.add(experiment);db.commit()
    experiment_id=experiment.id
    output=settings.artifact_root/'experiments'/experiment_id
    results=[]
    try:
        output.mkdir(parents=True)
        (output/'source-snapshot.zip').write_bytes(snapshot)
        (output/'manifest.json').write_text(json.dumps(config,indent=2))
        np.savez_compressed(output/'training-inputs.npz',**inputs,study_ids=np.asarray([s.id for s in studies]))
        for seed in seeds:
            def save_fold(arm,fold,model,history,predictions,ids):
                actual_device=next(model.parameters()).device
                if actual_device.type!=device.type:
                    raise RuntimeError(f'Expected training device {device}, got {actual_device}')
                torch.save(model.state_dict(),output/f'{arm}-seed{seed}-fold{fold}.pt')
                (output/f'{arm}-seed{seed}-fold{fold}.json').write_text(json.dumps(dict(history=history,predictions=predictions.tolist(),study_ids=ids,training_device=str(actual_device))))
                print(json.dumps(dict(experiment=experiment.id,seed=seed,arm=arm,fold=fold,status='completed',device=str(actual_device))),flush=True)
                experiment.results={'progress':dict(seed=seed,arm=arm,fold=fold,completed_seeds=len(results),device=str(actual_device)),'runs':list(results)}
                db.commit()
            result=crossfit(features,masks,states,gold,records,llm=probabilities,epochs=epochs,arms=arms,seed=seed,device=device,on_fold=save_fold)
            result['seed']=seed
            if 'binary' in arms and 'pooled' in arms:
                selected=np.isfinite(gold).all(1)
                groups=[r['leakage_group'] for i,r in enumerate(records) if selected[i]]
                result['paired_brier']=paired_group_bootstrap(gold[selected],result['predictions']['binary'],result['predictions']['pooled'],groups,metric='brier')
                result['paired_auc']=paired_group_bootstrap(gold[selected],result['predictions']['binary'],result['predictions']['pooled'],groups)
            if 'binary' in arms and 'reserve' in arms:
                selected=np.isfinite(gold).all(1)
                groups=[r['leakage_group'] for i,r in enumerate(records) if selected[i]]
                result['reserve_vs_binary']=dict(exploratory=True,
                    brier=paired_group_bootstrap(gold[selected],result['predictions']['binary'],result['predictions']['reserve'],groups,metric='brier'),
                    auc=paired_group_bootstrap(gold[selected],result['predictions']['binary'],result['predictions']['reserve'],groups))
            results.append(result)
            experiment.results={'runs':list(results),'progress':{'completed_seeds':len(results)}};db.commit()
            (output/f'results-seed{seed}.json').write_text(json.dumps(result,indent=2))
        plane_coverage=masks.reshape(len(masks),3,2).any(2).sum(1)
        experiment.status='completed';experiment.results=dict(runs=results,manifest_hash=manifest_hash,artifact_path=str(output),
            plane_coverage={s.id:int(plane_coverage[i]) for i,s in enumerate(studies) if s.is_gold});db.commit()
    except BaseException as exc:
        db.rollback();experiment=db.get(Experiment,experiment_id)
        experiment.status='interrupted' if isinstance(exc,(KeyboardInterrupt,SystemExit)) else 'failed'
        experiment.results={**experiment.results,'error':str(exc),'runs':list(results),'completed_seeds':len(results)}
        db.commit();raise
    return experiment.id
