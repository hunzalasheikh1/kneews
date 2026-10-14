"""Fit the proposed pooled model on all studies after completing evaluation.

Artifacts are separate from Experiment results. No in-sample score is reported.
"""
import hashlib
import io
import json
import time
import zipfile
from pathlib import Path

import numpy as np
import torch

from ..config import ROOT,settings
from .constants import FINDINGS
from .devices import resolve_device,device_provenance
from .experiments import array_digest,source_snapshot
from .reliability import PooledReliability
from .reporting import completed_runs
from .training import SlotHead,fit_head,predict
from .abstention import summarize_seed_predictions


def digest_json(value):
    return hashlib.sha256(json.dumps(value,sort_keys=True).encode()).hexdigest()


def write_json(path,value):
    temporary=path.with_suffix('.tmp')
    temporary.write_text(json.dumps(value,indent=2))
    temporary.replace(path)


def load_parent_inputs(parent,root):
    completed_runs(parent)
    if 'pooled' not in parent.config['arms']:
        raise ValueError('The completed comparison must include the proposed pooled method')
    manifest=json.loads((root/'manifest.json').read_text())
    if digest_json(manifest)!=parent.manifest_hash or manifest!=parent.config:
        raise ValueError('Parent comparison manifest differs from its recorded hash/configuration')
    names=('features','masks','states','gold','llm_probabilities')
    with np.load(root/'training-inputs.npz',allow_pickle=False) as saved:
        arrays={name:saved[name] for name in names}
        ids=saved['study_ids'].tolist()
    for name,array in arrays.items():
        if array_digest(array)!=manifest['training_input_hashes'][name]:
            raise ValueError(f'Parent input hash mismatch: {name}')
    if ids!=[r['id'] for r in manifest['folds']] or len(ids)!=manifest['dataset_count'] or len(set(ids))!=len(ids):
        raise ValueError('Parent input snapshot does not cover the exact study cohort')
    x,mask,states,gold=[arrays[n] for n in ('features','masks','states','gold')]
    if x.ndim!=3 or x.shape[:2]!=(len(ids),6) or x.dtype!=np.float32 or not np.isfinite(x).all():
        raise ValueError('Invalid full-cohort feature snapshot')
    if mask.shape!=x.shape[:2] or mask.dtype!=np.bool_ or not mask.any(1).all():
        raise ValueError('Invalid full-cohort feature mask')
    if states.shape!=(len(ids),12) or not np.isin(states,range(4)).all():
        raise ValueError('Invalid full-cohort states')
    if gold.shape!=states.shape:
        raise ValueError('Invalid full-cohort expert targets')
    expert=np.asarray([r['is_gold'] for r in manifest['folds']],bool)
    if not expert.any() or not np.isin(gold[expert],[0,1]).all() or not np.isnan(gold[~expert]).all():
        raise ValueError('Expert flags and source targets disagree')
    return arrays,ids,expert


def all_study_targets(states,gold,ids,expert,concentration):
    estimator=PooledReliability(concentration=concentration).fit(
        states[expert],gold[expert],[uid for uid,flag in zip(ids,expert) if flag])
    targets=estimator.transform(states).astype(np.float32)
    targets[expert]=gold[expert].astype(np.float32)
    return targets,estimator.table()


def run_final_fit(parent,device='mps'):
    actual=resolve_device(device)
    parent_root=settings.artifact_root/'experiments'/parent.id
    arrays,ids,expert=load_parent_inputs(parent,parent_root)
    protocol_path=ROOT/'docs/research/FINAL_MODEL_PROTOCOL.json'
    protocol_bytes=protocol_path.read_bytes();protocol=json.loads(protocol_bytes)
    output=settings.artifact_root/'final_models'/parent.id
    if output.exists():
        raise ValueError(f'Final-model artifacts already exist at {output}; preserve and inspect them before another fit')
    targets,reliability=all_study_targets(arrays['states'],arrays['gold'],ids,expert,
        parent.config['preregistration']['pooling_concentration'])
    source,hashes=source_snapshot()
    buffer=io.BytesIO(source)
    with zipfile.ZipFile(buffer,'a',compression=zipfile.ZIP_DEFLATED) as archive:
        archive.writestr('docs/research/FINAL_MODEL_PROTOCOL.json',protocol_bytes)
    source=buffer.getvalue()
    hashes['docs/research/FINAL_MODEL_PROTOCOL.json']=hashlib.sha256(protocol_bytes).hexdigest()
    config=dict(kind='final_all_studies',protocol=protocol,parent_experiment_id=parent.id,
                parent_manifest_hash=parent.manifest_hash,parent_training_input_hashes=parent.config['training_input_hashes'],
                label_version=parent.config['label_version'],label_repairs=parent.config.get('label_repairs',[]),
                feature_signature=parent.config['feature_signature'],feature_dimension=arrays['features'].shape[2],
                study_ids=ids,training_studies=len(ids),expert_target_studies=int(expert.sum()),
                pooled_report_target_studies=int((~expert).sum()),arm='pooled',
                epochs=parent.config['epochs'],seeds=parent.config['seeds'],batch_size=32,
                optimizer='AdamW',learning_rate=.001,weight_decay=.001,
                target_sha256=array_digest(targets),training_device=device_provenance(device,actual),
                independent_evaluation_available=False,
                source_snapshot=dict(filename='source-snapshot.zip',sha256=hashlib.sha256(source).hexdigest(),files=hashes))
    manifest_hash=digest_json(config)
    output.mkdir(parents=True)
    write_json(output/'manifest.json',config)
    (output/'source-snapshot.zip').write_bytes(source)
    np.savez_compressed(output/'targets.npz',targets=targets,expert=expert,study_ids=np.asarray(ids))
    write_json(output/'reliability.json',reliability)
    status=dict(status='training',parent_experiment_id=parent.id,manifest_hash=manifest_hash,
                training_studies=len(ids),seeds=config['seeds'],completed_seeds=[],checkpoints={},started_at=time.time())
    def publish():
        status['updated_at']=time.time();write_json(output/'status.json',status)
        print(json.dumps({k:status[k] for k in ['status','training_studies','completed_seeds','updated_at']}),flush=True)
    publish()
    try:
        for seed in config['seeds']:
            model,history=fit_head(arrays['features'],arrays['masks'],targets,np.ones_like(targets),
                                   epochs=config['epochs'],seed=seed,batch_size=32,lr=.001,device=actual)
            parameter_device=next(model.parameters()).device
            if parameter_device.type!=actual.type or not np.isfinite(history).all():
                raise RuntimeError('Final-model device or loss validation failed')
            if not all(torch.isfinite(p).all().item() for p in model.parameters()):
                raise RuntimeError('Final-model parameters are nonfinite')
            path=output/f'pooled-seed{seed}.pt';temporary=path.with_suffix('.tmp')
            torch.save(model.state_dict(),temporary);temporary.replace(path)
            sha=hashlib.sha256(path.read_bytes()).hexdigest()
            write_json(output/f'pooled-seed{seed}.json',dict(seed=seed,history=history,
                training_studies=len(ids),training_device=str(parameter_device),checkpoint_sha256=sha))
            status['completed_seeds'].append(seed);status['checkpoints'][str(seed)]=dict(filename=path.name,sha256=sha)
            del model
            publish()
        status['status']='completed';status['finished_at']=time.time();publish()
    except BaseException as exc:
        status['status']='interrupted' if isinstance(exc,(KeyboardInterrupt,SystemExit)) else 'failed'
        status['error']=str(exc) or type(exc).__name__;publish();raise
    return str(output)


def predict_final(root,features,mask,device='cpu'):
    root=Path(root);config=json.loads((root/'manifest.json').read_text());status=json.loads((root/'status.json').read_text())
    if config.get('kind')!='final_all_studies' or status['status']!='completed' or status['completed_seeds']!=config['seeds']:
        raise ValueError('Final inference requires every completed final-model seed')
    if digest_json(config)!=status['manifest_hash']:
        raise ValueError('Final-model manifest hash mismatch')
    features=np.asarray(features,np.float32);mask=np.asarray(mask)
    if features.shape!=(6,config['feature_dimension']) or mask.shape!=(6,) or mask.dtype!=np.bool_ or not mask.any() or not np.isfinite(features).all():
        raise ValueError('Invalid final-model feature input')
    actual=resolve_device(device);predictions=[]
    for seed in config['seeds']:
        checkpoint=status['checkpoints'][str(seed)]
        filename=f'pooled-seed{seed}.pt'
        if checkpoint['filename']!=filename:
            raise ValueError('Unexpected checkpoint filename')
        path=root/filename
        if hashlib.sha256(path.read_bytes()).hexdigest()!=checkpoint['sha256']:
            raise ValueError('Final checkpoint hash mismatch')
        model=SlotHead(config['feature_dimension'])
        model.load_state_dict(torch.load(path,map_location='cpu',weights_only=True));model.to(actual)
        predictions.append(predict(model,features[None],mask[None],device=actual)[0])
    result=summarize_seed_predictions(predictions,int(mask.reshape(3,2).any(1).sum()))
    for finding,row in zip(FINDINGS,result['findings']):row['finding']=finding
    return dict(**result,prediction_source='final_all_studies_seed_ensemble',parent_experiment_id=config['parent_experiment_id'],
                final_manifest_hash=status['manifest_hash'],training_studies=config['training_studies'],
                independent_evaluation_available=False,clinically_validated=False)
