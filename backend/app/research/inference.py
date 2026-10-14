"""External-study ensemble inference; separate from held-out evaluation results."""
import hashlib
import numpy as np
import torch
from ..config import settings,ROOT
from ..models import Study,Experiment
from .constants import FINDINGS
from .features import FrozenEncoder,PLANES
from .training import SlotHead,predict
from .abstention import summarize_seed_predictions


def validate_external(study):
    if study is None or study.split!='external' or study.source!='upload':
        raise ValueError('New-study inference is restricted to external uploads')
    if not study.series:raise ValueError('Study has no image series')
    for series in study.series:
        meta=series.metadata_json
        if series.plane not in PLANES or not meta.get('protocol_reviewed') or meta.get('fluid_sensitive') not in (0,1) or meta.get('laterality') not in ('L','R'):
            raise ValueError('Confirm acquisition plane, fluid sensitivity and knee laterality for every uploaded series')
    if len({s.metadata_json['laterality'] for s in study.series})!=1:
        raise ValueError('All series must describe the same knee laterality')


def ensemble(checkpoint_root,config,features,mask,arm):
    if features.ndim!=2 or features.shape[0]!=6 or mask.shape!=(6,) or not mask.any() or not np.isfinite(features).all():
        raise ValueError('Expected finite six-slot features with at least one image slot')
    if arm not in config['arms']:raise ValueError('Arm not present in experiment')
    folds=sorted({r['fold'] for r in config['folds'] if r['is_gold']})
    if not folds or not config['seeds']:raise ValueError('Experiment has no completed fold/seed configuration')
    predictions=[];provenance=[]
    for seed in config['seeds']:
        fold_predictions=[]
        for fold in folds:
            path=checkpoint_root/f'{arm}-seed{seed}-fold{fold}.pt'
            model=SlotHead(features.shape[-1])
            model.load_state_dict(torch.load(path,map_location='cpu',weights_only=True))
            fold_predictions.append(predict(model,features[None],mask[None])[0])
            with path.open('rb') as stream:sha=hashlib.file_digest(stream,'sha256').hexdigest()
            provenance.append(dict(seed=seed,fold=fold,checkpoint_sha256=sha))
        predictions.append(np.mean(fold_predictions,axis=0))
    result=summarize_seed_predictions(predictions,int(mask.reshape(3,2).any(1).sum()))
    for finding,row in zip(FINDINGS,result['findings']):row['finding']=finding
    return dict(**result,checkpoints=provenance,seeds=len(predictions),folds_per_seed=len(folds),
                prediction_source='external_fold_ensemble',aggregation='Mean across outer-fold heads within each seed, then mean across seeds')


def infer_study(db,study_id,experiment_id,arm='pooled',progress=None):
    study=db.get(Study,study_id);validate_external(study)
    experiment=db.get(Experiment,experiment_id)
    if experiment is None or experiment.status!='completed':raise ValueError('A completed experiment is required')
    if arm not in experiment.config['arms']:raise ValueError('Arm not present in experiment')
    if progress:progress(.05,'Loading frozen image encoder')
    encoder=FrozenEncoder(ROOT/'models/dinov2-small')
    if encoder.signature!=experiment.config['feature_signature']:
        raise ValueError('Installed encoder/preprocessing does not match the completed experiment')
    features,mask,metadata=encoder.study(study.series,settings.data_root)
    del encoder
    if progress:progress(.7,'Applying saved experiment heads')
    result=ensemble(settings.artifact_root/'experiments'/experiment.id,experiment.config,features,mask,arm)
    if metadata['errors']:
        for finding in result['findings']:
            finding['review_required']=True;finding['reasons'].append('One or more image series could not be processed')
    return dict(**result,study_id=study_id,experiment_id=experiment.id,manifest_hash=experiment.manifest_hash,
                arm=arm,feature_signature=experiment.config['feature_signature'],feature_provenance=metadata,
                protocol=[dict(series_id=s.id,plane=s.plane,metadata=s.metadata_json) for s in study.series],
                clinically_validated=False)


def final_model_root():
    """Most recently completed final all-study model directory."""
    candidates=[]
    for status_path in (settings.artifact_root/'final_models').glob('*/status.json'):
        import json
        status=json.loads(status_path.read_text())
        if status.get('status')=='completed':candidates.append((status.get('finished_at') or status.get('started_at') or 0,status_path.parent))
    if not candidates:raise ValueError('No completed final model is available')
    return max(candidates)[1]


def reference_performance(db,parent_experiment_id,arm='pooled'):
    """Cross-validated per-finding AUC/Brier of the same method on the 58 expert studies (mean over seeds)."""
    experiment=db.get(Experiment,parent_experiment_id)
    runs=[r for r in (experiment.results or {}).get('runs',[]) if arm in r.get('metrics',{})] if experiment else []
    if not runs:return None
    rows=[]
    for i,finding in enumerate(FINDINGS):
        aucs=[r['metrics'][arm]['per_finding'][i]['auc'] for r in runs if r['metrics'][arm]['per_finding'][i]['auc'] is not None]
        briers=[r['metrics'][arm]['per_finding'][i]['brier'] for r in runs]
        rows.append(dict(finding=finding,auc=float(np.mean(aucs)) if aucs else None,brier=float(np.mean(briers)),
                         expert_positives=runs[0]['metrics'][arm]['per_finding'][i]['positives'],expert_n=runs[0]['metrics'][arm]['per_finding'][i]['n']))
    return dict(arm=arm,seeds=len(runs),per_finding=rows,
                macro_auc=float(np.mean([r['metrics'][arm]['macro_auc'] for r in runs])),
                source='Five-fold site-grouped cross-fitting on 58 expert studies; the final model itself has no independent test')


def validate_final_target(study):
    if study is None:raise ValueError('Study not found')
    if study.split=='train':raise ValueError('Training studies were used to fit the final model; predictions on them are in-sample')
    if study.split=='external':validate_external(study)
    elif study.split!='test':raise ValueError('Final-model prediction is limited to uploads and held-out example test studies')
    if not study.series:raise ValueError('Study has no image series')


def infer_final_study(db,study_id,progress=None):
    import json
    from .final_fit import predict_final
    study=db.get(Study,study_id);validate_final_target(study)
    root=final_model_root();config=json.loads((root/'manifest.json').read_text())
    if progress:progress(.05,'Loading frozen image encoder')
    encoder=FrozenEncoder(ROOT/'models/dinov2-small')
    if encoder.signature!=config['feature_signature']:
        raise ValueError('Installed encoder/preprocessing does not match the final model')
    if progress:progress(.15,'Encoding MRI series')
    features,mask,metadata=encoder.study(study.series,settings.data_root)
    del encoder
    if progress:progress(.8,'Applying final model ensemble')
    result=predict_final(root,features,mask)
    if metadata['errors']:
        for finding in result['findings']:
            finding['review_required']=True;finding['reasons'].append('One or more image series could not be processed')
    return dict(**result,study_id=study_id,model='final',final_model_id=root.name,feature_signature=config['feature_signature'],
                feature_provenance=metadata,reference_performance=reference_performance(db,config['parent_experiment_id']),
                protocol=[dict(series_id=s.id,plane=s.plane,metadata=s.metadata_json) for s in study.series])
