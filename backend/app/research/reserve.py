"""Gold-free confident-joint estimator from demonstrably out-of-fold predictions.

This binary reserve follows the confident-learning counting principle. It is a
separate method from four-state gold-informed pooling and reports its assumptions.
"""
import numpy as np


def confident_joint(binary_labels,oof_probabilities,study_ids,folds,prediction_training_ids):
    y=np.asarray(binary_labels,int); p=np.asarray(oof_probabilities,float)
    if y.shape!=p.shape or y.ndim!=2 or y.shape[1]!=12: raise ValueError('Expected N x 12 arrays')
    if not np.isin(y,[0,1]).all() or not np.isfinite(p).all() or ((p<0)|(p>1)).any():
        raise ValueError('Invalid binary labels or probabilities')
    if len(study_ids)!=len(y) or len(folds)!=len(y) or len(set(study_ids))!=len(y): raise ValueError('Unique prediction IDs and complete fold provenance required')
    for fold in set(folds):
        if fold not in prediction_training_ids:raise ValueError('Missing OOF training provenance')
        training=set(prediction_training_ids[fold])
        held={uid for uid,f in zip(study_ids,folds) if f==fold}
        if not training or training & held:raise ValueError('In-sample fold predictions cannot fit the reserve estimator')
    for uid,fold in zip(study_ids,folds):
        if fold not in prediction_training_ids: raise ValueError('Missing OOF training provenance')
        if uid in prediction_training_ids[fold]: raise ValueError('In-sample predictions cannot fit the reserve estimator')
    joint=np.zeros((12,2,2)); thresholds=np.zeros((12,2))
    for k in range(12):
        probabilities=np.column_stack([1-p[:,k],p[:,k]])
        for observed in (0,1):
            mask=y[:,k]==observed
            if not mask.any(): raise ValueError(f'Finding {k} lacks observed class {observed}')
            thresholds[k,observed]=probabilities[mask,observed].mean()
        confident=probabilities>=thresholds[k]
        assignment=np.where(confident.sum(1)==1,confident.argmax(1),probabilities.argmax(1))
        usable=confident.any(1)
        for observed in (0,1):
            for latent in (0,1):
                joint[k,observed,latent]=np.sum(usable&(y[:,k]==observed)&(assignment==latent))
            # Recalibrate row counts to observed label frequency, with Jeffreys smoothing.
            joint[k,observed]=(joint[k,observed]+.5)/(joint[k,observed].sum()+1)*(y[:,k]==observed).sum()
    reliability=joint[:,:,1]/joint.sum(2)
    soft=np.stack([reliability[k,y[:,k]] for k in range(12)],axis=1)
    return dict(joint=joint.tolist(),thresholds=thresholds.tolist(),reliability=reliability.tolist(),targets=soft,
                method='binary_confident_joint',uses_expert_labels=False)


def reserve_targets(features,masks,states,records,epochs,seed,device='cpu',teacher_cache=None):
    """Nested grouped OOF targets, called only with an outer training partition."""
    from .training import fit_head,predict
    from .splits import assert_no_leakage
    assert_no_leakage(records)
    folds=np.asarray([r['fold'] for r in records]);ids=[r['id'] for r in records]
    if len(set(folds))<2:raise ValueError('Reserve teacher needs at least two independent inner folds')
    y=(np.asarray(states)==0).astype(np.float32)
    predictions=np.full(y.shape,np.nan);provenance={}
    cache=teacher_cache if teacher_cache is not None else {}
    for fold in sorted(set(folds.tolist())):
        train=folds!=fold;held=~train
        train_ids=[uid for i,uid in enumerate(ids) if train[i]]
        key=(seed,epochs,tuple(train_ids))
        if key not in cache:
            model,_=fit_head(features[train],masks[train],y[train],np.ones_like(y[train]),epochs=epochs,seed=seed,device=device)
            cache[key]=model
        predictions[held]=predict(cache[key],features[held],masks[held],device)
        provenance[int(fold)]=train_ids
    result=confident_joint(y,predictions,ids,folds.tolist(),provenance)
    result['teacher_training_ids']=provenance
    result['teacher_prediction_ids']=ids
    result['teacher_folds']=folds.tolist()
    return result
