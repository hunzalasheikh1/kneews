"""Matched heads over fixed features. All 4 arms share splits and initialization."""
import random
import numpy as np
import torch
from torch import nn
from .reliability import PooledReliability
from .metrics import evaluate,omission_rates,subgroup_evaluation

ARMS=('binary','masked','pooled','llm_probability')
class SlotHead(nn.Module):
    def __init__(self,dim,hidden=128):
        super().__init__()
        self.slot=nn.Parameter(torch.zeros(6,hidden))
        self.project=nn.Sequential(nn.LayerNorm(dim),nn.Linear(dim,hidden),nn.GELU())
        self.attention=nn.Linear(hidden,1)
        self.classifier=nn.Sequential(nn.LayerNorm(hidden),nn.Linear(hidden,12))
    def forward(self,x,mask):
        h=self.project(x)+self.slot[None]
        logits=self.attention(h).squeeze(-1).masked_fill(~mask,-1e9)
        pooled=(h*torch.softmax(logits,dim=1)[...,None]).sum(1)
        return self.classifier(pooled)


def targets(arm,states,reliability=None,llm_probabilities=None):
    states=np.asarray(states); weights=np.ones(states.shape,np.float32)
    if arm=='binary': target=(states==0).astype(np.float32)
    elif arm=='masked':
        target=(states==0).astype(np.float32); weights=(states<2).astype(np.float32)
    elif arm=='pooled':
        if reliability is None: raise ValueError('Pooled arm requires a training-only estimator')
        target=reliability.transform(states).astype(np.float32)
    elif arm=='llm_probability':
        if llm_probabilities is None: raise ValueError('LLM probability arm requires real local LLM output')
        target=np.asarray(llm_probabilities,np.float32)
        if target.shape!=states.shape: raise ValueError('Probability shape mismatch')
        weights=np.isfinite(target).astype(np.float32)
        if not weights.any(): raise ValueError('No LLM probabilities available')
        target=np.nan_to_num(target,nan=0)
        if ((target<0)|(target>1)).any(): raise ValueError('Invalid LLM probability')
    else: raise ValueError('Unknown arm')
    return target,weights


def fit_head(x,mask,y,weights,epochs=30,seed=2026,batch_size=32,lr=1e-3,device='cpu'):
    if x.ndim!=3 or x.shape[1]!=6 or mask.shape!=x.shape[:2] or y.shape!=(len(x),12) or weights.shape!=y.shape:
        raise ValueError('Invalid feature/target shapes')
    if not mask.any(1).all() or not all(np.isfinite(a).all() for a in (x,y,weights)):
        raise ValueError('Missing image slots or non-finite training values')
    if (weights<0).any() or weights.sum()<=0 or ((y<0)|(y>1)).any() or epochs<1:
        raise ValueError('Training requires supported probability targets and positive weight')
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
    model=SlotHead(x.shape[-1]).to(device)
    optimizer=torch.optim.AdamW(model.parameters(),lr=lr,weight_decay=1e-3)
    tensors=[torch.as_tensor(a) for a in (x,mask,y,weights)]
    generator=torch.Generator().manual_seed(seed)
    loader=torch.utils.data.DataLoader(torch.utils.data.TensorDataset(*tensors),batch_size=batch_size,shuffle=True,generator=generator)
    history=[]
    for epoch in range(epochs):
        model.train(); losses=[]
        for features,available,target,weight in loader:
            features,available,target,weight=[a.to(device) for a in (features,available,target,weight)]
            if weight.sum()==0: continue
            loss=(nn.functional.binary_cross_entropy_with_logits(model(features,available),target,reduction='none')*weight).sum()/weight.sum()
            optimizer.zero_grad(); loss.backward(); optimizer.step(); losses.append(float(loss.detach().cpu()))
        history.append(float(np.mean(losses)) if losses else None)
    return model,history


@torch.inference_mode()
def predict(model,x,mask,device='cpu'):
    model.eval(); outputs=[]
    for start in range(0,len(x),64):
        outputs.append(model(torch.as_tensor(x[start:start+64]).to(device),torch.as_tensor(mask[start:start+64]).to(device)).sigmoid().cpu().numpy())
    return np.concatenate(outputs)


def crossfit(features,masks,states,gold,records,llm=None,epochs=30,arms=ARMS,seed=2026,device='cpu',on_fold=None):
    from .splits import assert_no_leakage
    assert_no_leakage(records)
    folds=np.asarray([r['fold'] for r in records]); gold=np.asarray(gold,float)
    is_gold=np.isfinite(gold).all(1)
    if not arms or any(a not in (*ARMS,'reserve') for a in arms):raise ValueError('Unsupported experiment arms')
    output={arm:np.full(gold.shape,np.nan) for arm in arms}; estimators=[];reserve_estimators=[];teacher_cache={}
    for fold in sorted(set(folds.tolist())):
        held=(folds==fold)&is_gold
        if not held.any(): continue
        estimate=(folds!=fold)&is_gold
        train=(folds!=fold)&~is_gold
        if not train.any() or ('pooled' in arms and not estimate.any()): raise ValueError('Insufficient independent training/estimator cases')
        reliability=None
        if 'pooled' in arms:
            reliability=PooledReliability().fit(states[estimate],gold[estimate],
                         [r['id'] for i,r in enumerate(records) if estimate[i]],
                         [r['id'] for i,r in enumerate(records) if held[i]])
            estimators.append(dict(fold=fold,table=reliability.table(),fit_ids=reliability.fit_ids_))
        for arm in arms:
            if arm=='reserve':
                from .reserve import reserve_targets
                result=reserve_targets(features[train],masks[train],states[train],
                    [r for i,r in enumerate(records) if train[i]],epochs,seed,device,teacher_cache)
                y=result.pop('targets').astype(np.float32);w=np.ones_like(y)
                reserve_estimators.append(dict(fold=fold,**result))
            else:y,w=targets(arm,states[train],reliability,llm[train] if llm is not None else None)
            model,history=fit_head(features[train],masks[train],y,w,epochs=epochs,seed=seed,device=device)
            output[arm][held]=predict(model,features[held],masks[held],device)
            if on_fold: on_fold(arm,fold,model,history,output[arm][held],[r['id'] for i,r in enumerate(records) if held[i]])
    if any(not np.isfinite(p[is_gold]).all() for p in output.values()): raise ValueError('Incomplete outer-fold predictions')
    return dict(metrics={arm:evaluate(gold[is_gold],p[is_gold]) for arm,p in output.items()},
                subgroups=subgroup_evaluation(gold[is_gold],{arm:p[is_gold] for arm,p in output.items()},
                                             [r for i,r in enumerate(records) if is_gold[i]]),
                predictions={arm:p[is_gold].tolist() for arm,p in output.items()},
                study_ids=[r['id'] for i,r in enumerate(records) if is_gold[i]],estimators=estimators,reserve_estimators=reserve_estimators,
                omission=omission_rates(states[is_gold],gold[is_gold],[r.get('language','unknown') for i,r in enumerate(records) if is_gold[i]],
                    groups=[r['leakage_group'] for i,r in enumerate(records) if is_gold[i]]))
