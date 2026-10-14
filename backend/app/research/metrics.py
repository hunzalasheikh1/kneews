import numpy as np
from sklearn.metrics import roc_auc_score
from .constants import FINDINGS


def calibration_bins(y,p,bins=10):
    y,p=np.asarray(y,float),np.asarray(p,float)
    if y.ndim!=1 or y.shape!=p.shape or bins<1 or not np.isin(y,[0,1]).all() or not np.isfinite(p).all() or ((p<0)|(p>1)).any():
        raise ValueError('Calibration requires binary labels and finite probabilities')
    edges=np.linspace(0,1,bins+1)
    groups=np.minimum(np.searchsorted(edges,p,side='right')-1,bins-1)
    return [dict(lower=float(edges[b]),upper=float(edges[b+1]),count=int((groups==b).sum()),
                 mean_probability=float(p[groups==b].mean()) if (groups==b).any() else None,
                 observed_frequency=float(y[groups==b].mean()) if (groups==b).any() else None)
            for b in range(bins)]


def expected_calibration_error(y, p, bins=10):
    edges = np.linspace(0, 1, bins+1)
    groups = np.minimum(np.searchsorted(edges, p, side='right')-1, bins-1)
    return float(sum(np.mean(groups == b)*abs(np.mean(y[groups == b])-np.mean(p[groups == b]))
                     for b in range(bins) if np.any(groups == b)))


def evaluate(gold, predictions,include_calibration=True):
    y, p = np.asarray(gold, float), np.asarray(predictions, float)
    if y.shape != p.shape or y.ndim != 2 or y.shape[1] != 12:
        raise ValueError('Expected matching N x 12 arrays')
    if not np.isfinite(p).all() or ((p<0)|(p>1)).any():
        raise ValueError('Predictions must be finite probabilities')
    if not np.isin(y[np.isfinite(y)], [0,1]).all():
        raise ValueError('Gold labels must be binary')
    per_finding = []
    for i,name in enumerate(FINDINGS):
        mask = np.isfinite(y[:,i]); a,b = y[mask,i],p[mask,i]
        per_finding.append(dict(finding=name, n=len(a), positives=int(a.sum()),
            auc=float(roc_auc_score(a,b)) if len(np.unique(a))==2 else None,
            brier=float(np.mean((a-b)**2)) if len(a) else None,
            ece=expected_calibration_error(a,b) if len(a) else None))
        if include_calibration:per_finding[-1]['calibration_bins']=calibration_bins(a,b)
    aucs = [r['auc'] for r in per_finding if r['auc'] is not None]
    # Never silently present an incomplete macro as a 12-finding result.
    return dict(per_finding=per_finding, macro_auc=float(np.mean(aucs)) if len(aucs)==12 else None,
                available_class_macro_auc=float(np.mean(aucs)) if aucs else None,
                evaluable_findings=len(aucs), worst_finding_auc=min(aucs) if aucs else None,
                brier=float(np.mean((y[np.isfinite(y)]-p[np.isfinite(y)])**2)) if np.isfinite(y).any() else None)


def omission_rates(states, gold, languages,groups=None,iterations=1000,seed=2026):
    states, gold, languages = np.asarray(states), np.asarray(gold,float), np.asarray(languages)
    if states.shape!=gold.shape or gold.ndim!=2 or gold.shape[1]!=12 or languages.shape!=(len(gold),):
        raise ValueError('Omission requires matching N x 12 arrays and N language codes')
    if not np.isin(states,[0,1,2,3]).all() or not np.isin(gold[np.isfinite(gold)],[0,1]).all():
        raise ValueError('Invalid report state or expert label')
    groups=None if groups is None else np.asarray(groups)
    if groups is not None and groups.shape!=(len(gold),):raise ValueError('One group is required per study')
    if iterations<1:raise ValueError('Positive bootstrap iterations required')
    rng=np.random.default_rng(seed)
    output = []
    for language in ['all', *sorted(set(languages.tolist()))]:
        rows = np.ones(len(gold),bool) if language=='all' else languages==language
        unique=np.unique(groups[rows]) if groups is not None else []
        samples=rng.integers(0,len(unique),size=(iterations,len(unique))) if len(unique)>=2 else None
        for i,name in enumerate(FINDINGS):
            positive = rows & (gold[:,i]==1)
            n = int(positive.sum()); omitted = int(((states[:,i]==3)&positive).sum())
            interval=None;valid=0
            if samples is not None:
                denominators=np.array([(positive&(groups==g)).sum() for g in unique])[samples].sum(1)
                numerators=np.array([(positive&(groups==g)&(states[:,i]==3)).sum() for g in unique])[samples].sum(1)
                supported=denominators>0;valid=int(supported.sum())
                if valid:interval=np.quantile(numerators[supported]/denominators[supported],[.025,.975]).tolist()
            output.append(dict(language=language,finding=name,expert_positives=n,unmentioned_positives=omitted,
                               omission_rate=omitted/n if n else None,omission_interval=interval,
                               independent_groups=len(unique) if groups is not None else None,
                               positive_groups=len(np.unique(groups[positive])) if groups is not None else None,
                               interval_method='95% percentile connected-group bootstrap' if groups is not None else 'Unavailable: no group provenance',
                               valid_replicates=valid,requested_replicates=iterations,bootstrap_seed=seed))
    return output


def subgroup_evaluation(gold,predictions,records):
    """Descriptive held-out strata, retaining single-class and small groups."""
    gold=np.asarray(gold,float)
    if len(gold)!=len(records):raise ValueError('Subgroup records must align with held-out outcomes')
    output=[]
    for axis,key in [('language','language'),('scanner_proxy_group','leakage_group')]:
        strata=np.asarray([r.get(key,'unknown') for r in records])
        for stratum in sorted(set(strata.tolist())):
            selected=strata==stratum
            groups={r['leakage_group'] for i,r in enumerate(records) if selected[i]}
            output.append(dict(axis=axis,stratum=stratum,n=int(selected.sum()),independent_groups=len(groups),
                metrics={arm:evaluate(gold[selected],np.asarray(values)[selected],include_calibration=False)
                         for arm,values in predictions.items()},
                uncertainty_interval=None,
                uncertainty_note='Descriptive subgroup estimate; no subgroup confidence interval or significance claim. Small strata and absent classes limit interpretation.',
                hospital_identity_verified=False,language_source='automatic_unreviewed'))
    return output


def paired_group_bootstrap(gold, first, second, groups, iterations=1000, seed=2026, metric='macro_auc'):
    groups=np.asarray(groups); unique=np.unique(groups); rng=np.random.default_rng(seed)
    if len(unique)<2:
        raise ValueError('At least two independent groups required')
    differences=[]
    for _ in range(iterations):
        indices=np.concatenate([np.flatnonzero(groups==g) for g in rng.choice(unique,len(unique),replace=True)])
        a=evaluate(np.asarray(gold)[indices],np.asarray(first)[indices],include_calibration=False)[metric]
        b=evaluate(np.asarray(gold)[indices],np.asarray(second)[indices],include_calibration=False)[metric]
        if a is not None and b is not None: differences.append(b-a)
    return dict(valid_replicates=len(differences),requested_replicates=iterations,
                delta_interval=np.quantile(differences,[.025,.975]).tolist() if differences else None,
                metric=metric,seed=seed)
