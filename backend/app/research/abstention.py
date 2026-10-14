"""A prespecified research review policy, not a clinical decision threshold."""
import numpy as np

POLICY=dict(version='research-review-0.1',lower=0.2,upper=0.8,max_seed_range=0.2,
            minimum_anatomical_planes=3,clinically_validated=False)


def summarize_seed_predictions(predictions,available_planes=3):
    values=np.asarray(predictions,float)
    if values.ndim!=2 or values.shape[1]!=12 or not np.isfinite(values).all() or ((values<0)|(values>1)).any():
        raise ValueError('Expected finite seed-by-12 probabilities')
    mean=values.mean(0);low=values.min(0);high=values.max(0)
    result=[]
    for i,p in enumerate(mean):
        reasons=[]
        if len(values)<2:reasons.append('Seed variability unavailable')
        if available_planes<POLICY['minimum_anatomical_planes']:reasons.append('Incomplete anatomical plane coverage')
        if POLICY['lower']<p<POLICY['upper']:reasons.append('Probability inside the review band')
        if high[i]-low[i]>POLICY['max_seed_range']:reasons.append('Predictions differ across seeds')
        result.append(dict(probability=float(p),seed_min=float(low[i]),seed_max=float(high[i]),
                           seed_range_is_confidence_interval=False,review_required=bool(reasons),reasons=reasons))
    return dict(policy=POLICY,findings=result)
