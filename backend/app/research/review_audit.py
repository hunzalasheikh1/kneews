"""Report-fidelity audits kept separate from image-derived expert labels."""
from collections import defaultdict
from math import sqrt


def wilson(successes,total,z=1.95996398454):
    if total==0:return None
    p=successes/total;denom=1+z*z/total
    center=(p+z*z/(2*total))/denom
    half=z*sqrt(p*(1-p)/total+z*z/(4*total*total))/denom
    return [max(0,center-half),min(1,center+half)]


def review_metrics(reviews,label_lookup):
    cells=defaultdict(dict)
    for review in sorted(reviews,key=lambda r:r.created_at):
        key=(review.study_id,review.finding,review.label_set_id)
        cells[key][review.reviewer.strip().casefold()]=review.state
    counts=defaultdict(lambda:dict(reviewed=0,errors=0))
    agreed=disagreed=single=0
    for (_,finding,label_id),raters in cells.items():
        if len(raters)<2:single+=1;continue
        if len(set(raters.values()))!=1:disagreed+=1;continue
        label=label_lookup.get(label_id)
        if label is None:continue
        gold=next(iter(raters.values()));predicted=label.findings[finding]['state']
        key=(label.method,label.version,finding)
        counts[key]['reviewed']+=1;counts[key]['errors']+=int(gold!=predicted);agreed+=1
    return dict(agreed_cells=agreed,disputed_cells=disagreed,single_reviewer_cells=single,
        independent_identity_verified=False,
        strata=[dict(method=m,version=v,finding=f,**values,error_rate=values['errors']/values['reviewed'],
                     error_rate_interval=wilson(values['errors'],values['reviewed'])) for (m,v,f),values in counts.items()],
        interpretation='Report fidelity against agreeing named reviewers. This is not MRI diagnostic accuracy. Reviewer independence requires external verification.')
