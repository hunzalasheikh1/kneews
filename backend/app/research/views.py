"""Read models for research screens; no raw reports in aggregate endpoints."""
from collections import Counter,defaultdict
from sqlalchemy import select,func
from sqlalchemy.orm import aliased
from ..models import Study,LabelSet,Review,Experiment
from .constants import FINDINGS


def label_versions(db):
    total=db.scalar(select(func.count()).select_from(Study).where(Study.split=='train'))
    rows=db.execute(select(LabelSet.method,LabelSet.version,func.count(),func.max(LabelSet.created_at))
        .join(Study,Study.id==LabelSet.study_id).where(Study.split=='train')
        .group_by(LabelSet.method,LabelSet.version).order_by(func.max(LabelSet.created_at).desc())).all()
    return [dict(method=method,version=version,studies=count,total_studies=total,
                 coverage=count/total if total else 0,complete=count==total,latest_at=latest) for method,version,count,latest in rows]


def disagreements(db,left_version,right_version,language=None,offset=0,limit=30):
    left=aliased(LabelSet);right=aliased(LabelSet)
    query=select(Study.id,Study.language,left.id,right.id,left.findings,right.findings).join(left,left.study_id==Study.id).join(right,right.study_id==Study.id).where(
        Study.split=='train',left.version==left_version,right.version==right_version).order_by(Study.id)
    if language:query=query.where(Study.language==language)
    items=[];total=0;counts=Counter()
    for study_id,lang,left_id,right_id,a,b in db.execute(query):
        for finding in FINDINGS:
            if a[finding]['state']==b[finding]['state']:continue
            counts[finding]+=1
            if offset<=total<offset+limit:
                items.append(dict(study_id=study_id,language=lang,finding=finding,left_state=a[finding]['state'],
                    right_state=b[finding]['state'],left_label_set_id=left_id,right_label_set_id=right_id))
            total+=1
    return dict(total=total,items=items,by_finding=dict(counts),interpretation='A disagreement indicates a review candidate, not which source is correct.')


def review_queue(db,version,language=None,offset=0,limit=30):
    query=select(Study,LabelSet).join(LabelSet,LabelSet.study_id==Study.id).where(Study.split=='train',LabelSet.version==version)
    if language:query=query.where(Study.language==language)
    reviews=db.scalars(select(Review).join(LabelSet,LabelSet.id==Review.label_set_id)
                      .where(LabelSet.version==version).order_by(Review.created_at,Review.id)).all()
    reviewed=defaultdict(lambda:defaultdict(dict))
    for review in reviews:
        reviewed[(review.study_id,review.label_set_id)][review.finding][review.reviewer.strip().casefold()]=review.state
    items=[]
    for study,label in db.execute(query.order_by(Study.language,Study.id)):
        cells=reviewed.get((study.id,label.id),{})
        n=len({name for raters in cells.values() for name in raters})
        covered=sum(len(raters)>=2 for raters in cells.values())
        disputed=sum(len(raters)>=2 and len(set(raters.values()))>1 for raters in cells.values())
        items.append(dict(study_id=study.id,language=study.language,label_set_id=label.id,reviewers=n,
            two_reviewer_findings=covered,disputed_findings=disputed,fully_reviewed=covered==len(FINDINGS),
            uncertain_findings=sum(x['state']=='uncertain' for x in label.findings.values()),
            unmentioned_findings=sum(x['state']=='unmentioned' for x in label.findings.values())))
    # Unreviewed first; within it retain language and study ordering without gold-label access.
    items.sort(key=lambda x:(x['fully_reviewed'] and not x['disputed_findings'],x['language'],x['study_id']))
    return dict(total=len(items),items=items[offset:offset+limit],language_source='automatic_unreviewed')


def experiment_summary(row):
    runs=row.results.get('runs',[])
    return dict(id=row.id,name=row.name,status=row.status,created_at=row.created_at,manifest_hash=row.manifest_hash,
                label_version=row.config.get('label_version'),arms=row.config.get('arms',[]),seeds=row.config.get('seeds',[]),
                completed_seeds=len(runs),metrics=[dict(seed=r['seed'],arms=r.get('metrics',{})) for r in runs],
                error=row.results.get('error'),progress=row.results.get('progress'))
