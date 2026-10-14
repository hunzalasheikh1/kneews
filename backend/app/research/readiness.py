"""Inspect actual inputs without treating progress journals as training evidence."""
import hashlib
import json
import time
from collections import Counter
from sqlalchemy import select
from ..models import Study,LabelSet
from ..config import settings
from .cache import read_cache
from .labelling import validate_labels


def validate_label(label,study):
    validate_labels({'findings':label.findings},study.report)
    if any(v['state']!='unmentioned' and v.get('probability') is None for v in label.findings.values()):
        raise ValueError('Local mentioned finding lacks its probability')
    provenance=label.provenance
    if 'report_sha256' not in provenance:raise ValueError('Label report hash is missing')
    if provenance.get('report_sha256')!=hashlib.sha256(study.report.encode()).hexdigest():
        raise ValueError('Label report hash differs from current source')
    expected=':'.join([provenance.get('model',''),provenance.get('digest','')[:12],
                       provenance.get('prompt_sha256','')[:12],provenance.get('schema_version','')])
    if expected!=label.version or provenance.get('local_only') is not True:
        raise ValueError('Label provenance differs from the frozen local version')


def inspect_inputs(db,label_version,signature):
    studies=db.scalars(select(Study).where(Study.split=='train').order_by(Study.id)).all()
    labels={r.study_id:r for r in db.scalars(select(LabelSet).where(LabelSet.method=='local_llm',LabelSet.version==label_version))}
    result=dict(label_version=label_version,feature_signature=signature,total=len(studies),valid_features=0,valid_labels=0,
                missing_features=[],invalid_features=[],missing_labels=[],invalid_labels=[],plane_coverage={},series_decode_errors=0,
                inspected_at=time.time(),human_review_verified=False)
    coverage=Counter()
    for study in studies:
        path=settings.artifact_root/'features'/signature/(study.id+'.npz')
        if not path.exists():result['missing_features'].append(study.id)
        else:
            try:
                _,mask,metadata=read_cache(path,study.id,signature)
                coverage[int(mask.reshape(3,2).any(1).sum())]+=1
                result['series_decode_errors']+=len(metadata.get('errors',[]));result['valid_features']+=1
            except Exception as exc:result['invalid_features'].append(dict(study_id=study.id,error=str(exc)))
        label=labels.get(study.id)
        if label is None:result['missing_labels'].append(study.id)
        else:
            try:validate_label(label,study);result['valid_labels']+=1
            except Exception as exc:result['invalid_labels'].append(dict(study_id=study.id,error=str(exc)))
    result['plane_coverage']=dict(coverage)
    result['ready_for_training']=bool(studies) and result['valid_features']==len(studies) and result['valid_labels']==len(studies)
    root=settings.artifact_root/'readiness';root.mkdir(parents=True,exist_ok=True)
    temporary=root/'latest.tmp';temporary.write_text(json.dumps(result,indent=2));temporary.replace(root/'latest.json')
    return result


def repair_labels(db,label_version):
    from ..models import Review
    from .labelling import installed_model,local_label_version,label_local
    descriptor=installed_model(settings.local_llm_url,settings.local_llm_model)
    if local_label_version(descriptor)!=label_version:
        raise ValueError('Repair requires the exact installed model, prompt and schema version')
    labels=db.scalars(select(LabelSet).where(LabelSet.method=='local_llm',LabelSet.version==label_version)).all()
    repaired=0;unchanged=0
    for label in labels:
        study=db.get(Study,label.study_id)
        try:validate_label(label,study);unchanged+=1;continue
        except ValueError as exc:reason=str(exc)
        if db.scalar(select(Review.id).where(Review.label_set_id==label.id).limit(1)):
            raise ValueError('Cannot replace a reviewed label set; register a new source version instead')
        # Re-run extraction against the original report rather than manufacturing missing provenance.
        findings,provenance=label_local(study.report,settings.local_llm_url,settings.local_llm_model,descriptor)
        output=settings.artifact_root/'label_repairs';output.mkdir(parents=True,exist_ok=True)
        archive=output/f'{label.id}-{time.time_ns()}.json'
        archive.write_text(json.dumps(dict(id=label.id,study_id=label.study_id,version=label.version,
            findings=label.findings,provenance=label.provenance,reason=reason),indent=2,ensure_ascii=False))
        label.findings=findings;label.provenance={**provenance,'repaired_at':time.time(),'previous_source_archive':str(archive),'repair_reason':reason}
        validate_label(label,study);db.commit();repaired+=1
        print(json.dumps(dict(study_id=study.id,status='revalidated',reason=reason)),flush=True)
    return dict(repaired=repaired,already_valid=unchanged,version=label_version)
