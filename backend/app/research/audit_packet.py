"""Deterministic, blinded report review packets. No expert targets or model states."""
from collections import Counter,defaultdict
import hashlib
import json
from pathlib import Path
from sqlalchemy import select,text
from ..models import Study,LabelSet,Review
from .constants import FINDINGS,STATE_INDEX


def digest(value):
    return hashlib.sha256(value.encode()).hexdigest()


def select_sample(studies,per_language=10,seed=2026):
    if per_language<1:raise ValueError('A positive sample size per language is required')
    groups=defaultdict(list)
    for study in studies:groups[study.language].append(study)
    return [study for language,items in sorted(groups.items())
            for study in sorted(items,key=lambda s:digest(f'{seed}:{s.id}'))[:per_language]]


def export_packet(db,root,label_version,per_language=10,seed=2026):
    studies=db.scalars(select(Study).where(Study.split=='train').order_by(Study.id)).all()
    if not studies:raise ValueError('No training reports imported')
    selected=select_sample(studies,per_language,seed)
    manifest=dict(schema='kneews-blind-audit-1',label_version=label_version,seed=seed,
        per_language=per_language,population=dict(Counter(s.language for s in studies)),
        language_source='automatic_unreviewed',selection_basis='All training reports, independent of model output and label availability',
        studies=[dict(study_id=s.id,language=s.language,report_sha256=digest(s.report)) for s in selected])
    packet_id=digest(json.dumps(manifest,sort_keys=True))
    output=Path(root)/packet_id
    if output.exists():raise ValueError(f'Audit packet already exists: {output}; refusing to overwrite reviewer work')
    output.mkdir(parents=True)
    (output/'manifest.json').write_text(json.dumps(manifest,indent=2))
    worksheet=dict(packet_id=packet_id,reviewer='',role='',languages_reviewed=[],
        studies=[dict(study_id=s.id,report=s.report,confirmed_language='',
                      findings={finding:dict(state='',evidence='',note='') for finding in FINDINGS}) for s in selected])
    for reviewer in ('a','b'):
        (output/f'reviewer-{reviewer}.json').write_text(json.dumps(worksheet,indent=2,ensure_ascii=False))
    return dict(packet_id=packet_id,path=str(output),reports=len(selected),cells_per_reviewer=len(selected)*len(FINDINGS))


def import_packet(db,manifest_path,worksheet_path):
    manifest=json.loads(Path(manifest_path).read_text())
    worksheet=json.loads(Path(worksheet_path).read_text())
    packet_id=digest(json.dumps(manifest,sort_keys=True))
    if manifest.get('schema')!='kneews-blind-audit-1' or worksheet.get('packet_id')!=packet_id:
        raise ValueError('Worksheet does not match the audit manifest')
    reviewer=worksheet.get('reviewer','').strip()
    if not 2<=len(reviewer)<=128:raise ValueError('Provide a reviewer name of 2–128 characters')
    if not worksheet.get('role','').strip() or not worksheet.get('languages_reviewed'):
        raise ValueError('Reviewer role and reviewed languages are required')
    expected={s['study_id']:s for s in manifest['studies']}
    rows=worksheet.get('studies',[])
    if len(rows)!=len(expected) or {s['study_id'] for s in rows}!=set(expected):
        raise ValueError('Worksheet must contain exactly the sampled studies, without duplicates')
    # Serialize duplicate imports of this packet/reviewer. Every row is validated before writing.
    db.execute(text('SELECT pg_advisory_xact_lock(hashtext(:key))'),{'key':f'audit:{packet_id}:{reviewer.casefold()}'})
    pending=[];skipped=0
    for row in rows:
        study=db.get(Study,row['study_id'])
        if study is None or digest(study.report)!=expected[row['study_id']]['report_sha256'] or row.get('report')!=study.report:
            raise ValueError('Report changed or worksheet source text was edited')
        language=row.get('confirmed_language','').strip()
        if not language or language not in worksheet['languages_reviewed']:
            raise ValueError('Each report requires a confirmed language listed in reviewer languages')
        label=db.scalar(select(LabelSet).where(LabelSet.study_id==study.id,LabelSet.method=='local_llm',LabelSet.version==manifest['label_version']))
        if label is None:raise ValueError('The selected local label version is not available for every sampled report')
        if set(row.get('findings',{}))!=set(FINDINGS):raise ValueError('Every report must contain all 12 findings')
        for finding,value in row['findings'].items():
            state=value.get('state');evidence=value.get('evidence','')
            if state not in STATE_INDEX:raise ValueError('Every finding requires a valid four-state decision')
            if not isinstance(evidence,str) or (state=='unmentioned' and evidence) or (state!='unmentioned' and (not evidence.strip() or evidence not in study.report)):
                raise ValueError('Mentioned states require an exact report quotation; unmentioned requires empty evidence')
            note=json.dumps(dict(packet_id=packet_id,role=worksheet['role'],confirmed_language=language,
                                 reviewer_note=value.get('note',''),independence_verified=False),ensure_ascii=False,sort_keys=True)
            previous=db.scalars(select(Review).where(Review.study_id==study.id,Review.label_set_id==label.id,Review.finding==finding)).all()
            same=[r for r in previous if r.reviewer.strip().casefold()==reviewer.casefold()]
            if same:
                latest=max(same,key=lambda r:r.created_at)
                if (latest.state,latest.evidence,latest.note)!=(state,evidence,note):
                    raise ValueError('Conflicting existing review; use the review UI for an explicit correction')
                skipped+=1;continue
            pending.append(Review(study_id=study.id,label_set_id=label.id,finding=finding,state=state,
                                  evidence=evidence,reviewer=reviewer,note=note))
    db.add_all(pending);db.commit()
    return dict(packet_id=packet_id,imported=len(pending),already_imported=skipped,independence_verified=False)
