import argparse
from collections import Counter
import csv
import hashlib
import json
from pathlib import Path
import time
import numpy as np
from sqlalchemy import select
from ..config import settings,ROOT
from ..db import SessionLocal
from ..models import Study,LabelSet
from .constants import FINDINGS,STATE_INDEX
from .splits import make_folds


def audit(db):
    studies=db.scalars(select(Study).where(Study.split=='train').order_by(Study.id)).all()
    counts=Counter(s.site_group for s in studies)
    records=[dict(id=s.id,site_group=s.site_group,report_hash=s.report_hash,language=s.language,is_gold=s.is_gold) for s in studies]
    folds=make_folds(records)
    output=dict(train_studies=len(studies),expert_studies=sum(s.is_gold for s in studies),
                series=sum(len(s.series) for s in studies),scanner_proxy_groups=len(counts),
                unknown_group_studies=counts['unknown'],hospital_identity_verified=False,
                largest_groups=counts.most_common(10),gold_per_fold=dict(Counter(r['fold'] for r in folds if r['is_gold'])),
                studies_per_fold=dict(Counter(r['fold'] for r in folds)),independent_groups=len(set(r['leakage_group'] for r in folds)),
                duplicated_report_groups=sum(n>1 for n in Counter(s.report_hash for s in studies if s.report_hash).values()))
    character_lengths=np.asarray([len(s.report) for s in studies])
    byte_lengths=np.asarray([len(s.report.encode('utf-8')) for s in studies])
    output['report_lengths']={name:dict(max=int(values.max()),median=float(np.median(values)),p95=float(np.quantile(values,.95)))
                              for name,values in [('characters',character_lengths),('utf8_bytes',byte_lengths)]}
    root=settings.artifact_root/'audit';root.mkdir(parents=True,exist_ok=True)
    (root/'dataset.json').write_text(json.dumps(output,indent=2))
    (root/'folds.json').write_text(json.dumps(folds,indent=2))
    print(json.dumps(output,indent=2),flush=True)


def features(db,limit,device):
    from .features import FrozenEncoder,cache_study
    from .cache import read_cache
    encoder=FrozenEncoder(ROOT/'models/dinov2-small',device=device)
    studies=db.scalars(select(Study).where(Study.split=='train').order_by(Study.id)).all()
    if limit: studies=studies[:limit]
    cache=settings.artifact_root/'features'/encoder.signature
    cache.mkdir(parents=True,exist_ok=True)
    manifest=dict(signature=encoder.signature,config=encoder.config,device=str(encoder.device),total=len(studies),completed=0,failed=[],invalidated=[])
    start=time.monotonic()
    for i,s in enumerate(studies):
        path=cache/(s.id+'.npz')
        if path.exists():
            try:read_cache(path,s.id,encoder.signature,encoder.model.config.hidden_size*2)
            except Exception as exc:
                quarantined=path.with_suffix(f'.invalid-{time.time_ns()}')
                path.replace(quarantined)
                manifest['invalidated'].append(dict(study_id=s.id,error=str(exc),preserved_path=str(quarantined)))
        if not path.exists():
            try:
                x,mask,metadata=encoder.study(s.series,settings.data_root)
                cache_study(path,x,mask,{**metadata,'study_id':s.id})
            except Exception as exc:
                manifest['failed'].append(dict(study_id=s.id,error=str(exc)))
        if path.exists(): manifest['completed']+=1
        manifest['elapsed_seconds']=time.monotonic()-start
        temp=cache/'manifest.tmp';temp.write_text(json.dumps(manifest,indent=2));temp.replace(cache/'manifest.json')
        print(json.dumps(dict(study=i+1,total=len(studies),completed=manifest['completed'],elapsed=round(manifest['elapsed_seconds']),device=str(encoder.device))),flush=True)
    (settings.artifact_root/'features'/'latest.json').write_text(json.dumps({'signature':encoder.signature}))


def local_labels(db,limit,per_language=None,workers=1,repair_invalid_evidence=False):
    from .leases import stage_lease
    with stage_lease('local-label-generation'):
        return _local_labels(db,limit,per_language,workers,repair_invalid_evidence)


def _local_labels(db,limit,per_language=None,workers=1,repair_invalid_evidence=False):
    import httpx
    from .labelling import label_local,installed_model,local_label_version
    from collections import defaultdict
    from .batching import concurrent_results
    if not 1<=workers<=4:raise ValueError('Use one to four local requests')
    descriptor=installed_model(settings.local_llm_url,settings.local_llm_model)
    version=local_label_version(descriptor)
    studies=db.scalars(select(Study).where(Study.split=='train').order_by(Study.id)).all()
    if per_language:
        buckets=defaultdict(list)
        for study in studies:buckets[study.language].append(study)
        studies=[]
        for language,items in sorted(buckets.items()):
            ordered=sorted(items,key=lambda s:hashlib.sha256(('pilot-2026:'+s.id).encode()).hexdigest())
            studies.extend(ordered[:per_language])
    elif limit: studies=studies[:limit]
    existing=set(db.scalars(select(LabelSet.study_id).where(LabelSet.method=='local_llm',LabelSet.version==version)).all())
    output=settings.artifact_root/'labels'/version.replace(':','_');output.mkdir(parents=True,exist_ok=True)
    run_id=str(time.time_ns())
    status=dict(version=version,model_digest=descriptor['digest'],total=len(studies),completed=0,skipped=0,
                failures=[],status='running',selection='stratified_language' if per_language else ('first_n' if limit else 'full_corpus'),
                selected_ids=[s.id for s in studies],source_languages=dict(Counter(s.language for s in studies)),
                started_at=time.time(),updated_at=time.time(),workers=workers,repair_invalid_evidence=repair_invalid_evidence)
    def persist():
        status['updated_at']=time.time()
        temp=output/(run_id+'.tmp');temp.write_text(json.dumps(status,indent=2));temp.replace(output/(run_id+'.json'))
    persist()
    try:
        pending=[(i,study.id,study.report) for i,study in enumerate(studies) if study.id not in existing]
        status['completed']=len(studies)-len(pending);status['skipped']=status['completed'];persist()
        def infer(item):
            extra={'repair_invalid_evidence':True} if repair_invalid_evidence else {}
            return label_local(item[2],settings.local_llm_url,settings.local_llm_model,descriptor,**extra)
        for (i,study_id,_),result,error in concurrent_results(infer,pending,workers):
            try:
                if error is not None:raise error
                labels,provenance=result
                db.add(LabelSet(study_id=study_id,method='local_llm',version=version,findings=labels,
                                provenance={**provenance,'client_workers':workers}));db.commit()
                status['completed']+=1
                print(json.dumps(dict(study=i+1,total=len(studies),completed=status['completed'],id=study_id,status='validated')),flush=True)
            except httpx.TransportError as exc:
                db.rollback();status['status']='interrupted';status['transport_error']=str(exc);persist();raise
            except Exception as exc:
                db.rollback();status['failures'].append(dict(study_id=study_id,error=str(exc)))
                print(json.dumps(dict(study=i+1,total=len(studies),status='failed',error=str(exc))),flush=True)
            persist()
        status['status']='failed' if status['failures'] else 'completed';persist()
        if status['failures']:raise RuntimeError(f"{len(status['failures'])} reports failed validation; see {output}")
    except BaseException:
        if status['status']=='running':status['status']='interrupted';persist()
        raise


def languages(db):
    from lingua import LanguageDetectorBuilder
    detector=LanguageDetectorBuilder.from_all_languages().build()
    studies=db.scalars(select(Study).where(Study.split=='train')).all()
    counts=Counter()
    for i,s in enumerate(studies):
        detected=detector.detect_language_of(s.report)
        s.language=detected.iso_code_639_1.name.lower() if detected else 'unknown'
        s.metadata_json={**s.metadata_json,'language_source':'lingua-automatic-unreviewed'}
        counts[s.language]+=1
        if i%100==0:db.commit();print(f'Language identification {i}/{len(studies)}',flush=True)
    db.commit();print(json.dumps(counts,indent=2))


def main():
    parser=argparse.ArgumentParser();parser.add_argument('task',choices=['audit','audit-label-runtime','features','local-labels','languages','train','benchmark','benchmark-sampling','audit-export','audit-import','readiness','repair-labels','benchmark-labels','verify-device','final-fit'])
    parser.add_argument('--limit',type=int);parser.add_argument('--device',default='auto')
    parser.add_argument('--per-language',type=int)
    parser.add_argument('--workers',type=int,choices=[1,2,3,4],default=1)
    parser.add_argument('--repair-invalid-evidence',action='store_true',help='Allow two bounded source-ID repair attempts after the original three extraction attempts fail; record repair provenance')
    parser.add_argument('--baseline-workers',type=int,default=1)
    parser.add_argument('--benchmark-workers',type=int,default=2)
    parser.add_argument('--label-version');parser.add_argument('--feature-signature')
    parser.add_argument('--source-experiment')
    parser.add_argument('--include-reserve',action='store_true',help='Add the exploratory gold-free nested-OOF reserve arm to the four registered arms')
    parser.add_argument('--wait-for-inputs',action='store_true',help='Wait for full validated inputs, then start a fresh registered training process')
    parser.add_argument('--manifest',type=Path);parser.add_argument('--worksheet',type=Path)
    args=parser.parse_args();settings.artifact_root.mkdir(parents=True,exist_ok=True)
    if args.task=='verify-device':
        from .devices import verify_training_device
        print(json.dumps(verify_training_device(args.device),indent=2));return
    with SessionLocal() as db:
        if args.task=='audit':audit(db)
        elif args.task=='final-fit':
            if not args.source_experiment:parser.error('Final fitting requires --source-experiment')
            from ..models import Experiment
            from .final_fit import run_final_fit
            from .leases import stage_lease
            parent=db.get(Experiment,args.source_experiment)
            if parent is None:parser.error('Source experiment not found')
            with stage_lease('image-training'):
                print(run_final_fit(parent,device=args.device))
        elif args.task=='audit-label-runtime':
            if not args.label_version:parser.error('Runtime audit requires --label-version')
            from .runtime_audit import audit as audit_runtime
            print(json.dumps(audit_runtime(db,args.label_version),indent=2))
        elif args.task=='repair-labels':
            if not args.label_version:parser.error('Label repair requires --label-version')
            from .readiness import repair_labels
            print(json.dumps(repair_labels(db,args.label_version)))
        elif args.task=='readiness':
            if not args.label_version or not args.feature_signature:parser.error('Readiness requires --label-version and --feature-signature')
            from .readiness import inspect_inputs
            result=inspect_inputs(db,args.label_version,args.feature_signature)
            print(json.dumps({k:len(v) if isinstance(v,list) else v for k,v in result.items()},indent=2))
        elif args.task=='audit-export':
            if not args.label_version:parser.error('Audit export requires --label-version')
            from .audit_packet import export_packet
            print(json.dumps(export_packet(db,settings.artifact_root/'review_packets',args.label_version,args.per_language or 10)))
        elif args.task=='audit-import':
            if not args.manifest or not args.worksheet:parser.error('Audit import requires --manifest and --worksheet')
            from .audit_packet import import_packet
            print(json.dumps(import_packet(db,args.manifest,args.worksheet)))
        elif args.task=='features':features(db,args.limit,args.device)
        elif args.task=='local-labels':local_labels(db,args.limit,args.per_language,args.workers,args.repair_invalid_evidence)
        elif args.task=='languages':languages(db)
        elif args.task=='benchmark-labels':
            from .benchmark_labels import benchmark
            benchmark(db,args.baseline_workers,args.benchmark_workers)
        elif args.task=='benchmark-sampling':
            from .benchmark_sampling import benchmark
            benchmark(db)
        elif args.task=='benchmark':
            from .benchmark import benchmark
            benchmark(db)
        elif args.task=='train':
            if not args.label_version or not args.feature_signature:
                parser.error('Training requires --label-version and --feature-signature')
            if args.wait_for_inputs:
                from .wait_training import wait_and_train
                wait_and_train(args.label_version,args.feature_signature,args.include_reserve,args.device)
                return
            from .experiments import run_experiment
            from .training import ARMS
            arms=(*ARMS,'reserve') if args.include_reserve else ARMS
            from .leases import stage_lease
            with stage_lease('image-training'):
                print(run_experiment(db,args.label_version,args.feature_signature,arms=arms,device=args.device))

if __name__=='__main__': main()
