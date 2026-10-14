"""ABBA throughput check on six median-length language-stratified reports."""
import hashlib
import json
import time
from collections import Counter
from sqlalchemy import select
from ..models import Study,LabelSet
from ..config import settings
from .labelling import installed_model,local_label_version,label_local
from .batching import concurrent_results


def benchmark(db,baseline_workers=1,workers=2):
    if not 1<=baseline_workers<workers<=4:raise ValueError('Benchmark requires 1 <= baseline workers < workers <= 4')
    descriptor=installed_model(settings.local_llm_url,settings.local_llm_model)
    version=local_label_version(descriptor)
    studies=db.scalars(select(Study).where(Study.split=='train').order_by(Study.id)).all()
    existing={r.study_id:r.findings for r in db.scalars(select(LabelSet).where(LabelSet.method=='local_llm',LabelSet.version==version))}
    if baseline_workers>1:
        studies=[s for s in studies if s.id in existing]
        if not studies:raise ValueError('Higher-concurrency benchmark requires existing frozen-version labels')
    languages=[language for language,_ in Counter(s.language for s in studies).most_common(6)]
    selected=[]
    for language in languages:
        candidates=sorted([s for s in studies if s.language==language],key=lambda s:(len(s.report),s.id))
        selected.append(candidates[len(candidates)//2])
    reference={s.id:existing[s.id] for s in selected if s.id in existing};measurements=[]
    output=settings.artifact_root/'benchmarks';output.mkdir(parents=True,exist_ok=True)
    result=dict(version=local_label_version(descriptor),sample=[dict(id=s.id,language=s.language,characters=len(s.report),
        report_sha256=hashlib.sha256(s.report.encode()).hexdigest()) for s in selected],rounds=measurements,
        scope='Engineering throughput/consistency check; not a clinical accuracy audit. Concurrent image extraction may affect timings.',
        order=[baseline_workers,workers,workers,baseline_workers],
        existing_reference_count=len(reference),
        acceptance='Parallel throughput faster, no changed finding states or source evidence, and probability differences <=0.01 versus existing saved labels when present, otherwise first baseline pass')
    destination=output/f'labels-concurrency-{baseline_workers}-vs-{workers}-{time.time_ns()}.json'
    def infer(study):return label_local(study.report,settings.local_llm_url,settings.local_llm_model,descriptor)
    for request_workers in result['order']:
        started=time.perf_counter();rows=[]
        for study,value,error in concurrent_results(infer,selected,request_workers):
            if error:raise error
            findings,provenance=value
            if study.id not in reference:reference[study.id]=findings
            previous=reference[study.id]
            rows.append(dict(id=study.id,seconds=provenance['wall_seconds'],runtime=provenance['runtime'],
                state_changes=sum(previous[k]['state']!=v['state'] for k,v in findings.items()),
                evidence_changes=sum(previous[k]['evidence']!=v['evidence'] for k,v in findings.items()),
                max_probability_delta=max(abs((previous[k]['probability'] or 0)-(v['probability'] or 0)) for k,v in findings.items())))
        measurements.append(dict(workers=request_workers,seconds=time.perf_counter()-started,reports=rows))
        destination.write_text(json.dumps(result,indent=2))
        print(json.dumps(dict(round=len(measurements),workers=request_workers,seconds=measurements[-1]['seconds'],
                              state_changes=sum(r['state_changes'] for r in rows),evidence_changes=sum(r['evidence_changes'] for r in rows))),flush=True)
    elapsed={w:sum(r['seconds'] for r in measurements if r['workers']==w) for w in [baseline_workers,workers]}
    result['speedup']=elapsed[baseline_workers]/elapsed[workers]
    result['accepted']=result['speedup']>1 and all(r['state_changes']==0 and r['evidence_changes']==0 and r['max_probability_delta']<=.01 for m in measurements for r in m['reports'])
    destination.write_text(json.dumps(result,indent=2))
    print(json.dumps(dict(speedup=result['speedup'],accepted=result['accepted'])),flush=True)
