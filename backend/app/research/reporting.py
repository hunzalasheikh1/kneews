"""Portable experiment reports generated only from complete stored outcomes."""
import csv
import html
import io
import json
import zipfile
from .constants import FINDINGS

LIMITS=[
    'Research use only; no clinical validation or diagnostic-use claim.',
    'Groups connect scanner fingerprints and duplicate reports. Hospital independence has not been established.',
    'Expert cases are cross-fitted: held-out cases do not fit their fold’s reliability estimator or image head.',
    'Independent human report-label accuracy and reviewer independence are not established by this experiment export.',
    'Language codes are automatic and unreviewed. Sparse findings, bins and language strata may have undefined or unstable estimates.',
    'Reliability intervals are conditional beta posteriors and exclude uncertainty in the empirical pooling prior.',
    'Per-seed results are reported separately. Training seeds are not independent patient cohorts.',
    'The reserve arm, when included, is exploratory. Improvement over the binary baseline is a hypothesis, not a guaranteed result.'
]


def completed_runs(experiment):
    if experiment.status!='completed':raise ValueError('Reports require a completed experiment')
    runs=experiment.results.get('runs',[]);seeds=experiment.config.get('seeds',[]);arms=experiment.config.get('arms',[])
    if not seeds or not arms or len(runs)!=len(seeds) or {r['seed'] for r in runs}!=set(seeds):
        raise ValueError('Stored outcomes do not cover every registered seed')
    for run in runs:
        if set(run.get('metrics',{}))!=set(arms):raise ValueError('Stored outcomes do not cover every arm')
        for metrics in run['metrics'].values():
            rows=metrics.get('per_finding',[])
            if len(rows)!=12 or {r['finding'] for r in rows}!=set(FINDINGS):
                raise ValueError('Stored outcomes do not cover every finding')
    return sorted(runs,key=lambda r:r['seed'])


def tables(experiment):
    result={name:[] for name in ['summary','findings','calibration','omission','reliability','comparisons','subgroups','subgroup_findings']}
    for run in completed_runs(experiment):
        seed=run['seed']
        for arm,metrics in run['metrics'].items():
            result['summary'].append(dict(seed=seed,arm=arm,**{k:v for k,v in metrics.items() if k!='per_finding'}))
            for finding in metrics['per_finding']:
                result['findings'].append(dict(seed=seed,arm=arm,**{k:v for k,v in finding.items() if k!='calibration_bins'}))
                for row in finding.get('calibration_bins',[]):
                    result['calibration'].append(dict(seed=seed,arm=arm,finding=finding['finding'],**row))
        for row in run.get('omission',[]):result['omission'].append(dict(seed=seed,**row))
        for subgroup in run.get('subgroups',[]):
            common={k:v for k,v in subgroup.items() if k!='metrics'}
            for arm,metrics in subgroup['metrics'].items():
                result['subgroups'].append(dict(seed=seed,arm=arm,**common,**{k:v for k,v in metrics.items() if k!='per_finding'}))
                for finding in metrics['per_finding']:
                    result['subgroup_findings'].append(dict(seed=seed,arm=arm,axis=subgroup['axis'],stratum=subgroup['stratum'],**finding))
        for estimator in run.get('estimators',[]):
            for row in estimator['table']:
                result['reliability'].append(dict(seed=seed,fold=estimator['fold'],fit_studies=len(estimator['fit_ids']),**row))
        for name in ['paired_brier','paired_auc']:
            if name in run:
                comparison=run[name]
                result['comparisons'].append(dict(seed=seed,bootstrap_seed=comparison.get('seed'),comparison='pooled-minus-binary',
                    exploratory=False,**{k:v for k,v in comparison.items() if k!='seed'}))
        for row in [run.get('reserve_vs_binary',{}).get(metric) for metric in ['brier','auc']]:
            if row:result['comparisons'].append(dict(seed=seed,bootstrap_seed=row.get('seed'),comparison='reserve-minus-binary',
                exploratory=True,**{k:v for k,v in row.items() if k!='seed'}))
    return result


def csv_text(rows):
    keys=list(dict.fromkeys(key for row in rows for key in row))
    stream=io.StringIO(newline='');writer=csv.DictWriter(stream,fieldnames=keys)
    writer.writeheader()
    for row in rows:
        values={}
        for key,value in row.items():
            if isinstance(value,(list,dict)):value=json.dumps(value,ensure_ascii=False)
            # Prevent spreadsheet formula evaluation while retaining ordinary numeric cells.
            if isinstance(value,str) and value.lstrip().startswith(('=','+','-','@')):value="'"+value
            values[key]=value
        writer.writerow(values)
    return stream.getvalue()


def display(value):
    if value is None:return 'Not estimable'
    if isinstance(value,float):return f'{value:.4f}'
    if isinstance(value,(list,dict)):return json.dumps(value,ensure_ascii=False)
    return str(value)


def table_html(rows):
    if not rows:return '<p>No entries for this experiment.</p>'
    keys=list(rows[0])
    return '<div class="table"><table><thead><tr>'+''.join('<th>'+html.escape(k.replace('_',' '))+'</th>' for k in keys)+'</tr></thead><tbody>'+''.join(
        '<tr>'+''.join('<td>'+html.escape(display(row.get(key)))+'</td>' for key in keys)+'</tr>' for row in rows)+'</tbody></table></div>'


def report_html(experiment):
    data=tables(experiment);config=experiment.config
    title=html.escape(experiment.name)
    provenance=dict(experiment_id=experiment.id,manifest_hash=experiment.manifest_hash,
        label_version=config.get('label_version'),feature_signature=config.get('feature_signature'),
        training_device=config.get('training_device'),
        repaired_label_sets=len(config['label_repairs']) if 'label_repairs' in config else None,
        evidence_repair_policies=sorted({r['version'] for r in config.get('label_repairs',[])}),
        training_studies=config.get('dataset_count'),seeds=config.get('seeds'),epochs=config.get('epochs'),
        registered_primary_protocol_match=config.get('registered_primary_protocol_match'),protocol_deviations=config.get('protocol_deviations',{}),
        expert_cases=len(completed_runs(experiment)[0].get('study_ids',[])))
    data['subgroups']=[{key:row.get(key) for key in ['seed','arm','axis','stratum','n','independent_groups','macro_auc','worst_finding_auc','brier','evaluable_findings']} for row in data['subgroups']]
    sections=''.join('<section><h2>'+name+'</h2>'+table_html(data[key])+'</section>' for key,name in [
        ('summary','Per-seed arm summary'),('comparisons','Paired comparisons'),('findings','Per-finding evaluation'),('subgroups','Descriptive subgroup evaluation'),
        ('omission','Omission among expert-positive cases'),('reliability','Training-fold reliability estimates'),
        ('calibration','Calibration bins')])
    return ('<!doctype html><html lang="en"><meta charset="utf-8"><meta name="viewport" content="width=device-width, initial-scale=1"><title>'+title+'</title><style>'+
        'body{font:14px/1.6 system-ui,sans-serif;color:#183440;max-width:1200px;margin:40px auto;padding:0 24px}h1{font-size:28px}h2{font-size:20px;margin-top:36px}section{margin:28px 0}.table{overflow:auto}table{border-collapse:collapse;width:100%;font-size:12px}th,td{border:1px solid #d5dfe1;padding:8px;text-align:left}th{background:#eef5f3}li{margin:8px 0}pre{white-space:pre-wrap;overflow-wrap:anywhere;background:#f4f7f7;padding:18px}@media print{body{margin:0;font-size:11px}thead{display:table-header-group}tr{break-inside:avoid}.table{overflow:visible}th,td{font-size:9px;padding:4px}}'+
        '</style><body><h1>'+title+'</h1><p>KneeWS · Completed experiment research report</p><h2>Interpretation and limits</h2><ul>'+''.join('<li>'+html.escape(x)+'</li>' for x in LIMITS)+'</ul><h2>Provenance</h2><pre>'+html.escape(json.dumps(provenance,indent=2))+'</pre>'+sections+
        '<h2>Method lineage</h2><p>Reliability-based noisy-label learning: <a href="https://arxiv.org/abs/2104.05261">Gündel et al.</a>; gold-free reserve inspired by <a href="https://arxiv.org/abs/1911.00068">Confident Learning</a>. These references describe prior methods; their reported results are not KneeWS results.</p></body></html>')


def report_bundle(experiment):
    data=tables(experiment);buffer=io.BytesIO()
    with zipfile.ZipFile(buffer,'w',compression=zipfile.ZIP_DEFLATED) as archive:
        for name,rows in data.items():archive.writestr(name+'.csv',csv_text(rows))
        archive.writestr('report.html',report_html(experiment))
        archive.writestr('provenance.json',json.dumps(dict(id=experiment.id,name=experiment.name,
            manifest_hash=experiment.manifest_hash,config=experiment.config),indent=2))
        archive.writestr('README.txt','KneeWS completed experiment tables. Empty numeric CSV cells mean not estimable, not zero.\n'+
                         '\n'.join(LIMITS)+'\nSee provenance.json for registered configurations and amendments.\n')
    return buffer.getvalue()
