from types import SimpleNamespace
import csv
import io
import zipfile
import numpy as np
import pytest
from app.research.metrics import evaluate
from app.research.reporting import report_bundle,report_html,csv_text,tables


def experiment():
    metrics=evaluate(np.zeros((2,12)),np.ones((2,12))*.2)
    return SimpleNamespace(id='synthetic-report',name='<script>untrusted title</script>',status='completed',manifest_hash='fixture',
        config=dict(seeds=[2026],arms=['binary']),results={'runs':[dict(seed=2026,study_ids=['a','b'],metrics={'binary':metrics})]})


def test_report_preserves_undefined_metrics_and_all_calibration_bins():
    row=experiment()
    with zipfile.ZipFile(io.BytesIO(report_bundle(row))) as archive:
        findings=list(csv.DictReader(io.StringIO(archive.read('findings.csv').decode())))
        assert len(findings)==12 and all(r['auc']=='' for r in findings)
        assert len(list(csv.DictReader(io.StringIO(archive.read('calibration.csv').decode()))))==120
        assert 'provenance.json' in archive.namelist()
    text=report_html(row)
    assert '<script>' not in text and '&lt;script&gt;' in text
    assert 'Not estimable' in text and 'Hospital independence has not been established' in text
    assert 'Independent human report-label accuracy' in text


def test_report_rejects_partial_runs_and_incomplete_findings():
    row=experiment();row.status='running'
    with pytest.raises(ValueError,match='completed'):report_html(row)
    row.status='completed';row.config['seeds'].append(2027)
    with pytest.raises(ValueError,match='every registered seed'):report_bundle(row)
    row.config['seeds']=[2026];row.results['runs'][0]['metrics']['binary']['per_finding'].pop()
    with pytest.raises(ValueError,match='every finding'):report_bundle(row)


def test_csv_does_not_execute_text_formulas_or_turn_missing_values_into_zero():
    text=csv_text([{'label':'=1+1','delta':-.25,'missing':None}])
    row=next(csv.DictReader(io.StringIO(text)))
    assert row=={'label':"'=1+1",'delta':'-0.25','missing':''}


def test_comparison_export_distinguishes_training_and_bootstrap_seeds():
    row=experiment();row.config['seeds']=[2027]
    run=row.results['runs'][0];run['seed']=2027
    comparison=dict(seed=2026,metric='brier',delta_interval=[-.02,.01],valid_replicates=1000,requested_replicates=1000)
    run['paired_brier']=comparison;run['reserve_vs_binary']={'brier':comparison,'exploratory':True}
    rows=tables(row)['comparisons']
    assert len(rows)==2 and all(r['seed']==2027 and r['bootstrap_seed']==2026 for r in rows)
    assert rows[0]['exploratory'] is False and rows[1]['exploratory'] is True
