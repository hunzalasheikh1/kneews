import numpy as np
import pytest
from app.research.reliability import PooledReliability
from app.research.splits import make_folds, assert_no_leakage, connected_groups
from app.research.metrics import evaluate, omission_rates


def test_posterior_uses_observed_evidence_and_marks_empty_cells():
    states=np.zeros((20,12),int); gold=np.ones((20,12))
    model=PooledReliability().fit(states,gold,[str(i) for i in range(20)])
    assert model.transform(states).min()>.9
    absent=[r for r in model.table() if r['state']=='absent']
    assert all(r['prior_only'] and r['lower']<r['probability']<r['upper'] for r in absent)


def test_fit_rejects_heldout_gold():
    with pytest.raises(ValueError,match='leaked'):
        PooledReliability().fit(np.zeros((1,12)),np.ones((1,12)),['a'],['a'])


def test_duplicate_reports_join_scanners_transitively():
    rows=[dict(id='a',site_group='s1',report_hash='r1'),dict(id='b',site_group='s2',report_hash='r1'),
          dict(id='c',site_group='s2',report_hash='r2'),dict(id='d',site_group='s3',report_hash='r3')]
    groups=connected_groups(rows)
    assert groups[0]==groups[1]==groups[2] and groups[3]!=groups[0]
    assert_no_leakage(make_folds(rows,2))


def test_unknown_groups_cannot_be_split_as_sites():
    with pytest.raises(ValueError,match='independent groups'):
        make_folds([dict(id=str(i)) for i in range(10)],5)


def test_metrics_do_not_hide_undefined_classes():
    y=np.zeros((10,12)); p=np.full((10,12),.1)
    result=evaluate(y,p)
    assert result['macro_auc'] is None and result['evaluable_findings']==0
    assert result['brier']==pytest.approx(.01)


def test_omission_denominator_is_expert_positives():
    states=np.full((4,12),3); gold=np.zeros((4,12)); gold[:2]=1; states[0]=0
    result=omission_rates(states,gold,['en']*4)
    assert result[0]['omission_rate']==.5 and result[0]['expert_positives']==2


def test_calibration_bins_include_boundaries_and_empty_bins():
    from app.research.metrics import calibration_bins,expected_calibration_error
    y=np.array([0,1,1,0]);p=np.array([0,.1,.95,1])
    bins=calibration_bins(y,p)
    assert sum(b['count'] for b in bins)==4
    assert bins[0]['count']==1 and bins[1]['count']==1 and bins[-1]['count']==2
    assert bins[2]['mean_probability'] is None and bins[2]['observed_frequency'] is None
    weighted=sum(b['count']/len(y)*abs(b['mean_probability']-b['observed_frequency']) for b in bins if b['count'])
    assert weighted==pytest.approx(expected_calibration_error(y,p))


def test_omission_bootstrap_uses_whole_groups_and_retains_undefined_strata():
    states=np.full((6,12),3);states[3:]=0;gold=np.ones((6,12));gold[:,1]=0
    result=omission_rates(states,gold,['en']*3+['nl']*3,groups=['a']*3+['b']*3,iterations=100)
    all_acl=next(r for r in result if r['language']=='all' and r['finding']=='ACL')
    assert all_acl['omission_rate']==.5 and all_acl['omission_interval']==[0,1]
    assert all_acl['valid_replicates']==100 and all_acl['positive_groups']==2
    no_positive=next(r for r in result if r['language']=='all' and r['finding']=='MCL')
    assert no_positive['omission_rate'] is None and no_positive['omission_interval'] is None
    assert no_positive['valid_replicates']==0
    one_group=next(r for r in result if r['language']=='en' and r['finding']=='ACL')
    assert one_group['omission_rate']==1 and one_group['omission_interval'] is None


def test_subgroup_results_retain_rare_languages_and_undefined_auc():
    from app.research.metrics import subgroup_evaluation
    gold=np.tile([[0],[1],[1]],(1,12));predictions={'binary':np.full((3,12),.5)}
    records=[dict(language='en',leakage_group='a'),dict(language='en',leakage_group='b'),dict(language='rare',leakage_group='c')]
    rows=subgroup_evaluation(gold,predictions,records)
    language={r['stratum']:r for r in rows if r['axis']=='language'}
    assert language['en']['n']==2 and language['en']['independent_groups']==2
    assert language['en']['metrics']['binary']['macro_auc']==.5
    rare=language['rare']
    assert rare['n']==1 and rare['metrics']['binary']['macro_auc'] is None
    assert rare['metrics']['binary']['evaluable_findings']==0
    assert rare['uncertainty_interval'] is None and rare['hospital_identity_verified'] is False
    assert len([r for r in rows if r['axis']=='scanner_proxy_group'])==3


def test_reserve_rejects_in_sample_teacher():
    from app.research.reserve import confident_joint
    y=np.tile(np.array([[0],[1]]),(1,12)); p=y*.8+.1
    with pytest.raises(ValueError,match='In-sample'):
        confident_joint(y,p,['a','b'],[0,1],{0:['a'],1:['a']})


def test_brier_bootstrap_reports_requested_metric():
    from app.research.metrics import paired_group_bootstrap
    y=np.tile(np.array([[0],[1],[0],[1]]),(1,12))
    result=paired_group_bootstrap(y,np.full(y.shape,.5),y*.8+.1,['a','b','c','d'],iterations=20,metric='brier')
    assert result['metric']=='brier' and result['delta_interval'][1]<0


def test_review_metrics_require_distinct_reviewers():
    from types import SimpleNamespace
    from datetime import datetime,timezone
    from app.research.review_audit import review_metrics
    label=SimpleNamespace(method='rules',version='test',findings={'ACL':{'state':'present'}})
    rows=[SimpleNamespace(study_id='a',finding='ACL',label_set_id='l',reviewer='Reviewer One',state='absent',created_at=datetime.now(timezone.utc)) for _ in range(2)]
    result=review_metrics(rows,{'l':label})
    assert result['agreed_cells']==0 and result['single_reviewer_cells']==1
    rows[-1].reviewer='Reviewer Two'
    result=review_metrics(rows,{'l':label})
    assert result['agreed_cells']==1 and result['strata'][0]['error_rate']==1
