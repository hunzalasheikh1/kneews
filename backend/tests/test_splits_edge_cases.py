"""Edge cases for leakage-safe fold construction (module M1)."""
import pytest
from app.research.splits import report_hash,connected_groups,make_folds,assert_no_leakage


def rows(n,site=None,report=None):
    return [dict(id=f's{i}',site_group=site(i) if site else f'site{i}',report_hash=report(i) if report else None) for i in range(n)]


def test_report_hash_ignores_case_and_whitespace_and_empty_reports():
    assert report_hash('ACL  intact.\n') == report_hash('acl intact.')
    assert report_hash('   ') is None


def test_missing_report_hash_never_joins_studies():
    groups=connected_groups(rows(4))
    assert len(set(groups))==4


def test_missing_site_metadata_forms_one_conservative_group():
    data=[dict(id='a'),dict(id='b',site_group=''),dict(id='c',site_group='scanner-x')]
    groups=connected_groups(data)
    assert groups[0]==groups[1] and groups[2]!=groups[0]


def test_every_study_gets_a_valid_fold_and_leakage_group():
    folds=make_folds(rows(25,site=lambda i:f'site{i%10}'),5)
    assert {r['fold'] for r in folds}=={0,1,2,3,4}
    assert all(r['leakage_group'] for r in folds)
    by_site={}
    for r in folds: by_site.setdefault(r['site_group'],set()).add(r['fold'])
    assert all(len(f)==1 for f in by_site.values())


def test_duplicate_study_ids_are_rejected():
    with pytest.raises(ValueError,match='unique'):
        make_folds([dict(id='a',site_group='x'),dict(id='a',site_group='y')]*3,2)


def test_leakage_check_detects_a_site_split_across_folds():
    folds=make_folds(rows(10),5)
    folds[0]['site_group']=folds[1]['site_group']='shared'
    folds[0]['fold'],folds[1]['fold']=0,1
    with pytest.raises(ValueError,match='site_group leakage'):
        assert_no_leakage(folds)


def test_fold_assignment_is_deterministic():
    data=rows(30,site=lambda i:f'site{i%12}')
    assert [r['fold'] for r in make_folds(data,5)]==[r['fold'] for r in make_folds(data,5)]
