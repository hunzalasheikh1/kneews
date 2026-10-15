import pytest
from app.research.language_audit import audit_language_codes


def test_latin_and_confusable_codes_are_flagged():
    rows={r['code']:r for r in audit_language_codes({'en':1672,'hr':401,'la':53,'bs':5})}
    assert not rows['en']['needs_review']
    assert 'not a clinical reporting language' in rows['la']['reasons']
    assert 'confusable with hr' in rows['bs']['reasons'] and 'confusable with bs' in rows['hr']['reasons']
    assert 'fewer than 20 reports' in rows['bs']['reasons']


def test_rows_are_sorted_by_count_and_unknown_is_flagged():
    rows=audit_language_codes({'de':10,'en':50,'unknown':30})
    assert [r['code'] for r in rows]==['en','unknown','de']
    assert rows[1]['reasons']==['language not detected']


def test_unexpected_code_and_invalid_counts():
    assert audit_language_codes({'ja':100})[0]['reasons']==['unexpected for this dataset']
    with pytest.raises(ValueError):
        audit_language_codes({'en':-1})
