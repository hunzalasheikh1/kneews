from types import SimpleNamespace
from app.research.audit_packet import select_sample


def test_language_sample_is_order_independent_and_includes_small_strata():
    studies=[SimpleNamespace(id=f'{language}-{i}',language=language)
             for language,n in [('en',30),('nl',2),('tr',20)] for i in range(n)]
    selected=select_sample(studies,per_language=5)
    reversed_selection=select_sample(list(reversed(studies)),per_language=5)
    assert [s.id for s in selected]==[s.id for s in reversed_selection]
    assert len(selected)==12 and sum(s.language=='nl' for s in selected)==2
    assert {s.id for s in selected}!={s.id for s in select_sample(studies,per_language=5,seed=2027)}
