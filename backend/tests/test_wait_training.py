from app.research.wait_training import await_ready,inputs_complete


def state(**updates):
    return dict(dict(ready_for_training=True,total=3,valid_features=3,valid_labels=3,
                     missing_features=[],invalid_features=[],missing_labels=[],invalid_labels=[]),**updates)


def test_waiter_rechecks_inputs_and_launches_only_once_after_all_checks():
    snapshots=iter([state(valid_features=2,missing_features=['x']),state(invalid_labels=[{'study_id':'x'}]),state()])
    events=[];waits=[];launched=[]
    result=await_ready(lambda:next(snapshots),lambda:launched.append(True) or 0,events.append,waits.append)
    assert result==0 and launched==[True] and waits==[60,60]
    assert [e['status'] for e in events]==['waiting','waiting','waiting','launching']
    assert not inputs_complete(state(total=0,valid_features=0,valid_labels=0))
