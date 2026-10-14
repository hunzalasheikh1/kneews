import numpy as np
from app.research.final_fit import all_study_targets


def test_full_cohort_targets_use_gold_where_available_and_pool_other_cases():
    states=np.tile(np.arange(4), (5,3))
    gold=np.full((5,12),np.nan)
    gold[0]=np.arange(12)%2;gold[1]=1-gold[0]
    expert=np.array([True,True,False,False,False])
    targets,table=all_study_targets(states,gold,[str(i) for i in range(5)],expert,8)
    assert targets.shape==(5,12) and targets.dtype==np.float32
    np.testing.assert_array_equal(targets[expert],gold[expert])
    assert np.isfinite(targets).all() and ((targets[~expert]>0)&(targets[~expert]<1)).all()
    assert len(table)==48
