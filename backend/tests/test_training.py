import numpy as np
import pytest
import torch
from app.research.training import targets,SlotHead,fit_head,predict

def test_masking_ignores_unknown_and_uncertain():
    s=np.tile([0,1,2,3],(2,3)); y,w=targets('masked',s)
    assert np.all(w[s>=2]==0) and np.all(y[s==0]==1)

def test_llm_arm_cannot_fabricate_probabilities():
    with pytest.raises(ValueError,match='real local LLM'):
        targets('llm_probability',np.zeros((2,12)))

def test_missing_slots_do_not_affect_predictions():
    torch.manual_seed(2026); model=SlotHead(8).eval()
    x=torch.randn(2,6,8); mask=torch.tensor([[True,False,False,False,False,False]]*2)
    y=model(x,mask); x[:,1:]=1e6
    assert torch.allclose(y,model(x,mask))

def test_matched_head_training_is_reproducible():
    rng=np.random.default_rng(1); x=rng.normal(size=(8,6,8)).astype('float32'); mask=np.ones((8,6),bool)
    y=np.ones((8,12),np.float32); w=np.ones_like(y)
    a,_=fit_head(x,mask,y,w,epochs=2); b,_=fit_head(x,mask,y,w,epochs=2)
    assert np.allclose(predict(a,x,mask),predict(b,x,mask))


def test_no_supervision_cannot_produce_fake_training_success():
    x=np.zeros((2,6,8),np.float32);mask=np.ones((2,6),bool);y=np.zeros((2,12),np.float32)
    with pytest.raises(ValueError,match='positive weight'):
        fit_head(x,mask,y,np.zeros_like(y),epochs=1)


def test_gold_free_reserve_predictions_do_not_use_expert_values():
    from app.research.training import crossfit
    rng=np.random.default_rng(5)
    records=[];states=[];gold=[]
    for fold in range(3):
        for j in range(5):
            records.append(dict(id=f'{fold}-{j}',fold=fold,site_group=f's{fold}',leakage_group=f's{fold}',report_hash=f'{fold}-{j}'))
            states.append([j%2]*12)
            gold.append([1.0]*12 if j==4 else [float('nan')]*12)
    x=rng.normal(size=(15,6,8)).astype('float32');mask=np.ones((15,6),bool);states=np.array(states);gold=np.array(gold)
    a=crossfit(x,mask,states,gold,records,epochs=1,arms=('reserve',))
    b=crossfit(x,mask,states,1-gold,records,epochs=1,arms=('reserve',))
    assert np.allclose(a['predictions']['reserve'],b['predictions']['reserve'])
    assert a['estimators']==[]
    for outer in a['reserve_estimators']:
        held={r['id'] for r in records if r['fold']==outer['fold']}
        assert not held.intersection(outer['teacher_prediction_ids'])
        assert all(not held.intersection(ids) for ids in outer['teacher_training_ids'].values())


def test_pooled_held_out_predictions_cannot_depend_on_held_out_expert_answers():
    from app.research.training import crossfit
    rng = np.random.default_rng(17)
    records, states, gold = [], [], []
    for fold in range(3):
        for j in range(6):
            records.append(dict(id=f'{fold}-{j}', fold=fold, site_group=f's{fold}',
                                leakage_group=f's{fold}', report_hash=f'{fold}-{j}'))
            states.append([(j + finding) % 4 for finding in range(12)])
            gold.append([(fold + j + finding) % 2 for finding in range(12)]
                        if j >= 4 else [float('nan')] * 12)
    x = rng.normal(size=(18, 6, 8)).astype('float32')
    masks = np.ones((18, 6), bool)
    gold = np.asarray(gold)
    changed = gold.copy()
    held = np.asarray([r['fold'] == 0 for r in records])
    changed[held] = 1 - changed[held]
    original = crossfit(x, masks, np.asarray(states), gold, records, epochs=1, arms=('pooled',))
    perturbed = crossfit(x, masks, np.asarray(states), changed, records, epochs=1, arms=('pooled',))
    selected = [i for i, study_id in enumerate(original['study_ids']) if study_id.startswith('0-')]
    np.testing.assert_array_equal(np.asarray(original['predictions']['pooled'])[selected],
                                  np.asarray(perturbed['predictions']['pooled'])[selected])
    assert original['estimators'][0] == perturbed['estimators'][0]
    assert not set(original['estimators'][0]['fit_ids']) & {r['id'] for r in records if r['fold'] == 0}
    # The perturbation must affect other folds' estimators: otherwise the test
    # could pass because the pooled method ignored all expert labels.
    assert original['estimators'][1]['table'] != perturbed['estimators'][1]['table']


def test_abstention_explains_missing_coverage_and_seed_disagreement():
    from app.research.abstention import summarize_seed_predictions
    x=np.tile([.95,.95,.05,.95,.95,.95,.95,.95,.95,.95,.95,.95],(3,1))
    x[:,1]=[.1,.5,.9]
    result=summarize_seed_predictions(x)
    assert not result['findings'][0]['review_required']
    assert result['findings'][1]['review_required']
    assert not result['findings'][1]['seed_range_is_confidence_interval']
    assert all(r['review_required'] for r in summarize_seed_predictions(x,available_planes=2)['findings'])
