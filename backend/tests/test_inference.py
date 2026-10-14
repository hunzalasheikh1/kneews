from types import SimpleNamespace
import numpy as np
import nibabel as nib
import pytest
import torch
from app.research.training import SlotHead
from app.research.inference import ensemble,validate_external
from app.research.features import FrozenEncoder


def test_external_ensemble_averages_folds_before_seed_ranges(tmp_path):
    for seed,value in [(1,0.),(2,2.)]:
        for fold in [0,1]:
            model=SlotHead(8)
            for parameter in model.parameters():parameter.data.zero_()
            model.classifier[-1].bias.data.fill_(value+fold)
            torch.save(model.state_dict(),tmp_path/f'pooled-seed{seed}-fold{fold}.pt')
    config=dict(arms=['pooled'],seeds=[1,2],folds=[dict(fold=i,is_gold=True) for i in [0,1]])
    result=ensemble(tmp_path,config,np.zeros((6,8),np.float32),np.array([1,0,0,0,0,0],bool),'pooled')
    sigmoid=lambda x:1/(1+np.exp(-x))
    expected=[np.mean([sigmoid(s),sigmoid(s+1)]) for s in [0,2]]
    row=result['findings'][0]
    assert row['probability']==pytest.approx(np.mean(expected))
    assert row['seed_min']==pytest.approx(expected[0]) and row['seed_max']==pytest.approx(expected[1])
    assert row['review_required'] and 'Incomplete anatomical plane coverage' in row['reasons']
    assert len(result['checkpoints'])==4 and all(len(x['checkpoint_sha256'])==64 for x in result['checkpoints'])
    assert result['prediction_source']=='external_fold_ensemble'
    (tmp_path/'pooled-seed2-fold1.pt').unlink()
    with pytest.raises(FileNotFoundError):ensemble(tmp_path,config,np.zeros((6,8),np.float32),np.ones(6,bool),'pooled')


def test_external_inference_requires_explicit_protocol_and_matching_knee():
    series=SimpleNamespace(plane='Sagittal',metadata_json={})
    study=SimpleNamespace(split='external',source='upload',series=[series])
    with pytest.raises(ValueError,match='Confirm acquisition'):validate_external(study)
    series.metadata_json=dict(protocol_reviewed=True,fluid_sensitive=True,laterality='L')
    validate_external(study)
    study.series.append(SimpleNamespace(plane='Coronal',metadata_json=dict(protocol_reviewed=True,fluid_sensitive=False,laterality='R')))
    with pytest.raises(ValueError,match='same knee'):validate_external(study)
    study.split='train'
    with pytest.raises(ValueError,match='external uploads'):validate_external(study)


def test_nifti_uses_only_its_confirmed_acquisition_slot(tmp_path):
    nib.save(nib.Nifti1Image(np.arange(9*9*9,dtype=np.float32).reshape(9,9,9),np.eye(4)),tmp_path/'volume.nii')
    encoder=FrozenEncoder.__new__(FrozenEncoder)
    encoder.config=dict(samples=2,size=8,field_mm=8);encoder.signature='fixture'
    encoder.model=SimpleNamespace(config=SimpleNamespace(hidden_size=1))
    encoder.encode=lambda triplets:np.array([triplets.mean(),triplets.max()],np.float32)
    series=SimpleNamespace(id='nifti-test',plane='Coronal',format='nifti',relative_path='volume.nii',
                           metadata_json=dict(fluid_sensitive=True,laterality='R',protocol_reviewed=True))
    features,mask,metadata=encoder.study([series],tmp_path)
    assert mask.tolist()==[False,False,False,True,False,False]
    assert not features[~mask].any() and features[mask].any()
    assert metadata['sources'][0]['laterality']=='R' and not metadata['errors']
