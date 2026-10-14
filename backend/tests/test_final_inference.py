from types import SimpleNamespace
import pytest
from app.research.inference import validate_final_target


def study(split, series=('s',)):
    return SimpleNamespace(split=split, source='rsna', series=list(series))


def test_final_inference_refuses_training_studies():
    with pytest.raises(ValueError, match='in-sample'):
        validate_final_target(study('train'))


def test_final_inference_allows_held_out_test_studies():
    validate_final_target(study('test'))


def test_final_inference_rejects_missing_and_empty():
    with pytest.raises(ValueError):
        validate_final_target(None)
    with pytest.raises(ValueError):
        validate_final_target(study('test', series=()))
