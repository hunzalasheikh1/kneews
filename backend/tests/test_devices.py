import pytest
import torch
from app.research.devices import resolve_device


def test_explicit_mps_never_falls_back_to_cpu(monkeypatch):
    monkeypatch.setattr(torch.backends.mps,'is_available',lambda:False)
    with pytest.raises(RuntimeError,match='CPU fallback is disabled'):
        resolve_device('mps')


def test_gpu_request_rejects_operator_cpu_fallback(monkeypatch):
    monkeypatch.setattr(torch.backends.mps,'is_available',lambda:True)
    monkeypatch.setenv('PYTORCH_ENABLE_MPS_FALLBACK','1')
    with pytest.raises(RuntimeError,match='unsupported operators must fail visibly'):
        resolve_device('mps')


def test_auto_selects_apple_gpu_when_available(monkeypatch):
    monkeypatch.setattr(torch.cuda,'is_available',lambda:False)
    monkeypatch.setattr(torch.backends.mps,'is_available',lambda:True)
    monkeypatch.delenv('PYTORCH_ENABLE_MPS_FALLBACK',raising=False)
    assert resolve_device('auto').type=='mps'
    assert resolve_device('cpu').type=='cpu'
