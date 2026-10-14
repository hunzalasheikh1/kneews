"""Resolve and record the training backend; explicit GPU requests never fall back."""
import os
import platform
import torch


def resolve_device(requested='auto'):
    if requested=='auto':
        requested='cuda' if torch.cuda.is_available() else ('mps' if torch.backends.mps.is_available() else 'cpu')
    device=torch.device(requested)
    if device.type not in ('cpu','mps','cuda'):
        raise ValueError('Training supports cpu, mps or cuda')
    if device.type=='mps':
        if not torch.backends.mps.is_available():
            raise RuntimeError('Apple GPU (MPS) was requested but is unavailable in this process; CPU fallback is disabled')
        if os.environ.get('PYTORCH_ENABLE_MPS_FALLBACK')=='1':
            raise RuntimeError('Disable PYTORCH_ENABLE_MPS_FALLBACK before GPU training; unsupported operators must fail visibly')
        device=torch.device('mps')
    if device.type=='cuda' and not torch.cuda.is_available():
        raise RuntimeError('CUDA was requested but is unavailable; CPU fallback is disabled')
    return device


def device_provenance(requested,device):
    return dict(requested=requested,resolved=str(device),framework='pytorch',torch_version=torch.__version__,
                platform=platform.platform(),machine=platform.machine(),
                mps_cpu_fallback_enabled=os.environ.get('PYTORCH_ENABLE_MPS_FALLBACK')=='1',
                cpu_role='Input preparation, orchestration and statistical evaluation; model fitting uses the resolved device')


def verify_training_device(requested):
    """Exercise the actual head, backward pass and prediction on synthetic inputs."""
    import json
    import time
    import numpy as np
    from .training import fit_head,predict
    from ..config import settings
    device=resolve_device(requested)
    rng=np.random.default_rng(2026)
    x=rng.standard_normal((3500,6,768),dtype=np.float32)
    mask=np.ones((3500,6),bool)
    y=rng.random((3500,12),dtype=np.float32)
    start=time.perf_counter()
    model,history=fit_head(x,mask,y,np.ones_like(y),epochs=2,device=device)
    actual=next(model.parameters()).device
    if actual.type!=device.type:
        raise RuntimeError(f'Model landed on {actual}, expected {device}')
    predictions=predict(model,x[:64],mask[:64],device=device)
    if not np.isfinite(predictions).all() or not np.isfinite(history).all():
        raise RuntimeError('Device verification produced nonfinite output')
    memory=None
    if device.type=='mps':
        torch.mps.synchronize()
        memory=torch.mps.current_allocated_memory()
        if memory<=0:raise RuntimeError('No MPS tensor memory was allocated')
    result=dict(**device_provenance(requested,device),model_parameter_device=str(actual),
                synthetic_timing_only=True,studies=3500,epochs=2,seconds=time.perf_counter()-start,
                finite_loss_and_predictions=True,mps_tensor_bytes=memory)
    root=settings.artifact_root/'benchmarks';root.mkdir(parents=True,exist_ok=True)
    (root/f'training-device-{device.type}.json').write_text(json.dumps(result,indent=2))
    return result
