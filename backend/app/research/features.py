"""Frozen DINOv2 2.5D embeddings cached by encoder/preprocessing version."""
import hashlib
import json
import os
from pathlib import Path
import numpy as np
import torch
from transformers import AutoModel
from .imaging import read_dicom_series,sample_25d,load_nifti

PLANES=('Sagittal','Coronal','Axial')

class FrozenEncoder:
    def __init__(self,model_dir,device='auto',samples=16,size=224,field_mm=130,precision='float32'):
        self.model_dir=Path(model_dir)
        if device=='auto': device='cuda' if torch.cuda.is_available() else ('mps' if torch.backends.mps.is_available() else 'cpu')
        self.device=torch.device(device)
        self.model=AutoModel.from_pretrained(model_dir,local_files_only=True).eval().to(self.device)
        self.model.requires_grad_(False)
        if precision not in ('float32','float16'):raise ValueError('Unsupported encoder precision')
        self.dtype=torch.float16 if precision=='float16' else torch.float32
        self.model.to(dtype=self.dtype)
        self.config=dict(samples=samples,size=size,field_mm=field_mm,normalization='percentile_1_99_then_imagenet',
                         pooling='cls_and_mean_patch_then_mean_slices',version='0.1',
                         encoder_sha256=hashlib.file_digest((self.model_dir/'model.safetensors').open('rb'),'sha256').hexdigest())
        if precision!='float32':self.config['precision']=precision
        self.signature=hashlib.sha256(json.dumps(self.config,sort_keys=True).encode()).hexdigest()

    @torch.inference_mode()
    def encode(self,triplets,batch_size=16):
        embeddings=[]
        mean=torch.tensor([.485,.456,.406],device=self.device,dtype=self.dtype)[None,:,None,None]
        std=torch.tensor([.229,.224,.225],device=self.device,dtype=self.dtype)[None,:,None,None]
        for start in range(0,len(triplets),batch_size):
            x=torch.from_numpy(triplets[start:start+batch_size]).to(self.device,dtype=self.dtype)
            hidden=self.model(pixel_values=(x-mean)/std).last_hidden_state
            embeddings.append(torch.cat([hidden[:,0],hidden[:,1:].mean(1)],dim=1).float().cpu().numpy())
        return np.concatenate(embeddings).mean(0)

    def study(self,series,data_root):
        features=np.zeros((6,self.model.config.hidden_size*2),np.float32); mask=np.zeros(6,bool); sources=[]; errors=[]
        for plane_index,plane in enumerate(PLANES):
            for sensitive in (0,1):
                slot=plane_index*2+sensitive
                candidates=sorted([s for s in series if s.plane==plane and int(s.metadata_json.get('fluid_sensitive',s.metadata_json.get('fluid_sensitive_csv',0)))==sensitive],key=lambda s:s.id)
                for candidate in candidates:
                    try:
                        if candidate.format=='nifti':
                            image=load_nifti(Path(data_root)/candidate.relative_path)
                            meta={'laterality':candidate.metadata_json.get('laterality','unknown')}
                        else:
                            files=sorted((Path(data_root)/candidate.relative_path).glob('*.dcm'))
                            image,meta=read_dicom_series(files)
                            if candidate.metadata_json.get('protocol_reviewed'):
                                meta['laterality']=candidate.metadata_json['laterality']
                        triplets=sample_25d(image,plane,meta['laterality'],samples=self.config['samples'],
                                            size=self.config['size'],field_mm=self.config['field_mm'])
                        features[slot]=self.encode(triplets); mask[slot]=True
                        sources.append(dict(slot=slot,series_id=candidate.id,laterality=meta['laterality'],
                                            laterality_known=meta['laterality'].upper() in ('R','RIGHT','L','LEFT')))
                        break
                    except (ValueError,RuntimeError,OSError) as exc:
                        errors.append(dict(series_id=candidate.id,error=str(exc)))
        if not mask.any(): raise ValueError('No valid series available')
        return features,mask,dict(sources=sources,errors=errors,encoder=self.config,signature=self.signature)


def cache_study(path,features,mask,metadata):
    path=Path(path); path.parent.mkdir(parents=True,exist_ok=True)
    temporary=path.with_suffix('.partial')
    with temporary.open('wb') as f:
        np.savez_compressed(f,features=features,mask=mask,metadata=json.dumps(metadata))
    os.replace(temporary,path)
