"""Geometry-aware DICOM and NIfTI loading; physical crops for 2.5D inputs."""
from pathlib import Path
import numpy as np
import pydicom
import nibabel as nib
from scipy.ndimage import zoom
from PIL import Image
import io


def read_dicom_series(paths):
    slices=[pydicom.dcmread(str(p)) for p in paths]
    if not slices:
        raise ValueError('No DICOM slices')
    first=slices[0]
    required=['ImageOrientationPatient','ImagePositionPatient','PixelSpacing']
    if any(not all(hasattr(s, tag) for tag in required) for s in slices):
        raise ValueError('Missing spatial geometry; unsafe to infer slice order from filenames')
    orientation=np.asarray(first.ImageOrientationPatient,float)
    normal=np.cross(orientation[:3],orientation[3:]); normal/=np.linalg.norm(normal)
    if any(not np.allclose(s.ImageOrientationPatient,orientation,atol=1e-3) for s in slices):
        raise ValueError('Mixed orientations within series')
    slices.sort(key=lambda s:float(np.dot(np.asarray(s.ImagePositionPatient,float),normal)))
    positions=np.asarray([np.dot(np.asarray(s.ImagePositionPatient,float),normal) for s in slices])
    if len(slices)>1:
        gaps=np.diff(positions)
        if np.any(gaps<=1e-4): raise ValueError('Duplicate spatial slice positions')
        if not np.allclose(gaps,np.median(gaps),rtol=.1,atol=.1):
            raise ValueError('Irregular slice spacing requires explicit resampling')
        dz=float(np.median(gaps))
    else:
        dz=float(getattr(first,'SliceThickness',1))
    arrays=[]
    for ds in slices:
        pixels=ds.pixel_array.astype(np.float32)
        if pixels.ndim!=2: raise ValueError('Only single-frame monochrome slices supported in this loader')
        pixels=pixels*float(getattr(ds,'RescaleSlope',1))+float(getattr(ds,'RescaleIntercept',0))
        if getattr(ds,'PhotometricInterpretation','')=='MONOCHROME1': pixels=pixels.max()+pixels.min()-pixels
        arrays.append(pixels)
    volume=np.stack(arrays,axis=2)
    affine=np.eye(4)
    affine[:3,0]=orientation[3:]*float(first.PixelSpacing[0])
    affine[:3,1]=orientation[:3]*float(first.PixelSpacing[1])
    affine[:3,2]=normal*dz
    affine[:3,3]=np.asarray(slices[0].ImagePositionPatient,float)
    affine=np.diag([-1,-1,1,1])@affine  # DICOM LPS to NIfTI RAS
    image=nib.as_closest_canonical(nib.Nifti1Image(volume,affine))
    metadata=dict(manufacturer=str(getattr(first,'Manufacturer','unknown')),
                  model=str(getattr(first,'ManufacturerModelName','unknown')),
                  field_strength=str(getattr(first,'MagneticFieldStrength','unknown')),
                  laterality=str(getattr(first,'ImageLaterality',getattr(first,'Laterality','unknown'))),
                  description=str(getattr(first,'SeriesDescription','')),
                  repetition_time=str(getattr(first,'RepetitionTime','')),
                  echo_time=str(getattr(first,'EchoTime','')),
                  institution=str(getattr(first,'InstitutionName','')),
                  station=str(getattr(first,'StationName','')),
                  slices=len(slices),shape=list(image.shape),spacing=list(image.header.get_zooms()),
                  transfer_syntax=str(first.file_meta.TransferSyntaxUID))
    return image,metadata


def load_nifti(path):
    image=nib.as_closest_canonical(nib.load(str(path)))
    if len(image.shape)!=3: raise ValueError('Expected a 3D NIfTI volume')
    if min(image.shape)<1 or np.prod(image.shape)>256_000_000: raise ValueError('NIfTI volume exceeds the supported voxel limit')
    if not np.isfinite(image.affine).all(): raise ValueError('Invalid affine')
    return image


def normalize(volume):
    data=np.asarray(volume,np.float32)
    if not np.isfinite(data).all(): raise ValueError('Non-finite image voxels')
    lo,hi=np.percentile(data,[1,99])
    if hi<=lo: return np.zeros_like(data)
    return np.clip((data-lo)/(hi-lo),0,1)


def physical_crop(slice_image, spacing, field_mm=130, size=224):
    if field_mm<=0 or size<=0 or min(spacing)<=0: raise ValueError('Invalid crop geometry')
    crop_size=np.maximum(1,np.rint(field_mm/np.asarray(spacing)).astype(int))
    canvas=np.zeros(tuple(crop_size),dtype=np.float32)
    src=[]; dst=[]
    for existing,wanted in zip(slice_image.shape,crop_size):
        take=min(existing,wanted); src.append(slice((existing-take)//2,(existing-take)//2+take))
        dst.append(slice((wanted-take)//2,(wanted-take)//2+take))
    canvas[tuple(dst)]=slice_image[tuple(src)]
    return zoom(canvas,(size/crop_size[0],size/crop_size[1]),order=1,prefilter=False)


def sample_25d(image, plane, laterality='unknown', samples=16, size=224, field_mm=130):
    axes={'Sagittal':0,'Coronal':1,'Axial':2}
    if plane not in axes: raise ValueError('Unknown anatomical plane')
    data=normalize(image.get_fdata(dtype=np.float32)); spacing=image.header.get_zooms()
    # Canonical RAS x axis points right. Mirror right knees to a left-knee convention.
    if laterality.upper() in ('R','RIGHT'): data=np.flip(data,axis=0)
    axis=axes[plane]; depth=data.shape[axis]
    indices=np.rint(np.linspace(.1*(depth-1),.9*(depth-1),samples)).astype(int)
    pixel_spacing=[spacing[i] for i in range(3) if i!=axis]
    # np.take on a canonical reoriented (strided) volume may copy the entire
    # volume for every slice. Materialize once, then index inexpensive views.
    frames=np.ascontiguousarray(np.moveaxis(data,axis,0))
    out=[];cropped={}
    for center in indices:
        triplet=[]
        for offset in (-1,0,1):
            index=int(np.clip(center+offset,0,depth-1))
            if index not in cropped:cropped[index]=physical_crop(frames[index],pixel_spacing,field_mm,size)
            triplet.append(cropped[index])
        out.append(np.stack(triplet))
    return np.stack(out).astype(np.float32)


def slice_png(image,index,axis=2):
    if axis not in (0,1,2) or not 0<=index<image.shape[axis]: raise ValueError('Slice out of range')
    frame=np.rot90(np.take(normalize(image.get_fdata(dtype=np.float32)),index,axis=axis))
    spacing=[image.header.get_zooms()[i] for i in range(3) if i!=axis]
    # rot90 swaps dimensions: width corresponds to the first remaining voxel axis.
    width_mm=frame.shape[1]*spacing[0];height_mm=frame.shape[0]*spacing[1]
    scale=min(768/max(width_mm,height_mm),1/min(spacing))
    pixels=Image.fromarray((frame*255).astype(np.uint8)).resize(
        (max(1,round(width_mm*scale)),max(1,round(height_mm*scale))),Image.Resampling.BILINEAR)
    result=io.BytesIO(); pixels.save(result,format='PNG')
    return result.getvalue()
