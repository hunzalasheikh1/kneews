import numpy as np
import nibabel as nib
from app.research.imaging import sample_25d, physical_crop,normalize

def test_constant_scan_normalizes_without_nan():
    assert not normalize(np.ones((3,3,3))).any()

def test_physical_crop_pads_and_preserves_shape():
    crop=physical_crop(np.ones((10,20)),[1,1],field_mm=30,size=60)
    assert crop.shape==(60,60) and crop[0,0]==0 and crop[30,30]>.99

def test_25d_has_true_adjacent_channels():
    volume=np.broadcast_to(np.arange(20,dtype=np.float32)[None,None,:],(20,20,20)).copy()
    image=nib.Nifti1Image(volume,np.eye(4))
    x=sample_25d(image,'Axial',samples=4,size=14,field_mm=20)
    assert x.shape==(4,3,14,14)
    assert np.all(x[:,0].mean((1,2))<x[:,1].mean((1,2)))
    assert np.all(x[:,1].mean((1,2))<x[:,2].mean((1,2)))


def test_viewer_respects_anisotropic_voxel_spacing():
    import io
    from PIL import Image
    from app.research.imaging import slice_png
    image=nib.Nifti1Image(np.zeros((10,20,30),np.float32),np.diag([4,1,1,1]))
    rendered=Image.open(io.BytesIO(slice_png(image,0,axis=2)))
    assert rendered.width/rendered.height==2  # 10*4 mm across by 20*1 mm high
