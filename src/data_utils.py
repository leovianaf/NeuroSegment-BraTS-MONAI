import os
import glob
from monai.transforms import (
  Compose, LoadImaged, EnsureChannelFirstd,
  Orientationd, Spacingd, NormalizeIntensityd, EnsureTyped,
  DivisiblePadd, CenterSpatialCropd
)


def get_val_transforms():
  return Compose([
    LoadImaged(keys=["image", "label"]),
    EnsureChannelFirstd(keys=["image", "label"]),
    Orientationd(keys=["image", "label"], axcodes="RAS"),
    Spacingd(keys=["image", "label"], pixdim=(1.0, 1.0, 1.0), mode=("bilinear", "nearest")),
    NormalizeIntensityd(keys="image", nonzero=True, channel_wise=True),
    CenterSpatialCropd(keys=["image", "label"], roi_size=(128, 128, 128)),
    # CenterSpatialCropd(keys=["image", "label"], roi_size=(64, 64, 64)),
    DivisiblePadd(keys=["image", "label"], k=32),
    EnsureTyped(keys=["image", "label"]),
  ])

def get_sample_data(data_dir, patient_id):
  """Busca os caminhos das 4 modalidades de um paciente específico"""
  path = os.path.join(data_dir, patient_id)
  return {
    "image": [
        os.path.join(path, f"{patient_id}_flair.nii"),
        os.path.join(path, f"{patient_id}_t1.nii"),
        os.path.join(path, f"{patient_id}_t1ce.nii"),
        os.path.join(path, f"{patient_id}_t2.nii"),
    ],
    "label": os.path.join(path, f"{patient_id}_seg.nii")
  }