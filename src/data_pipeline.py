import os
import glob
from monai.transforms import (Compose, LoadImaged, EnsureChannelFirstd, Spacingd, 
                              Orientationd, CropForegroundd, NormalizeIntensityd, 
                              RandCropByPosNegLabeld, RandFlipd, RandRotate90d, ToTensord,
                              MapLabelValued)
from monai.data import CacheDataset, DataLoader, Dataset



# Adicionamos o caminho correto como padrão aqui
def get_dataloaders(data_path='../data/raw/BraTS2020_TrainingData/MICCAI_BraTS2020_TrainingData'):
    
    patient_folders = sorted(glob.glob(os.path.join(data_path, "BraTS20_Training_*")))
    
    # Trava de Segurança: Se não achar nada, ele avisa o erro de caminho antes do PyTorch travar
    if len(patient_folders) == 0:
        raise ValueError(f"Nenhum paciente encontrado no caminho: {data_path}. Verifique se você está rodando o notebook na pasta correta!")
    
    data_dicts = []
    for folder in patient_folders:
        patient_id = os.path.basename(folder)
        img_list = [
            os.path.join(folder, f"{patient_id}_flair.nii"),
            os.path.join(folder, f"{patient_id}_t1.nii"),
            os.path.join(folder, f"{patient_id}_t1ce.nii"),
            os.path.join(folder, f"{patient_id}_t2.nii")
        ]
        label_path = os.path.join(folder, f"{patient_id}_seg.nii")
        if all(os.path.exists(f) for f in img_list) and os.path.exists(label_path):
            data_dicts.append({"image": img_list, "label": label_path})
            
    print(f"Total de pacientes carregados na tubulação: {len(data_dicts)}")
            
    train_files, val_files = data_dicts[:-40], data_dicts[-40:]
    keys = ["image", "label"]

    train_transforms = Compose([
        LoadImaged(keys=keys), EnsureChannelFirstd(keys=keys),
        
        # O CONSERTO ESTÁ AQUI: Transforma a classe 4 na classe 3
        MapLabelValued(keys=["label"], orig_labels=[4], target_labels=[3]),
        
        Spacingd(keys=keys, pixdim=(1.0, 1.0, 1.0), mode=("bilinear", "nearest")),
        Orientationd(keys=keys, axcodes="RAS"), CropForegroundd(keys=keys, source_key="image"),
        NormalizeIntensityd(keys="image", nonzero=True, channel_wise=True),
        RandCropByPosNegLabeld(keys=keys, label_key="label", spatial_size=(96, 96, 96), pos=1, neg=1, num_samples=4, image_key="image", image_threshold=0),
        RandFlipd(keys=keys, spatial_axis=[0], prob=0.10),
        ToTensord(keys=keys),
    ])

    val_transforms = Compose([
        LoadImaged(keys=keys), EnsureChannelFirstd(keys=keys),
        
        # O CONSERTO ESTÁ AQUI TAMBÉM: Transforma a classe 4 na classe 3
        MapLabelValued(keys=["label"], orig_labels=[4], target_labels=[3]),
        
        Spacingd(keys=keys, pixdim=(1.0, 1.0, 1.0), mode=("bilinear", "nearest")),
        Orientationd(keys=keys, axcodes="RAS"), CropForegroundd(keys=keys, source_key="image"),
        NormalizeIntensityd(keys="image", nonzero=True, channel_wise=True),
        ToTensord(keys=keys),
    ])

    # Utilizando o Dataset normal (em vez de CacheDataset) para o seu teste inicial
    # Isso evita travar seus 16GB de RAM na primeira tentativa de leitura
    train_ds = Dataset(data=train_files, transform=train_transforms)
    val_ds = Dataset(data=val_files, transform=val_transforms)

    # Batch size de 1 para preservar os 16GB de VRAM
    train_loader = DataLoader(train_ds, batch_size=1, shuffle=True, num_workers=4)
    val_loader = DataLoader(val_ds, batch_size=1, num_workers=4)

    return train_loader, val_loader