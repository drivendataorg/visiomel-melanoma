# Standard libraries
import os
from PIL import Image
import numpy as np
import pandas as pd

# Third-party libraries
from torch.utils.data import Dataset
import torch
from torchvision.transforms import Compose, ToTensor, Normalize
from ast import literal_eval

__all__ = ["UlcerationDataset", "BreslowUlcerationRelapseDataset", "BreslowDataset", "RelapseDataset", "ImageTabular"]


def breslow_harmonization(metadata_df, tiff_info_df):
    """
        Usage : 
            metadataDF = pd.read_csv("/Users/Happpyyyyyyy/Documents/VisioMel/data/train_metadata.csv")
            tiffDF = pd.read_csv("/Users/Happpyyyyyyy/Documents/VisioMel/data/tiff_info_data.csv")
            new = breslow_harmonization(metadataDF, tiffDF)
    """
    breslow_mapping = {
        "<0.8":0.2,
        "[0.8 : 1[":0.9,
        "[1 : 2[":1.5,
        "[2 : 4[":3,
        ">=4":4,
        np.nan:0.1}
    new_breslow = []
    for index, row in tiff_info_df.iterrows():
        (img_name,_,_,init_size_width,init_size_height) = row
        meta_row = metadata_df[metadata_df.filename == img_name]
        breslow = breslow_mapping[meta_row.breslow.to_numpy()[0]] # in mm
        resolution = meta_row.resolution.to_numpy()[0] # microns/pixel

        breslow_pixels = resolution/(breslow*1000)
        new_breslow.append(breslow_pixels)
    metadata_df["new_breslow"] = new_breslow
    return metadata_df


class BodyDataset(Dataset):
    def __init__(self, dataframe, labels, base_path, transforms):
        self.dataframe = dataframe
        self.labels = labels
        self.base_path = base_path
        if transforms is None:
            self.transforms = Compose([
            ToTensor(),
            Normalize(mean=[0.485, 0.456, 0.406],std=[0.229, 0.224, 0.225])
      ])
        else:
            self.transforms = transforms

    def __getitem__(self, idx):
        sample = self.dataframe.iloc[idx]
        target = torch.tensor(self.labels[sample.body_site])
        slide_path = os.path.join(self.base_path, sample.filename)
        slide = Image.open(slide_path)
        slide = self.transforms(slide)
        return slide, target
    
    def __len__(self):
        return len(self.dataframe)
    

class UlcerationDataset(Dataset):
    def __init__(self, dataframe, base_path, transforms):
        self.dataframe = dataframe
        self.base_path = base_path
        if transforms is None:
            self.transforms = Compose([
            ToTensor(),
            Normalize(mean=[0.485, 0.456, 0.406],std=[0.229, 0.224, 0.225])
      ])
        else:
            self.transforms = transforms

    def __getitem__(self, idx):
        sample = self.dataframe.iloc[idx]
        target = torch.tensor(0.) if sample.ulceration == "NO" else torch.tensor(1.)
        slide_path = os.path.join(self.base_path, sample.filename)
        slide = Image.open(slide_path)
        slide = self.transforms(slide)
        return slide, target
    
    def __len__(self):
        return len(self.dataframe)
    
class RelapseDataset(Dataset):
    def __init__(self, dataframe, base_path, transforms):
        self.dataframe = dataframe
        self.base_path = base_path
        if transforms is None:
            self.transforms = Compose([
            ToTensor(),
            Normalize(mean=[0.485, 0.456, 0.406],std=[0.229, 0.224, 0.225])
      ])
        else:
            self.transforms = transforms

    def __getitem__(self, idx):
        sample = self.dataframe.iloc[idx]
        target = torch.tensor(sample.relapse).float()
        slide_path = os.path.join(self.base_path, sample.filename)
        slide = Image.open(slide_path)
        slide = self.transforms(slide)
        return slide, target
    
    def __len__(self):
        return len(self.dataframe)

class BreslowUlcerationRelapseDataset(Dataset):
    breslow_mapping = {
        "<0.8":0,
        "[0.8 : 1[":1,
        "[1 : 2[":2,
        "[2 : 4[":3,
        ">=4":4     
    }
    
    def __init__(self, dataframe, base_path, transforms):
        self.dataframe = dataframe
        self.base_path = base_path
        if transforms is None:
            self.transforms = Compose([
            ToTensor(),
            Normalize(mean=[0.485, 0.456, 0.406],std=[0.229, 0.224, 0.225])
      ])
        else:
            self.transforms = transforms
  
    def __getitem__(self, idx):
        sample = self.dataframe.iloc[idx]
        breslow = torch.tensor(self.breslow_mapping[sample.breslow])
        ulceration = torch.tensor(0.) if sample.ulceration == "NO" else torch.tensor(1.)
        relapse = torch.tensor(sample.relapse).float()
        slide_path = os.path.join(self.base_path, sample.filename)
        slide = Image.open(slide_path)
        slide = self.transforms(slide)
        item = {"slide": slide, "ulceration": ulceration, "relapse": relapse, "breslow": breslow}
        return item

    def __len__(self):
        return len(self.dataframe)


class BreslowDataset(Dataset):
    mapping = {
        "<0.8":0,
        "[0.8 : 1[":1,
        "[1 : 2[":2,
        "[2 : 4[":3,
        ">=4":4     
    }
    
    def __init__(self, dataframe, base_path, transforms):
        self.dataframe = dataframe
        self.base_path = base_path
        if transforms is None:
            self.transforms = Compose([
            ToTensor(),
            Normalize(mean=[0.485, 0.456, 0.406],std=[0.229, 0.224, 0.225])
      ])
        else:
            self.transforms = transforms
  
    def __getitem__(self, idx):
        sample = self.dataframe.iloc[idx]
        target = torch.tensor(self.mapping[sample.breslow])
        slide_path = os.path.join(self.base_path, sample.filename)
        slide = Image.open(slide_path)
        slide = self.transforms(slide)
        return slide, target

    def __len__(self):
        return len(self.dataframe)
    

class ImageTabular(Dataset):
    one_hot_age = {0: [1., 0., 0., 0., 0.],
              1: [0., 1., 0., 0., 0.],
              2: [0., 0., 1., 0., 0.],
              3: [0., 0., 0., 1., 0.],
              4: [0., 0., 0., 0., 1.]}
    
    one_hot_sex = {0: [1., 0.],
              1: [0., 1.]}

    one_hot_melanoma = {0: [1., 0., 0.],
                   1: [0., 1., 0.],
                   2: [0., 0., 1.]}
    
    key = [i for i in range(-1, 17)]
    value = np.eye(18).astype(float)
    one_hot_body = {k: list(v) for k, v in zip(key, value)}


    def __init__(self, dataframe, tabular_keys ,aug_embedding=None):
        self.dataframe = dataframe
        self.tabular_keys = tabular_keys
        self.aug_embedding = aug_embedding

    def __getitem__(self, idx):
        sample = self.dataframe.iloc[idx]

        target_relapse = torch.tensor(sample.relapse, dtype=torch.float32)
        target_breslow = torch.tensor(sample.breslow, dtype=torch.float32)
        
        target_ulceration = torch.tensor(0.) if sample.ulceration == "NO" else torch.tensor(1.)
       
        tabular = torch.tensor(sample[self.tabular_keys], dtype=torch.float32)
       
        img_embedding = torch.tensor(sample.latent, dtype=torch.float32)
        if self.aug_embedding is not None :
            img_embedding = self.aug_embedding(img_embedding)
        return img_embedding, tabular, target_relapse, target_breslow, target_ulceration
    
    def __len__(self):
        return len(self.dataframe)
    


class ImagePatchTabular(Dataset):
    one_hot_age = {0: [1., 0., 0., 0., 0.],
              1: [0., 1., 0., 0., 0.],
              2: [0., 0., 1., 0., 0.],
              3: [0., 0., 0., 1., 0.],
              4: [0., 0., 0., 0., 1.]}
    
    one_hot_sex = {0: [1., 0.],
              1: [0., 1.]}

    one_hot_melanoma = {0: [1., 0., 0.],
                   1: [0., 1., 0.],
                   2: [0., 0., 1.]}
    
    key = [i for i in range(-1, 17)]
    value = np.eye(18).astype(float)
    one_hot_body = {k: list(v) for k, v in zip(key, value)}


    def __init__(self, dataframe, path_patches, tabular_keys ,aug_embedding=None):
        self.dataframe = dataframe
        self.path_patches = path_patches
        self.tabular_keys = tabular_keys
        self.aug_embedding = aug_embedding

    def __getitem__(self, idx):
        sample = self.dataframe.iloc[idx]
        filename = sample.filename

        target_relapse = torch.tensor(sample.relapse, dtype=torch.float32)
        target_breslow = torch.tensor(sample.breslow, dtype=torch.float32)
        target_ulceration = torch.tensor(0.) if sample.ulceration == "NO" else torch.tensor(1.)
       
        tabular = torch.tensor(sample[self.tabular_keys], dtype=torch.float32)
       
        img_embedding = torch.tensor(sample.latent, dtype=torch.float32)

        patch_embedding = torch.tensor(pd.read_csv(os.path.join(self.path_patches, f'{filename}.csv')).values[:, 2:].astype(float)).float()

        if self.aug_embedding is not None :
            img_embedding = self.aug_embedding(img_embedding)
            patch_embedding = self.aug_embedding(patch_embedding)
        return img_embedding, patch_embedding, tabular, target_relapse, target_breslow, target_ulceration
    
    def __len__(self):
        return len(self.dataframe)