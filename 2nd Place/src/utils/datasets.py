# Standard libraries
import os

import torch
from PIL import Image

# Third-party libraries
from torch.utils.data import Dataset
from torchvision.transforms import Compose, Normalize, ToTensor

__all__ = ["BreslowUlcerationRelapseDataset", "ImageTabular"]


class BreslowUlcerationRelapseDataset(Dataset):
    breslow_mapping = {
        "<0.8": 0,
        "[0.8 : 1[": 1,
        "[1 : 2[": 2,
        "[2 : 4[": 3,
        ">=4": 4,
    }

    def __init__(self, dataframe, base_path, transforms):
        self.dataframe = dataframe
        self.base_path = base_path
        if transforms is None:
            self.transforms = Compose(
                [ToTensor(), Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])]
            )
        else:
            self.transforms = transforms

    def __getitem__(self, idx):
        sample = self.dataframe.iloc[idx]
        breslow = torch.tensor(self.breslow_mapping[sample.breslow])
        ulceration = torch.tensor(0.0) if sample.ulceration == "NO" else torch.tensor(1.0)
        relapse = torch.tensor(sample.relapse).float()
        slide_path = os.path.join(self.base_path, sample.filename)
        if not os.path.exists(slide_path):
            slide_path = os.path.join(self.base_path, sample.filename.replace(".tif", ".png"))

        slide = Image.open(slide_path)

        ####
        # slide_array = np.ndarray(slide, dtype=np.uint8)
        # slide = segment(slide_array)
        slide = self.transforms(slide)
        item = {"slide": slide, "ulceration": ulceration, "relapse": relapse, "breslow": breslow}
        return item

    def __len__(self):
        return len(self.dataframe)


class ImageTabular(Dataset):
    def __init__(self, dataframe, tabular_keys, aug_embedding=None, aug_tabular=None):
        self.dataframe = dataframe
        self.tabular_keys = tabular_keys
        self.aug_embedding = aug_embedding
        self.aug_tabular = aug_tabular

    def __getitem__(self, idx):
        sample = self.dataframe.iloc[idx]

        target_relapse = torch.tensor(sample.relapse, dtype=torch.float32)
        target_breslow = torch.tensor(sample.breslow, dtype=torch.float32)

        target_ulceration = torch.tensor(0.0) if sample.ulceration == "NO" else torch.tensor(1.0)

        tabular = torch.tensor(sample[self.tabular_keys], dtype=torch.float32)

        img_embedding = torch.tensor(sample.latent, dtype=torch.float32)
        if self.aug_embedding is not None:
            img_embedding = self.aug_embedding(img_embedding)
        if self.aug_tabular is not None:
            tabular = self.aug_tabular(tabular)
        return img_embedding, tabular, target_relapse, target_breslow, target_ulceration

    def __len__(self):
        return len(self.dataframe)
