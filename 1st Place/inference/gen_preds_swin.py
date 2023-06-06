import copy
import gc
import glob
import os
import random
import re
import sys
import time
import warnings

import cv2
import numpy as np
import pandas as pd
import pytorch_lightning as pl
import timm
import torch
import torch.nn.functional as F
from torch import nn
from torch.utils.data import DataLoader, Dataset

ARCH = sys.argv[1]  # swin_large_patch4_window12_384, swinv2_base_window12to16_192to256_22kft1k
SZ = int(sys.argv[2])  # 384,256
RANDOM_STATE = int(sys.argv[3])  # 888,256

MODEL_DIR = f"models/swin{SZ}"
PRED_DIR = "preds/swin"
os.makedirs(PRED_DIR, exist_ok=True)

N_SPLITS = 5


def fix_seed(seed):
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


fix_seed(RANDOM_STATE)
DEVICE = torch.device("cuda")


DATA_ROOT = "/code_execution/data"
ss = pd.read_csv(f"{DATA_ROOT}/submission_format.csv")
meta = pd.read_csv(f"{DATA_ROOT}/test_metadata.csv")

# IMG_DIR = f'{DATA_ROOT}/images/{SZ}'

IMG_DIR = f"images/{SZ}"
meta["path"] = IMG_DIR + "/" + meta.filename + ".png"

MEAN = 255 * np.array([0.5, 0.5, 0.5])
STD = 255 * np.array([0.5, 0.5, 0.5])


class VMelDataset(Dataset):
    def __init__(self, df):
        self.df = df

    def __len__(self):
        return len(self.df)

    def __getitem__(self, ix):
        path = self.df.path.values[ix]
        img = cv2.imread(path)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        ##normalize
        img = img.astype(np.float32)
        img = img.transpose(2, 0, 1)
        img = (img - MEAN[:, None, None]) / STD[:, None, None]
        img = torch.from_numpy(img.astype(np.float32))

        return img


bs = 32
NUM_WORKERS = 2
ds_test = VMelDataset(meta)
dl_test = DataLoader(ds_test, batch_size=bs, shuffle=False, num_workers=NUM_WORKERS)


class Model(pl.LightningModule):
    def __init__(self, model):
        super().__init__()
        self.model = model

    # def forward(self, image: torch.Tensor):
    def forward(self, x: torch.Tensor):
        x = self.model(x)
        return x


def get_preds(model):
    preds = []
    with torch.no_grad():
        for x in dl_test:
            x = x.to(DEVICE)
            pr = model(x).cpu()
            preds.append(pr)
    preds = torch.concatenate(preds)
    preds = F.softmax(preds, dim=1).numpy()[:, 1]
    return preds


fix_seed(RANDOM_STATE)

print(f"GENERATING p5 swin{SZ} {ARCH} PREDICTIONS")

m = timm.create_model(ARCH, pretrained=False, num_classes=2, in_chans=3)
model = Model(m)

all_preds = []
for fold in range(N_SPLITS):
    path = f"{MODEL_DIR}/{ARCH}_f{fold}_p1.pth"
    wts = torch.load(path)
    model.load_state_dict(wts, strict=True)
    model.to(DEVICE)
    model.eval()

    preds = get_preds(model)
    all_preds.append(preds)

preds = np.mean(all_preds, 0)

sub = meta[["filename"]].copy()
sub["relapse"] = preds
sub.to_csv(f"{PRED_DIR}/pred_swin{SZ}.csv", index=False)
