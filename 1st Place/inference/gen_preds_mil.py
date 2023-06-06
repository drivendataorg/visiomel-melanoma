import copy
import gc
import glob
import json
import math
import os
import pickle
import random
import re
import shutil
import sys
import time
import warnings

import cv2
import numpy as np
import pandas as pd
import PIL
import pytorch_lightning as pl
import torch
import torch.nn.functional as F
import torchvision
import torchvision.transforms as T
from pytorch_lightning import LightningModule, Trainer
from pytorch_lightning.callbacks import EarlyStopping, ModelCheckpoint
from torch import nn
from torch.utils.data import DataLoader, Dataset

EXPR = sys.argv[1]

print(f"GENERATING MIL PREDS {EXPR}")
MODEL_DIR = f"models/{EXPR}/final"
PRED_DIR = f"preds/{EXPR}"
os.makedirs(PRED_DIR, exist_ok=True)


def fix_seed(seed):
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    torch.use_deterministic_algorithms(True, warn_only=True)


DEVICE = torch.device("cpu")

N_SPLITS = 5

DATA_ROOT = "/code_execution/data/"
meta = pd.read_csv(f"{DATA_ROOT}/test_metadata.csv")
ss = pd.read_csv(f"{DATA_ROOT}/submission_format.csv")

d_breslow = pd.read_csv(f"factors/{EXPR}/breslow.csv")
d_ulceration = pd.read_csv(f"factors/{EXPR}/ulceration.csv")
d_factors = (
    pd.merge(d_breslow, d_ulceration, on="slide").sort_values(by="slide").reset_index(drop=True)
)

df = meta[["filename"]].copy()
df["slide"] = df.filename.str.split(".").str[0]
df = pd.merge(df, d_factors, on="slide")

print("factors", d_factors.shape, "data", df.shape)

feat_cols = ["age", "sex", "melanoma_history", "body_site"]  # ,'ulceration_pred','breslow_pred']

d = meta.copy()
d["age"] = d["age"].str.slice(1, 3).astype(np.float32) / 100
map_melanoma_history = {np.nan: -1, "NO": 0, "YES": 1}
d["melanoma_history"] = d["melanoma_history"].map(map_melanoma_history).astype(int)

map_body_site = {
    "head_neck": ["face", "neck", "scalp", "head/neck"],
    "upper_limb": ["upper limb/shoulder", "arm", "forearm", "hand"],
    "trunk": ["trunk", "trunc", "seat", "chest/thorax", "abdomen", "seat/buttocks"],
    "lower_limb": ["lower limb/hip", "thigh", "leg"],
    "extremities": ["hand/foot/nail", "nail", "finger", "foot", "sole", "toe"],
}

for i, (k, vals) in enumerate(map_body_site.items()):
    for c in vals:
        d.loc[d.body_site == c, "body_site"] = i
d["body_site"] = d.body_site.fillna(-1)

df = pd.merge(df, d[["filename"] + feat_cols], on="filename")

feat_cols = ["age", "sex", "melanoma_history", "body_site", "ulceration_pred", "breslow_pred"]
df[feat_cols] = df[feat_cols].astype(np.float32)


class VMelMetaDataset(Dataset):
    def __init__(self, df, aug=None):
        self.df = df
        self.aug = aug

    def __len__(self):
        return len(self.df)

    def __getitem__(self, ix):
        x = self.df[feat_cols].values[ix]
        return x


ds = VMelMetaDataset(df)


class Model(pl.LightningModule):
    def __init__(self):
        super().__init__()

        self.model = nn.Sequential(
            nn.Linear(len(feat_cols), 256),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(256, 32),
            nn.BatchNorm1d(32),
            nn.ReLU(),
            nn.Linear(32, 2),
        )

    def forward(self, x: torch.Tensor):
        x = self.model(x)
        return x


NUM_WORKERS = 1


for random_state in [41, 320, 888, 1948]:
    for fold in range(5):
        fix_seed(random_state)

        df_test = df.copy()

        ds_test = VMelMetaDataset(df_test)
        dl_test = DataLoader(ds_test, batch_size=32, shuffle=False, num_workers=NUM_WORKERS)

        model = Model()

        model_path = f"{MODEL_DIR}/r{random_state}_f{fold}.pth"
        wts = torch.load(model_path, map_location=torch.device("cpu"))
        model.load_state_dict(wts)
        model.to(DEVICE)
        model.eval()

        preds = []
        with torch.no_grad():
            for x in dl_test:
                x = x.to(DEVICE)
                pr = model(x).cpu()
                preds.append(pr)

        y_pred = torch.concatenate(preds)
        y_pred = F.softmax(y_pred, dim=1).numpy()[:, 1]

        df_test = df_test[["filename"]].copy()
        df_test["relapse"] = y_pred
        df_test.to_csv(f"{PRED_DIR}/pred_r{random_state}_f{fold}.csv", index=False)
