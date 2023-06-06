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
from datetime import timedelta

import cv2
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
import torchvision
import torchvision.transforms as T
from torch import nn
from torch.utils.data import DataLoader, Dataset

BASE_SZ = int(sys.argv[1])
PAGE = int(sys.argv[2])
SZ = int(sys.argv[3])
BS = int(sys.argv[4])

EXPR = f"res18_imnet_{BASE_SZ}_{PAGE}_{SZ}"
TILE_DIR = f"./workspace/tiles/test/{BASE_SZ}/{PAGE}_{SZ}"


START_TIME = time.time()

RANDOM_STATE = 41


def fix_seed(seed):
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    torch.use_deterministic_algorithms(True, warn_only=True)


fix_seed(RANDOM_STATE)

DEVICE = torch.device("cuda")  # if torch.cuda.is_available() else torch.device('cpu')

MODEL_DIR = "models"
embedder_model_path = f"{MODEL_DIR}/resnet18-f37072fd.pth"

DATA_ROOT = "/code_execution/data/"
meta = pd.read_csv(f"{DATA_ROOT}/test_metadata.csv")


paths = glob.glob(f"{TILE_DIR}/*/img/*")
d_tiles = pd.DataFrame(dict(path=paths))
d_tiles["slide"] = d_tiles.path.str.split("/").str[-3]
d = d_tiles.path.str.split("/").str[-1]
d = d.str.split(".").str[0].str.split("_", expand=True).astype(int)
d.columns = ["tile_id", "yloc", "xloc"]
d_tiles = pd.concat([d_tiles, d], axis=1)

d = d_tiles.groupby("slide").agg(n_tiles=("tile_id", "count")).reset_index()
d_tiles = d_tiles.merge(d).sort_values(by=["slide", "tile_id"]).reset_index(drop=True)

assert d_tiles.slide.nunique() == len(meta)


def get_model():
    m = torchvision.models.resnet18()
    m.fc = nn.Identity()
    # state_dict = torch.load(embedder_model_path)['state_dict']
    # for key in list(state_dict.keys()):
    #     state_dict[key.replace('model.', '').replace('resnet.', '')] = state_dict.pop(key)

    state_dict = torch.load(embedder_model_path, map_location=DEVICE)  # ['state_dict']

    todel = []
    for i, (k, v) in enumerate(state_dict.items()):
        if "fc" in k:
            # print('deleting model key',k)
            todel.append(k)
    for k in todel:
        del state_dict[k]

    m.load_state_dict(state_dict, strict=True)
    return m


model = get_model()
model.to(DEVICE)
model.eval()

MEAN = 255 * np.array([0.485, 0.456, 0.406])
STD = 255 * np.array([0.229, 0.224, 0.225])


class VMelTileDataset(Dataset):
    def __init__(self, df, aug=None):
        self.df = df
        self.aug = aug

    def __len__(self):
        return len(self.df)

    def load_img(self, path):
        img = cv2.imread(path)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = img.astype(np.float32)

        img = img.transpose(2, 0, 1)
        img = (img - MEAN[:, None, None]) / STD[:, None, None]
        img = torch.from_numpy(img.astype(np.float32))

        # img=img.transpose(2,0,1).astype(np.float32)
        return img

    def __getitem__(self, ix):
        path = self.df.path.values[ix]

        img = self.load_img(path)
        return img


FEATURE_DIR = f"features/{EXPR}"
os.makedirs(FEATURE_DIR, exist_ok=True)
print(
    f"##### GENERATING EMBEDDINGS {EXPR}, device: {DEVICE} n_tiles: {len(d_tiles)} BS {BS} #####"
)
print(
    d_tiles.n_tiles.min(),
    int(d_tiles.n_tiles.mean()),
    d_tiles.n_tiles.max(),
    d_tiles.tile_id.max(),
)

# for slide,df in tqdm(d_tiles.groupby('slide')):
for slide, df in d_tiles.groupby("slide"):
    ds = VMelTileDataset(df)
    dl = DataLoader(ds, batch_size=BS, shuffle=False, num_workers=2)
    features = []
    with torch.no_grad():
        for x in dl:
            x = x.to(DEVICE)
            pred = model(x)
            pred = pred.cpu().numpy()
            features.append(pred)

    features = np.concatenate(features)
    np.savez_compressed(f"{FEATURE_DIR}/{slide}", features)
d_tiles.to_csv(f"{FEATURE_DIR}/tiles.csv", index=False)


elapsed = time.time() - START_TIME
print(f"##### DONE EMBEDDING {EXPR} TOTAL TIME: {timedelta(seconds=elapsed)} #####")
gc.collect()
