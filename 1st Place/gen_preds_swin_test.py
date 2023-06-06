import argparse
import os
import random

import cv2
import numpy as np
import pandas as pd
import pytorch_lightning as pl
import timm
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset

parser = argparse.ArgumentParser()
parser.add_argument("--expr", help="experiment name e.g. swin256, swin384", required=True)
parser.add_argument("--meta_csv_path", help="Path to the test metadata csv file", required=True)
args = parser.parse_args()
EXPR = args.expr  # swin256, swin384
STAGE = "test"  #'train',test'
META_CSV_PATH = args.meta_csv_path


if EXPR == "swin256":
    arch = "swinv2_base_window12to16_192to256_22kft1k"
    SZ = 256
    RANDOM_STATE = 256

elif EXPR == "swin384":
    arch = "swin_large_patch4_window12_384"
    SZ = 384
    RANDOM_STATE = 888

DATA_ROOT = f"./workspace/data/{STAGE}"
MODEL_DIR = f"workspace/models/{EXPR}"

PRED_DIR = f"./workspace/preds/{STAGE}/"
os.makedirs(PRED_DIR, exist_ok=True)


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

N_SPLITS = 5

meta = pd.read_csv(META_CSV_PATH)
IMG_DIR = f"{DATA_ROOT}/resized/{SZ}"
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


NUM_WORKERS = 2
batch_size = 32
ds_test = VMelDataset(meta)
dl_test = DataLoader(ds_test, batch_size=batch_size, shuffle=False, num_workers=NUM_WORKERS)


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

print(f"GENERATING p5 {EXPR} {arch} PREDICTIONS")

m = timm.create_model(arch, pretrained=False, num_classes=2, in_chans=3)
model = Model(m)

all_preds = []
for fold in range(N_SPLITS):
    path = f"{MODEL_DIR}/{arch}_f{fold}_p1.pth"
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
