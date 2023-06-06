import argparse
import os
import random

import numpy as np
import pandas as pd
import pytorch_lightning as pl
import torch
import torch.nn.functional as F
from torch import nn
from torch.utils.data import DataLoader, Dataset

parser = argparse.ArgumentParser()
parser.add_argument(
    "--expr", help="experiment name e.g. expr_56_3_224,expr_40_2_320", required=True
)
parser.add_argument("--meta_csv_path", help="Path to the test metadata csv file", required=True)
args = parser.parse_args()
EXPR = args.expr  # expr_56_3_224,expr_40_2_320
STAGE = "test"  #'train',test'
META_CSV_PATH = args.meta_csv_path

DATA_ROOT = f"./workspace/data/{STAGE}"
FACTORS_DIR = f"{DATA_ROOT}/factors/{EXPR}"

MODEL_DIR = f"./workspace/models/{EXPR}/mlp"

PRED_DIR = f"./workspace/preds/{STAGE}/{EXPR}"
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

meta = pd.read_csv(META_CSV_PATH)

d_breslow = pd.read_csv(f"{FACTORS_DIR}/breslow.csv")[["slide", "breslow_pred"]]
d_ulceration = pd.read_csv(f"{FACTORS_DIR}/ulceration.csv")[["slide", "ulceration_pred"]]
d_factors = pd.merge(d_breslow, d_ulceration).sort_values(by="slide").reset_index(drop=True)

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


NUM_WORKERS = 2

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
