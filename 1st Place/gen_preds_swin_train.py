import argparse
import glob
import os
import random

import cv2
import numpy as np
import pandas as pd
import pytorch_lightning as pl
import sklearn.metrics as skm
import timm
import torch
import torch.nn.functional as F
from sklearn.model_selection import StratifiedKFold
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm

parser = argparse.ArgumentParser()
parser.add_argument("--expr", help="experiment name e.g. swin256, swin384", required=True)
parser.add_argument(
    "--meta_csv_path", help="Path to the train or test metadata csv file", required=True
)
parser.add_argument(
    "--label_csv_path", help="Path to the train or test metadata csv file", required=True
)
args = parser.parse_args()
EXPR = args.expr  # swin256, swin384
STAGE = "train"  #'train',test'
META_CSV_PATH = args.meta_csv_path
LABEL_CSV_PATH = args.label_csv_path


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
labels = pd.read_csv(LABEL_CSV_PATH)
labels["slide"] = labels.filename.str.split(".").str[0]

IMG_DIR = f"{DATA_ROOT}/resized/{SZ}"
paths = glob.glob(f"{IMG_DIR}/*")
print(len(paths))

df = labels.copy()
df["sid"] = df.filename.str.split(".").str[0]
df["path"] = IMG_DIR + "/" + df.filename + ".png"

feat_cols = ["ulceration", "breslow"]
d = meta.copy()
map_breslow = {np.nan: -1, "<0.8": 0.4, "[0.8 : 1[": 0.9, "[1 : 2[": 1.5, "[2 : 4[": 3, ">=4": 4}
map_ulceraton = {np.nan: -1, "NO": 0, "YES": 1}
map_melanoma_history = {np.nan: -1, "NO": 0, "YES": 1}

d["breslow"] = d["breslow"].map(map_breslow).astype(np.float32)
d["ulceration"] = d["ulceration"].map(map_ulceraton).astype(int)


df = pd.merge(df, d[["filename"] + feat_cols])


def split_data(df):
    kf = StratifiedKFold(n_splits=N_SPLITS, shuffle=True, random_state=RANDOM_STATE)
    # TODO groupkf by region
    df["fold"] = -1

    for fold_id, (train_index, test_index) in enumerate(kf.split(df, df.relapse)):
        df.loc[test_index, "fold"] = fold_id
    return df


def get_splits(df, fold):
    df_trn = df[df.fold != fold].copy()
    df_val = df[df.fold == fold].copy()
    return df_trn, df_val


df = split_data(df)

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


def get_dls(fold, drop_nan=False, bs=batch_size):
    df_trn, df_val = get_splits(df, fold)
    print("getting data fold", fold, len(df_trn), len(df_val))
    if drop_nan:
        df_trn = df_trn[df_trn.breslow != -1]
        df_val = df_val[df_val.breslow != -1]
        print("after drop bad samples", len(df_trn), len(df_val))

    ds_trn = VMelDataset(df_trn)
    dl_trn = DataLoader(ds_trn, batch_size=bs, shuffle=True, num_workers=NUM_WORKERS)
    ds_val = VMelDataset(df_val)
    dl_val = DataLoader(ds_val, batch_size=bs, shuffle=False, num_workers=NUM_WORKERS)
    # x,y = ds_val[0]
    return df_trn, df_val, dl_trn, dl_val


class Model(pl.LightningModule):
    def __init__(self, model):
        super().__init__()
        self.model = model

    # def forward(self, image: torch.Tensor):
    def forward(self, x: torch.Tensor):
        x = self.model(x)
        return x


m = timm.create_model(arch, pretrained=False, num_classes=2, in_chans=3)
model = Model(m)

pred_dfs = []
for FOLD in range(5):
    fix_seed(RANDOM_STATE)
    df_test = df[df.fold == FOLD].copy()
    ds_test = VMelDataset(df_test)
    dl_test = DataLoader(ds_test, batch_size=batch_size, shuffle=False, num_workers=NUM_WORKERS)
    weights_path = f"{MODEL_DIR}/{arch}_f{FOLD}_p1.pth"
    wts = torch.load(weights_path, map_location=torch.device("cpu"))
    model.load_state_dict(wts, strict=True)
    model.to(DEVICE)
    model.eval()
    print("Evaluating", FOLD)
    preds = []
    gts = []
    with torch.no_grad():
        for x in tqdm(dl_test):
            x = x.to(DEVICE)
            pr = model(x).cpu()
            preds.append(pr)

    y_pred = torch.concatenate(preds)
    y_pred = F.softmax(y_pred, dim=1).numpy()[:, 1]

    df_test = df_test[["filename", "relapse"]].copy()
    df_test["relapse_pred"] = y_pred
    d = df_test.copy()
    score = skm.log_loss(d.relapse, d.relapse_pred)
    print(f"{FOLD}: {score}")

    pred_dfs.append(df_test)

d = pd.concat(pred_dfs)
score = skm.log_loss(d.relapse, d.relapse_pred)
print(f"{EXPR} score: {score}")

d.to_csv(f"{PRED_DIR}/pred_swin{SZ}.csv", index=False)
