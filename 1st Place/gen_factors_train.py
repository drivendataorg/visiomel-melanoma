import argparse
import os
import random

import numpy as np
import pandas as pd
import pytorch_lightning as pl
import sklearn.metrics as skm
import torch
import torch.nn.functional as F
from sklearn.model_selection import StratifiedKFold
from torch.utils.data import DataLoader, Dataset

from sparseconvmilmodel import instantiate_sparseconvmil

parser = argparse.ArgumentParser()
parser.add_argument("--expr", help="experiment name e.g. expr_56_3_224", required=True)
parser.add_argument("--var", help="target variable: breslow or ulceration", required=True)
parser.add_argument("--seed", help="random seed", required=True, type=int)
parser.add_argument(
    "--meta_csv_path", help="Path to the train or test metadata csv file", required=True
)
parser.add_argument("--label_csv_path", help="Path to the train labels csv file", required=True)

args = parser.parse_args()
EXPR = args.expr  # expr_56_3_224,expr_40_2_320
VAR = args.var  # breslow,ulceration
RANDOM_STATE = args.seed
STAGE = "train"  #'train',test'
META_CSV_PATH = args.meta_csv_path
LABEL_CSV_PATH = args.label_csv_path

BASE_SZ, PAGE, SZ = [int(x) for x in EXPR.split("_")[1:]]
DATA_ROOT = f"./workspace/data/{STAGE}"
MODEL_DIR = f"./workspace/models/{EXPR}"

FACTORS_DIR = f"{DATA_ROOT}/factors/{EXPR}"
os.makedirs(FACTORS_DIR, exist_ok=True)

if VAR == "breslow":
    N_CLASSES = 1
elif VAR == "ulceration":
    N_CLASSES = 2


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

DEVICE = torch.device("cpu")

N_SPLITS = 5
meta = pd.read_csv(META_CSV_PATH)
labels = pd.read_csv(LABEL_CSV_PATH)
labels["slide"] = labels.filename.str.split(".").str[0]

FEATURE_DIR = f"{DATA_ROOT}/embeddings/{EXPR}"
d_tiles = pd.read_csv(f"{FEATURE_DIR}/tiles.csv")
d_tiles = d_tiles.sort_values(by=["slide", "tile_id"])
assert d_tiles.slide.nunique() == len(meta)


def split_data(df):
    kf = StratifiedKFold(n_splits=N_SPLITS, shuffle=True, random_state=RANDOM_STATE)
    df["fold"] = -1

    for fold_id, (train_index, test_index) in enumerate(kf.split(df, df.relapse)):
        df.loc[test_index, "fold"] = fold_id
    return df


def get_splits(df, fold):
    df_trn = df[df.fold != fold].copy()
    df_val = df[df.fold == fold].copy()
    return df_trn, df_val


labels = split_data(labels)

df = labels.copy()

feat_cols = ["ulceration", "breslow"]
d = meta.copy()
map_breslow = {np.nan: -1, "<0.8": 0.4, "[0.8 : 1[": 0.9, "[1 : 2[": 1.5, "[2 : 4[": 3, ">=4": 4}
map_ulceraton = {np.nan: -1, "NO": 0, "YES": 1}
d["breslow"] = d["breslow"].map(map_breslow).astype(np.float32)
d["ulceration"] = d["ulceration"].map(map_ulceraton).astype(int)


df = pd.merge(df, d[["filename"] + feat_cols])

df = pd.merge(df, d_tiles).sort_values(by=["slide", "tile_id"]).reset_index(drop=True)
df["path"] = FEATURE_DIR + "/" + df.slide + ".npz"


class VMelDataset(Dataset):
    def __init__(self, df, aug=None):
        self.df = df
        self.aug = aug
        self.slides = df.slide.unique()

    def __len__(self):
        return len(self.slides)

    def __getitem__(self, ix):
        slide = self.slides[ix]
        d = self.df[self.df.slide == slide].set_index("tile_id")
        path = d.path.values[0]
        x = np.load(path)["arr_0"]

        tile_locations = d[["yloc", "xloc"]].values.astype(np.float32)
        tile_locations = torch.from_numpy(tile_locations)
        # y = d.relapse.values[0]
        # y1 = d.breslow.values[0]
        # y2 = d.ulceration.values[0]

        x = x.astype(np.float32)
        x = torch.from_numpy(x)
        return x, tile_locations  # ,(y,y1,y2)


ds = VMelDataset(df)

params = {}
params["embedding_size"] = 512
params["sparse_conv_n_channels_conv1"] = 128  # 32
params["sparse_conv_n_channels_conv2"] = 128  # 32
params["sparse_map_downsample"] = SZ
params["wsi_embedding_classifier_n_inner_neurons"] = 32
params["n_classes"] = N_CLASSES
params["batch_size"] = 1


class Model(pl.LightningModule):
    def __init__(self, model):
        super().__init__()
        self.model = model

    # def forward(self, image: torch.Tensor):
    def forward(self, x: torch.Tensor, locations: torch.Tensor):
        x = self.model(x, locations)
        return x


NUM_WORKERS = 1

sparseconvmil_model = instantiate_sparseconvmil(
    params["embedding_size"],
    params["sparse_conv_n_channels_conv1"],
    params["sparse_conv_n_channels_conv2"],
    3,
    3,
    params["sparse_map_downsample"],
    params["wsi_embedding_classifier_n_inner_neurons"],
    n_classes=params["n_classes"],
)

model = Model(model=sparseconvmil_model)
dfs_preds = []

for FOLD in range(N_SPLITS):
    fix_seed(RANDOM_STATE)
    model_path = f"{MODEL_DIR}/f{FOLD}_{VAR}.pth"
    wts = torch.load(model_path)
    model.load_state_dict(wts)
    model.to(DEVICE)
    model.eval()

    df_test = df[df.fold == FOLD].copy()
    ds_test = VMelDataset(df_test)
    dl_test = DataLoader(ds_test, batch_size=1, shuffle=False, num_workers=NUM_WORKERS)
    df_test = df_test.drop_duplicates(subset=["slide"]).copy()

    preds = []
    with torch.no_grad():
        for x, locations in dl_test:
            x = x.to(DEVICE)
            locations = locations.to(DEVICE)
            pr = model(x, locations).cpu()
            preds.append(pr)

    y_pred = torch.concatenate(preds)
    if VAR == "breslow":
        y_pred = y_pred.flatten()
    else:
        y_pred = F.softmax(y_pred, dim=1).numpy()[:, 1]

    df_test = df_test[["slide", VAR]].copy()
    df_test[f"{VAR}_pred"] = y_pred
    dfs_preds.append(df_test)

    # d=df_test.copy();score=skm.mean_squared_error(d[VAR],d[f'{VAR}_pred'],squared=False);print(score)
    # d=d[d[VAR]!=-1];score=skm.mean_squared_error(d[VAR],d[f'{VAR}_pred'],squared=False);print(score)


if VAR == "breslow":
    df_preds_breslow = pd.concat(dfs_preds).sort_values(by="slide").reset_index(drop=True)
    df_preds_breslow.to_csv(f"{FACTORS_DIR}/breslow.csv", index=False)

    d = df_preds_breslow.copy()
    sc0 = skm.mean_squared_error(d[VAR], d[f"{VAR}_pred"], squared=False)
    d = d[d[VAR] != -1]
    sc1 = skm.mean_squared_error(d[VAR], d[f"{VAR}_pred"], squared=False)

else:
    df_preds_ulceration = pd.concat(dfs_preds).sort_values(by="slide").reset_index(drop=True)
    df_preds_ulceration.to_csv(f"{FACTORS_DIR}/ulceration.csv", index=False)

    d = df_preds_ulceration.copy()
    sc0 = skm.log_loss(d[VAR], d[f"{VAR}_pred"], labels=[0, 1])
    d = d[d[VAR] != -1]
    sc1 = skm.log_loss(d[VAR], d[f"{VAR}_pred"], labels=[0, 1])

print(EXPR, VAR, "avg score: ", sc1)
