import argparse
import os
import random

import numpy as np
import pandas as pd
import pytorch_lightning as pl
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset

from sparseconvmilmodel import instantiate_sparseconvmil

parser = argparse.ArgumentParser()
parser.add_argument("--expr", help="experiment name e.g. expr_56_3_224", required=True)
parser.add_argument("--var", help="target variable: breslow or ulceration", required=True)
parser.add_argument("--seed", help="random seed", required=True, type=int)
parser.add_argument("--meta_csv_path", help="Path to the test metadata csv file", required=True)
args = parser.parse_args()
EXPR = args.expr  # expr_56_3_224,expr_40_2_320
VAR = args.var  # breslow,ulceration
RANDOM_STATE = args.seed
STAGE = "test"  #'train',test'
META_CSV_PATH = args.meta_csv_path

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


FEATURE_DIR = f"{DATA_ROOT}/embeddings/{EXPR}"
d_tiles = pd.read_csv(f"{FEATURE_DIR}/tiles.csv")
d_tiles = d_tiles.sort_values(by=["slide", "tile_id"])
assert d_tiles.slide.nunique() == len(meta)

d_tiles["filename"] = d_tiles.slide + ".tif"
df = (
    pd.merge(d_tiles, meta, on="filename")
    .sort_values(by=["slide", "tile_id"])
    .reset_index(drop=True)
)

df["path"] = FEATURE_DIR + "/" + df.slide + ".npz"


class VMelTilesDataset(Dataset):
    def __init__(self, df, aug=None):
        self.df = df
        self.aug = aug
        self.slides = df.slide.unique()

    def __len__(self):
        return len(self.slides)

    def __getitem__(self, ix):
        slide = self.slides[ix]
        d = self.df[self.df.slide == slide]
        path = d.path.values[0]
        x = np.load(path)["arr_0"]

        tile_locations = d[["yloc", "xloc"]].values.astype(np.float32)
        tile_locations = torch.from_numpy(tile_locations)

        x = x.astype(np.float32)
        x = torch.from_numpy(x)
        return x, tile_locations  # ,(y,y1,y2)


ds = VMelTilesDataset(df)
x, tile_locations = ds[0]
print("VMelTilesDataset", x.shape, tile_locations.shape, len(ds))

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

    def forward(self, x: torch.Tensor, locations: torch.Tensor):
        x = self.model(x, locations)
        return x


NUM_WORKERS = 2
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
print(f"GENERATING {EXPR} {VAR} predictions")

for FOLD in range(N_SPLITS):
    fix_seed(RANDOM_STATE)
    model_path = f"{MODEL_DIR}/f{FOLD}_{VAR}.pth"
    wts = torch.load(model_path)
    model.load_state_dict(wts)
    model.to(DEVICE)
    model.eval()

    df_test = df.copy()
    ds_test = VMelTilesDataset(df_test)
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

    df_test = df_test[["slide"]].copy()
    df_test[f"{VAR}_pred"] = y_pred
    dfs_preds.append(df_test)

df = pd.concat(dfs_preds).groupby("slide").mean().reset_index()
df.to_csv(f"{FACTORS_DIR}/{VAR}.csv", index=False)
