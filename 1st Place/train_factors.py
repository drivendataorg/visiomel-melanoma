import argparse
import os
import random

import numpy as np
import pandas as pd
import pytorch_lightning as pl
import sklearn.metrics as skm
import torch
import torch.nn.functional as F
from pytorch_lightning import Trainer
from pytorch_lightning.callbacks import EarlyStopping, ModelCheckpoint
from sklearn.model_selection import StratifiedKFold
from torch.utils.data import DataLoader, Dataset

from sparseconvmilmodel import instantiate_sparseconvmil

N_SPLITS = 5

parser = argparse.ArgumentParser()
parser.add_argument("--expr", help="experiment name e.g. expr_56_3_224", required=True)
parser.add_argument("--var", help="target variable: breslow or ulceration", required=True)
parser.add_argument("--seed", help="random seed", required=True, type=int)
parser.add_argument("--meta_csv_path", help="Path to the train metadata csv file", required=True)
parser.add_argument("--label_csv_path", help="Path to the train labels csv file", required=False)
args = parser.parse_args()
EXPR = args.expr  # expr_56_3_224,expr_40_2_320
VAR = args.var  # breslow,ulceration
RANDOM_STATE = args.seed
STAGE = "train"  #'train',test'
META_CSV_PATH = args.meta_csv_path
LABEL_CSV_PATH = args.label_csv_path

BASE_SZ, PAGE, SZ = [int(x) for x in EXPR.split("_")[1:]]

DATA_ROOT = f"./workspace/data/{STAGE}"
MODEL_DIR = f"workspace/models/{EXPR}"
os.makedirs(MODEL_DIR, exist_ok=True)

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

meta = pd.read_csv(META_CSV_PATH)
labels = pd.read_csv(LABEL_CSV_PATH)
labels["slide"] = labels.filename.str.split(".").str[0]

FEATURE_DIR = f"{DATA_ROOT}/embeddings/{EXPR}"
d_tiles = pd.read_csv(f"{FEATURE_DIR}/tiles.csv")
# d_blur = pd.read_csv(f'{FEATURE_DIR}/blur.csv')
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
        # N=256
        d = self.df[self.df.slide == slide].set_index("tile_id")
        path = d.path.values[0]
        x = np.load(path)["arr_0"]

        tile_locations = d[["yloc", "xloc"]].values.astype(np.float32)
        tile_locations = torch.from_numpy(tile_locations)
        y = d.relapse.values[0]
        y1 = d.breslow.values[0]
        y2 = d.ulceration.values[0]

        x = x.astype(np.float32)
        x = torch.from_numpy(x)
        return x, tile_locations, (y, y1, y2)  # ,slide


ds = VMelDataset(df)
x, tile_locations, y = ds[0]
print(f"VMelDataset x: {x.shape} locations {tile_locations.shape}")

params = {}

params["embedding_size"] = 512
params["sparse_conv_n_channels_conv1"] = 128  # 32
params["sparse_conv_n_channels_conv2"] = 128  # 32
params["sparse_map_downsample"] = SZ
params["wsi_embedding_classifier_n_inner_neurons"] = 32
params["n_classes"] = N_CLASSES
params["batch_size"] = 1


NUM_WORKERS = 2


def get_dls(fold, drop_nan=False, drop_nan_var=None, bs=32, oversample=False):
    df_trn, df_val = get_splits(df, fold)

    print("getting data fold", fold, len(df_trn), len(df_val))
    if oversample:
        n0 = len(df_trn[df_trn.relapse == 0])
        n1 = len(df_trn[df_trn.relapse == 1])
        d = df_trn[df_trn.relapse == 1].sample(n0 - n1, replace=True)
        df_trn = df_trn.append(d).reset_index(drop=True)
        print("after oversample", len(df_trn), len(df_val))

    if drop_nan:
        df_trn = df_trn[df_trn[drop_nan_var] != -1]
        df_val = df_val[df_val[drop_nan_var] != -1]
        print("after drop bad samples", len(df_trn), len(df_val))

    ds_trn = VMelDataset(df_trn)
    dl_trn = DataLoader(ds_trn, batch_size=bs, shuffle=True, num_workers=NUM_WORKERS)
    ds_val = VMelDataset(df_val)
    dl_val = DataLoader(ds_val, batch_size=bs, shuffle=False, num_workers=NUM_WORKERS)
    # x,y = ds_val[0]
    return df_trn, df_val, dl_trn, dl_val


class Model(pl.LightningModule):
    def __init__(self, model, lr, val_gts, target):
        super().__init__()
        self.model = model
        self.lr = lr
        self.val_preds = []
        self.val_gts = val_gts
        self.target = target

    def forward(self, x: torch.Tensor, locations: torch.Tensor):
        x = self.model(x, locations)
        return x

    def clf_loss(self, logits, targets):
        l = F.cross_entropy(logits, targets)
        # l=FocalLoss()(logits,targets)
        return l

    def reg_loss(self, preds, targets):
        preds = preds.flatten()
        l = F.mse_loss(preds, targets)
        return l

    def training_step(self, batch, batch_idx):
        self.model.train()
        torch.set_grad_enabled(True)

        x, locations, y = batch
        pred = self.model(x, locations)

        if self.target == "breslow":
            loss = self.reg_loss(pred, y[1])
        else:
            loss = self.clf_loss(pred, y[2])

        self.log(
            "trn_loss",
            loss,
            on_step=True,
            on_epoch=True,
            prog_bar=True,
            logger=True,
            batch_size=params["batch_size"],
        )
        return loss

    def validation_step(self, batch, batch_idx):
        self.model.eval()
        torch.set_grad_enabled(False)

        x, locations, y = batch
        pred = self.model(x, locations)

        if self.target == "breslow":
            loss = self.reg_loss(pred, y[1])
            self.val_preds.append(pred.detach().cpu())
        else:
            loss = self.clf_loss(pred, y[2])
            self.log(
                "val_sc",
                loss,
                on_epoch=True,
                prog_bar=True,
                logger=True,
                batch_size=params["batch_size"],
            )

        return loss

    def on_validation_epoch_start(self):
        self.val_preds = []

    def on_validation_epoch_end(self):
        if self.target == "breslow":
            val_preds = torch.concatenate(self.val_preds)
            if len(val_preds) == len(self.val_gts):
                val_preds = val_preds.flatten()
                score = skm.mean_squared_error(self.val_gts, val_preds, squared=False)

                self.log("val_sc", score, on_step=False, on_epoch=True, prog_bar=True, logger=True)

    def configure_optimizers(self):
        opt = torch.optim.Adam(self.model.parameters(), lr=self.lr)
        return opt


EPOCHS = 100
for FOLD in range(N_SPLITS):
    fix_seed(RANDOM_STATE)
    df_trn, df_val, dl_trn, dl_val = get_dls(
        FOLD, bs=params["batch_size"], drop_nan=True, drop_nan_var=VAR
    )

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

    model = Model(
        model=sparseconvmil_model,
        lr=1e-4,
        val_gts=df_val.drop_duplicates(subset=["slide"])[VAR].values,
        target=VAR,
    )

    checkpoint_callback = ModelCheckpoint(
        save_top_k=1,
        monitor="val_sc",
        mode="min",
        dirpath="output",
        filename="{epoch:02d}-{val_sc}",
        verbose=True,
    )
    early_stop_callback = EarlyStopping(monitor="val_sc", min_delta=0.00, patience=5, mode="min")

    trainer = Trainer(
        accelerator="cpu",
        max_epochs=EPOCHS,
        callbacks=[checkpoint_callback, early_stop_callback],
        deterministic="warn",
        log_every_n_steps=1000,
        accumulate_grad_batches=32,
        enable_progress_bar=False,
    )

    trainer.fit(model, train_dataloaders=dl_trn, val_dataloaders=dl_val)
    print(f"{round(trainer.checkpoint_callback.best_model_score.item(),3)}")
    print(trainer.checkpoint_callback.best_model_path)

    t = trainer.checkpoint_callback.best_model_path.split("/")[-1]
    sc = t.split("=")[2].rsplit(".", 1)[0]
    ep = t.split("=")[1].split("-")[0]
    print(f"{VAR} FOLD {FOLD} ep: {ep}, sc: {sc}")

    model_path = f"{MODEL_DIR}/f{FOLD}_{VAR}.pth"
    wts = torch.load(trainer.checkpoint_callback.best_model_path)["state_dict"]
    model.load_state_dict(wts)
    torch.save(model.state_dict(), model_path)
