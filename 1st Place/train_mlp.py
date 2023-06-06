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
from torch import nn
from torch.utils.data import DataLoader, Dataset

parser = argparse.ArgumentParser()
parser.add_argument("--expr", help="experiment name e.g. expr_56_3_224", required=True)
parser.add_argument(
    "--meta_csv_path", help="Path to the train or test metadata csv file", required=True
)
parser.add_argument("--label_csv_path", help="Path to the train labels csv file", required=True)

args = parser.parse_args()
EXPR = args.expr  # expr_56_3_224,expr_40_2_320
STAGE = "train"  #'train',test'
META_CSV_PATH = args.meta_csv_path
LABEL_CSV_PATH = args.label_csv_path

DATA_ROOT = f"./workspace/data/{STAGE}"
FACTORS_DIR = f"{DATA_ROOT}/factors/{EXPR}"

MODEL_DIR = f"./workspace/models/{EXPR}/mlp"
os.makedirs(MODEL_DIR, exist_ok=True)

PRED_DIR = f"./workspace/preds/{STAGE}/{EXPR}"
os.makedirs(PRED_DIR, exist_ok=True)

DEVICE = torch.device("cpu")
NUM_WORKERS = 1
N_SPLITS = 5


def fix_seed(seed):
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


# fix_seed(RANDOM_STATE)


meta = pd.read_csv(META_CSV_PATH)
labels = pd.read_csv(LABEL_CSV_PATH)
labels["slide"] = labels.filename.str.split(".").str[0]


def split_data(df, random_state):
    kf = StratifiedKFold(n_splits=N_SPLITS, shuffle=True, random_state=random_state)
    df["fold"] = -1

    for fold_id, (train_index, test_index) in enumerate(kf.split(df, df.relapse)):
        df.loc[test_index, "fold"] = fold_id
    return df


def get_splits(df, fold):
    df_trn = df[df.fold != fold].copy()
    df_val = df[df.fold == fold].copy()
    return df_trn, df_val


d_breslow = pd.read_csv(f"{FACTORS_DIR}/breslow.csv")[["slide", "breslow_pred"]]
d_ulceration = pd.read_csv(f"{FACTORS_DIR}/ulceration.csv")[["slide", "ulceration_pred"]]
d_factors = pd.merge(d_breslow, d_ulceration)

df = pd.merge(labels, d_factors, on="slide")

feat_cols = ["age", "sex", "melanoma_history", "body_site"]
d = meta.copy()
d["age"] = d["age"].str.slice(1, 3).astype(np.float32) / 100

# map_breslow = {np.nan:-1,'<0.8':0.4,'[0.8 : 1[':0.9, '[1 : 2[':1.5,'[2 : 4[':3, '>=4':4}
# map_ulceraton = {np.nan:-1,'NO':0,'YES':1}
# d['breslow'] = d['breslow'].map(map_breslow).astype(np.float32)
# d['ulceration'] = d['ulceration'].map(map_ulceraton).astype(int)

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

df = pd.merge(df, d[["filename"] + feat_cols])


feat_cols = ["age", "sex", "melanoma_history", "body_site", "ulceration_pred", "breslow_pred"]


df[feat_cols] = df[feat_cols].astype(np.float32)
print(df.relapse.value_counts())


class VMelDataset(Dataset):
    def __init__(self, df, aug=None):
        self.df = df
        self.aug = aug

    def __len__(self):
        return len(self.df)

    def __getitem__(self, ix):
        y = self.df.relapse.values[ix]
        x = self.df[feat_cols].values[ix]
        return x, y


class Model(pl.LightningModule):
    def __init__(self, lr):
        super().__init__()

        self.learning_rate = lr
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
        x1 = self.model(x)
        return x1

    def criterion(self, logits, targets):
        l = F.cross_entropy(logits, targets)
        return l

    def score(self, logits, targets):
        sc = F.cross_entropy(logits, targets)
        return sc

    def training_step(self, batch, batch_idx):
        self.model.train()
        torch.set_grad_enabled(True)

        x, y = batch
        pred = self.model(x)
        loss = self.criterion(pred, y)
        self.log("trn_ce", loss)
        return loss

    def validation_step(self, batch, batch_idx):
        self.model.eval()
        torch.set_grad_enabled(False)

        x, y = batch
        pred = self.model(x)
        loss = self.criterion(pred, y)
        sc = self.score(pred, y)
        self.log("val_ce", loss)
        self.log("val_sc", sc)
        return loss

    def on_validation_epoch_start(self):
        self.val_preds = []

    def configure_optimizers(self):
        opt = torch.optim.Adam(self.model.parameters(), lr=self.learning_rate)
        # sch = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=10)
        return opt
        # return [opt], [sch]


for RANDOM_STATE in [41, 320, 888, 1948]:
    print("TRAINING RANDOM_STATE ", RANDOM_STATE)
    fix_seed(RANDOM_STATE)
    df = split_data(df, random_state=RANDOM_STATE)

    EPOCHS = 1000
    BATCH_SIZE = 128
    LR = 1e-3

    for fold in range(5):
        df_trn, df_val = get_splits(df, fold)
        print("data fold", fold, len(df_trn), len(df_val))
        ds_trn = VMelDataset(df_trn)
        dl_trn = DataLoader(ds_trn, batch_size=BATCH_SIZE, shuffle=True, num_workers=NUM_WORKERS)
        ds_val = VMelDataset(df_val)
        dl_val = DataLoader(ds_val, batch_size=32, shuffle=False, num_workers=NUM_WORKERS)

        model = Model(lr=LR)
        checkpoint_callback = ModelCheckpoint(
            save_top_k=1,
            monitor="val_sc",
            mode="min",
            dirpath="output",
            filename="{epoch:02d}-{val_sc}",
        )
        early_stop_callback = EarlyStopping(
            monitor="val_sc", min_delta=0.00, patience=5, verbose=False, mode="min"
        )

        trainer = Trainer(
            accelerator="cpu",
            max_epochs=EPOCHS,
            callbacks=[checkpoint_callback, early_stop_callback],
            deterministic=True,
            log_every_n_steps=1,
            enable_progress_bar=False,
        )
        trainer.fit(model, train_dataloaders=dl_trn, val_dataloaders=dl_val)

        t = trainer.checkpoint_callback.best_model_path.split("/")[-1]
        sc = t.split("=")[2].rsplit(".", 1)[0]
        ep = t.split("=")[1].split("-")[0]
        print(f"{ep}: {sc}")

        model_path = f"{MODEL_DIR}/r{RANDOM_STATE}_f{fold}.pth"
        wts = torch.load(trainer.checkpoint_callback.best_model_path)["state_dict"]
        model.load_state_dict(wts)
        torch.save(model.state_dict(), model_path)

        model.eval()

        preds = []
        gts = []
        with torch.no_grad():
            for x, y in dl_val:
                x = x.to(DEVICE)
                pr = model(x).cpu()
                preds.append(pr)
                gts.append(y.numpy())

        y = np.concatenate(gts)
        y_pred = torch.concatenate(preds)
        y_pred = F.softmax(y_pred, dim=1).numpy()[:, 1]
        print(f"FOLD {fold}", skm.log_loss(y, y_pred))
        d = df_val[["filename", "relapse"]].copy()
        d["pred"] = y_pred
        d.to_csv(f"{PRED_DIR}/pred_r{RANDOM_STATE}_f{fold}.csv", index=False)
