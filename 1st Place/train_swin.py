import argparse
import glob
import os
import random

import albumentations as A
import cv2
import numpy as np
import pandas as pd
import pytorch_lightning as pl
import sklearn.metrics as skm
import timm
import torch
import torch.nn.functional as F
from pytorch_lightning import Trainer
from pytorch_lightning.callbacks import EarlyStopping, ModelCheckpoint
from sklearn.model_selection import StratifiedKFold
from torch.utils.data import DataLoader, Dataset

parser = argparse.ArgumentParser()
parser.add_argument("--expr", help="experiment name e.g. swin256, swin384", required=True)
parser.add_argument("--fold", help="fold/split (1-5)", required=True, type=int)
parser.add_argument(
    "--meta_csv_path", help="Path to the train or test metadata csv file", required=True
)
parser.add_argument("--label_csv_path", help="Path to the train labels csv file", required=True)
args = parser.parse_args()
EXPR = args.expr  # swin256, swin384
FOLD = args.fold
STAGE = "train"  #'train',test'
META_CSV_PATH = args.meta_csv_path
LABEL_CSV_PATH = args.label_csv_path


if EXPR == "swin256":
    arch = "swinv2_base_window12to16_192to256_22kft1k"
    SZ = 256
    RANDOM_STATE = 256
    batch_size = 16
    batch_size1 = 32


elif EXPR == "swin384":
    arch = "swin_large_patch4_window12_384"
    SZ = 384
    RANDOM_STATE = 888
    batch_size = 10
    batch_size1 = 32

DATA_ROOT = f"./workspace/data/{STAGE}"
MODEL_DIR = f"workspace/models/{EXPR}"
os.makedirs(MODEL_DIR, exist_ok=True)


def fix_seed(seed):
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


fix_seed(RANDOM_STATE)

print("set seed: ", RANDOM_STATE)
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
print(df.breslow.value_counts())


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


# SZ=224 #380
class VMelDataset(Dataset):
    def __init__(self, df, aug=None, aug_stain=False):
        self.df = df
        self.aug = aug
        self.aug_stain = aug_stain

    def __len__(self):
        return len(self.df)

    def __getitem__(self, ix):
        y = self.df.relapse.values[ix]
        y1 = self.df.breslow.values[ix]
        slide = self.df.slide.values[ix]
        path = self.df.path.values[ix]

        img = cv2.imread(path)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        if self.aug:
            img = self.aug(image=img)["image"]  # albumentations

        ##normalize
        img = img.astype(np.float32)
        img = img.transpose(2, 0, 1)
        img = (img - MEAN[:, None, None]) / STD[:, None, None]
        img = torch.from_numpy(img.astype(np.float32))

        return img, (y, y1)


tfm_trn = A.Compose(
    [
        #     A.Transpose(p=0.5),
        A.VerticalFlip(p=0.5),
        A.RandomBrightnessContrast(
            brightness_limit=0.2, contrast_limit=0.2, brightness_by_max=True, p=0.25
        ),
        A.HueSaturationValue(hue_shift_limit=10, sat_shift_limit=20, val_shift_limit=10, p=0.25),
    ]
)
# tfm_trn=None
ds = VMelDataset(df.head(), aug=tfm_trn, aug_stain=False)
x, y = ds[0]
print(f"VMelDataset: x: {x.shape} y: {y[0]}, {y[1]}")

NUM_WORKERS = 1


def get_dls(fold, drop_nan=False, bs=batch_size):
    df_trn, df_val = get_splits(df, fold)
    print("getting data fold", fold, len(df_trn), len(df_val))
    if drop_nan:
        df_trn = df_trn[df_trn.breslow != -1]
        df_val = df_val[df_val.breslow != -1]
        print("after drop bad samples", len(df_trn), len(df_val))

    ds_trn = VMelDataset(df_trn, aug=tfm_trn, aug_stain=False)
    dl_trn = DataLoader(ds_trn, batch_size=bs, shuffle=True, num_workers=NUM_WORKERS)
    ds_val = VMelDataset(df_val)
    dl_val = DataLoader(ds_val, batch_size=bs, shuffle=False, num_workers=NUM_WORKERS)
    # x,y = ds_val[0]
    return df_trn, df_val, dl_trn, dl_val


def get_score_breslow(y_true, y_pred):
    score = skm.mean_squared_error(y_true, y_pred, squared=False)
    return score


class Model(pl.LightningModule):
    def __init__(self, model, lr, target, val_gts=None):
        super().__init__()
        self.model = model
        self.lr = lr
        self.val_preds = []
        self.val_gts = val_gts
        self.target = target

    def forward(self, x: torch.Tensor):
        x = self.model(x)
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

        x, y = batch
        pred = self.model(x)

        if self.target == "breslow":
            loss = self.reg_loss(pred, y[1])
        else:
            loss = self.clf_loss(pred, y[0])

        self.log("trn_loss", loss, on_step=True, on_epoch=True, prog_bar=True, logger=True)
        return loss

    def validation_step(self, batch, batch_idx):
        self.model.eval()
        torch.set_grad_enabled(False)

        x, y = batch
        pred = self.model(x)

        if self.target == "breslow":
            loss = self.reg_loss(pred, y[1])
            self.val_preds.append(pred.detach().cpu())
        else:
            loss = self.clf_loss(pred, y[0])
            self.log("val_sc", loss, on_epoch=True, prog_bar=True, logger=True)

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


EPOCHS = 1000
# train breslow
fix_seed(RANDOM_STATE)
m = timm.create_model(arch, pretrained=True, num_classes=1, in_chans=3)
df_trn, df_val, dl_trn, dl_val = get_dls(FOLD, drop_nan=True)
model = Model(model=m, lr=1e-4, target="breslow", val_gts=df_val.breslow.values)

checkpoint_callback = ModelCheckpoint(
    save_top_k=1, monitor="val_sc", mode="min", dirpath="output", filename="{epoch:02d}-{val_sc}"
)
early_stop_callback = EarlyStopping(
    monitor="val_sc", min_delta=0.00, patience=5, verbose=True, mode="min"
)

trainer = Trainer(
    accelerator="cuda",
    max_epochs=EPOCHS,
    callbacks=[checkpoint_callback, early_stop_callback],
    deterministic="warn",
    log_every_n_steps=1,
    enable_progress_bar=False,
)

trainer.fit(model, train_dataloaders=dl_trn, val_dataloaders=dl_val)
t = trainer.checkpoint_callback.best_model_path.split("/")[-1]
sc = t.split("=")[2].rsplit(".", 1)[0]
ep = t.split("=")[1].split("-")[0]
print(f"{ep}: {sc}")


pretrained_model_path = f"{MODEL_DIR}/{arch}_f{FOLD}_p0.pth"

wts = torch.load(trainer.checkpoint_callback.best_model_path)["state_dict"]
model.load_state_dict(wts)

torch.save(model.state_dict(), pretrained_model_path)


# train relapse
fix_seed(RANDOM_STATE)
m = timm.create_model(arch, pretrained=True, num_classes=2, in_chans=3)
df_trn, df_val, dl_trn, dl_val = get_dls(FOLD, bs=batch_size1)
model = Model(model=m, lr=1e-4, target="relapse")

wts = torch.load(pretrained_model_path)
todel = []
for i, (k, v) in enumerate(wts.items()):
    if "head" in k:
        print("deleting ", k)
        todel.append(k)

for k in todel:
    del wts[k]

model.load_state_dict(wts, strict=False)

for name, param in model.named_parameters():
    if "head" not in name:
        param.requires_grad = False

checkpoint_callback = ModelCheckpoint(
    save_top_k=1, monitor="val_sc", mode="min", dirpath="output", filename="{epoch:02d}-{val_sc}"
)
early_stop_callback = EarlyStopping(
    monitor="val_sc", min_delta=0.00, patience=5, verbose=True, mode="min"
)

trainer = Trainer(
    accelerator="cuda",
    max_epochs=EPOCHS,
    callbacks=[checkpoint_callback, early_stop_callback],
    deterministic="warn",
    log_every_n_steps=1,
    enable_progress_bar=False,
)
trainer.fit(model, train_dataloaders=dl_trn, val_dataloaders=dl_val)
print(f"{round(trainer.checkpoint_callback.best_model_score.item(),3)}")
print(trainer.checkpoint_callback.best_model_path)

t = trainer.checkpoint_callback.best_model_path.split("/")[-1]
sc = t.split("=")[2].rsplit(".", 1)[0]
ep = t.split("=")[1].split("-")[0]
print(f"FOLD {FOLD} ep: {ep}, sc: {sc}")


final_model_path = f"{MODEL_DIR}/{arch}_f{FOLD}_p1.pth"
wts = torch.load(trainer.checkpoint_callback.best_model_path)["state_dict"]
model.load_state_dict(wts)
torch.save(model.state_dict(), final_model_path)
