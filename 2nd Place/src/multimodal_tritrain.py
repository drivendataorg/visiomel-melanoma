import os
import random

import numpy as np

# Third-party libraries
import pandas as pd
import torch
import torch.nn as nn

# Local libraries
from utils.dataframe_process import clean_df, make_dataframe_with_latent_space
from utils.datasets import ImageTabular
from utils.models import ImageTabularModel
from utils.utils import train_epoch


################################################################
################################################################
def warn(*args, **kwargs):
    pass


import warnings

warnings.warn = warn

root = os.getcwd()

######## Create dataset :
make_dataframe_with_latent_space(
    model_path=root + "/models/tritrain.pth",
    image_path=root + "/data/images",
    path_train=root + "/data/train_dataframe.csv",
    path_val=root + "/data/val_dataframe.csv",
    save_path=root + "/data/",
    remove_filename=False,
)

### Auxiliary funcs :


def get_dataframes(train_path, val_path):
    train_dataframe = pd.read_pickle(train_path)
    val_dataframe = pd.read_pickle(val_path)

    ##### Clean Dataframes :
    train_dataframe = train_dataframe.fillna("nan")
    val_dataframe = val_dataframe.fillna("nan")

    ####### Get mappings
    body_site_mapping = {
        "nan": -1,
        "trunk": 0,
        "arm": 1,
        "seat": 2,
        "head": 3,
        "neck": 3,
        "head/neck": 3,
        "face": 4,
        "trunc": 0,
        "leg": 5,
        "forearm": 6,
        "upper limb": 7,
        "shoulder": 7,
        "upper limb/shoulder": 7,
        "lower limb": 8,
        "hip": 8,
        "lower limb/hip": 8,
        "hand": 9,
        "toe": 10,
        "foot": 9,
        "nail": 9,
        "hand/foot/nail": 9,
        "thigh": 11,
        "sole": 12,
        "finger": 13,
        "scalp": 14,
    }

    sex_mapping = {
        1: 0,
        2: 1,
        "nan": 2,
    }
    melanoma_history_mapping = {
        "NO": 0,
        "YES": 1,
        "nan": 2,
    }
    breslow_mapping = {
        "<0.8": 0,
        "[0.8 : 1[": 1,
        "[1 : 2[": 2,
        "[2 : 4[": 3,
        ">=4": 4,
    }
    maps = {
        "sex": sex_mapping,
        "melanoma_history": melanoma_history_mapping,
        "body_site": body_site_mapping,
        "breslow": breslow_mapping,
    }

    ####### Apply clean func
    train_dataframe = clean_df(train_dataframe, maps)
    val_dataframe = clean_df(val_dataframe, maps)

    # Get columns for data :
    features_names = list(train_dataframe.columns.values)
    features_names.remove("relapse")
    features_names.remove("filename")
    features_names.remove("latent")
    features_names.remove("ulceration")
    features_names.remove("breslow")

    return train_dataframe, val_dataframe, features_names


def set_seed(seed=0):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    np.random.seed(seed)
    random.seed(seed)


class FocalLoss(nn.Module):
    def __init__(self, gamma=1.0, pos_weight=1):
        super(FocalLoss, self).__init__()
        self.gamma = torch.tensor(gamma, dtype=torch.float32)
        self.eps = 1e-6
        self.pos_weight = pos_weight

    def forward(self, probs, target):
        # Predicted probabilities for the negative class
        q = 1 - probs
        p = probs
        # For numerical stability (so we don't inadvertently take the log of 0)
        p = p.clamp(self.eps, 1.0 - self.eps)
        q = q.clamp(self.eps, 1.0 - self.eps)

        # Loss for the positive examples
        pos_loss = -(q**self.gamma) * torch.log(p)
        if self.pos_weight is not None:
            pos_loss *= self.pos_weight

        # Loss for the negative examples
        neg_loss = -(p**self.gamma) * torch.log(q)

        loss = target * pos_loss + (1 - target) * neg_loss

        return loss.sum()


def run_model(config, train_dataframe, val_dataframe, features_names):
    #### Train
    model = ImageTabularModel(
        len(features_names),
        config["model_type"],
        dropout=config["dropout"],
        relapse_only=config["relapse_only"],
    )

    loss_breslow = torch.nn.CrossEntropyLoss()
    loss_ulceration = torch.nn.BCELoss()
    loss_relapse = torch.nn.BCELoss()

    optimizer = torch.optim.Adam(model.parameters(), lr=config["lr"])
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, patience=7, verbose=True, factor=0.2
    )
    best_loss = np.Inf
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    train_dataset = ImageTabular(
        train_dataframe, features_names, aug_embedding=None, aug_tabular=None
    )
    val_dataset = ImageTabular(val_dataframe, features_names)

    # Create data loaders for our datasets; shuffle for training, not for validation
    train_loader = torch.utils.data.DataLoader(
        train_dataset, batch_size=config["batch_size"], shuffle=True
    )
    val_loader = torch.utils.data.DataLoader(
        val_dataset, batch_size=config["batch_size"], shuffle=False
    )

    min_train_loss_r, min_val_loss_r = np.Inf, np.Inf
    for epoch in range(config["epochs"]):
        print("Epoch {}/{}".format(epoch + 1, config["epochs"]))

        loss_relapse = FocalLoss(gamma=config["gamma"], pos_weight=config["pos_weight_FL"])
        # Train Loop
        model.train()
        _, _, _, _, _, _, _, min_train_loss_r, _ = train_epoch(
            device,
            model,
            optimizer,
            relapse_only,
            train_loader,
            loss_breslow,
            loss_ulceration,
            loss_relapse,
            config["alpha"],
            config["beta"],
            config["gamma"],
            min_train_loss_r,
            clip=None,
            patches=False,
            verbose=False,
        )

        # Validation Loop
        model.eval()
        loss_relapse = torch.nn.BCELoss()
        with torch.no_grad():
            (
                val_loss,
                labels_r,
                labels_u,
                labels_b,
                predictions_r,
                predictions_u,
                predictions_b,
                min_val_loss_r,
                val_loss_r,
            ) = train_epoch(
                device,
                model,
                optimizer,
                relapse_only,
                val_loader,
                loss_breslow,
                loss_ulceration,
                loss_relapse,
                config["alpha"],
                config["beta"],
                config["gamma"],
                min_val_loss_r,
                patches=False,
                verbose=False,
            )

        scheduler.step(val_loss)
        if val_loss_r < best_loss:
            best_loss = val_loss_r
            torch.save(model.state_dict(), "models/before_finetune.pth")
            print(f"Best model saved at epoch {epoch+1}, has val relapse loss : {val_loss_r}")


def finetune_model(config, train_dataframe, val_dataframe, features_names, model_path):
    #### Train
    model = ImageTabularModel(
        len(features_names),
        config["model_type"],
        dropout=config["dropout"],
        relapse_only=config["relapse_only"],
    )
    model.load_state_dict(torch.load(model_path))

    loss_breslow = torch.nn.CrossEntropyLoss()
    loss_ulceration = torch.nn.BCELoss()
    loss_relapse = torch.nn.BCELoss()

    optimizer = torch.optim.Adam(model.parameters(), lr=config["lr"])
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, patience=7, verbose=True, factor=0.2
    )
    best_loss = np.Inf
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    train_dataset = ImageTabular(train_dataframe, features_names, aug_embedding=None)
    val_dataset = ImageTabular(val_dataframe, features_names)

    # Create data loaders for our datasets; shuffle for training, not for validation
    train_loader = torch.utils.data.DataLoader(
        train_dataset, batch_size=config["batch_size"], shuffle=True
    )
    val_loader = torch.utils.data.DataLoader(
        val_dataset, batch_size=config["batch_size"], shuffle=False
    )
    ##################################
    ##################################
    ##################################
    model.eval()
    loss_relapse = torch.nn.BCELoss()
    with torch.no_grad():
        (
            val_loss,
            labels_r,
            labels_u,
            labels_b,
            predictions_r,
            predictions_u,
            predictions_b,
            _,
            val_loss_r,
        ) = train_epoch(
            device,
            model,
            optimizer,
            relapse_only,
            val_loader,
            loss_breslow,
            loss_ulceration,
            loss_relapse,
            config["alpha"],
            config["beta"],
            config["gamma"],
            0,
            patches=False,
            verbose=False,
        )
    min_train_loss_r, min_val_loss_r = np.Inf, np.Inf
    for epoch in range(config["epochs"]):
        print("Finetune Epoch {}/{}".format(epoch + 1, config["epochs"]))

        loss_relapse = FocalLoss(gamma=config["gamma"], pos_weight=config["pos_weight_FL"])
        # Train Loop
        model.train()
        _, _, _, _, _, _, _, min_train_loss_r, _ = train_epoch(
            device,
            model,
            optimizer,
            relapse_only,
            train_loader,
            loss_breslow,
            loss_ulceration,
            loss_relapse,
            config["alpha"],
            config["beta"],
            config["gamma"],
            min_train_loss_r,
            clip=None,
            patches=False,
            verbose=False,
        )

        # Validation Loop
        model.eval()
        loss_relapse = torch.nn.BCELoss()
        with torch.no_grad():
            (
                val_loss,
                labels_r,
                labels_u,
                labels_b,
                predictions_r,
                predictions_u,
                predictions_b,
                min_val_loss_r,
                val_loss_r,
            ) = train_epoch(
                device,
                model,
                optimizer,
                relapse_only,
                val_loader,
                loss_breslow,
                loss_ulceration,
                loss_relapse,
                config["alpha"],
                config["beta"],
                config["gamma"],
                min_val_loss_r,
                patches=False,
                verbose=False,
            )

        scheduler.step(val_loss)
        if val_loss_r < best_loss:
            best_loss = val_loss_r
            torch.save(model.state_dict(), "models/after_finetune.pth")
            print(f"Best model saved at epoch {epoch+1}, has val relapse loss : {val_loss_r}")


epochs = 50
batch_size = 16

gamma = 0.2
beta = 0.5
alpha = 0.5

gammaFL = 0.2
pos_weight_FL = 1

lr = 1e-4
dropout = 0.2
relapse_only = False
model_type = "FC"
age_denominator = 100
breslow_denominator = 4
body_site_denominator = 14

path_train = root + "/data/train_df_latent_space"
path_val = root + "/data/val_df_latent_space"

# Set seed for rep
set_seed(seed=0)

train_dataframe, val_dataframe, features_names = get_dataframes(path_train, path_val)
# Process values
if age_denominator != 1.0:
    train_dataframe["age"] = train_dataframe["age"].div(age_denominator).round(2)
    val_dataframe["age"] = val_dataframe["age"].div(age_denominator).round(2)
if breslow_denominator != 1.0:
    train_dataframe["breslow"] = train_dataframe["breslow"].div(breslow_denominator).round(2)
    val_dataframe["breslow"] = val_dataframe["breslow"].div(breslow_denominator).round(2)
if body_site_denominator != 1.0:
    train_dataframe["body_site"] = train_dataframe["body_site"].div(body_site_denominator).round(2)
    val_dataframe["body_site"] = val_dataframe["body_site"].div(body_site_denominator).round(2)

config = {
    "batch_size": batch_size,
    "epochs": epochs,
    "relapse_only": relapse_only,
    "alpha": alpha,
    "beta": beta,
    "gamma": gamma,
    "lr": lr,
    "dropout": dropout,
    "model_type": model_type,
    "tabular_features": features_names,
    "age_denominator": age_denominator,
    "body_site_denominator": body_site_denominator,
    "breslow_denominator": breslow_denominator,
    "gamma_FL": gammaFL,
    "pos_weight_FL": pos_weight_FL,
}

run_model(config, train_dataframe, val_dataframe, features_names)

saved_model_path = "models/before_finetune.pth"
finetune_model(config, val_dataframe, val_dataframe, features_names, saved_model_path)
