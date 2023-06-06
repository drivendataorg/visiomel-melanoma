import glob
import os
import re
from collections import defaultdict

import numpy as np
# Third-party libraries
import pandas as pd
# from torch.utils.data import DataLoader, WeightedRandomSampler
import torch
import torch.nn as nn
import torchvision
from assets.src.datasets import ImageTabular
from assets.src.imputers import ulceration_breslow_from_relapse_imputer
# Local libraries
from assets.src.models import BreslowUlcerationRelapseModel, DummyModel
from PIL import Image
from torchvision.transforms import (ColorJitter, Compose, GaussianBlur,
                                    Normalize, RandomCrop,
                                    RandomHorizontalFlip, RandomRotation,
                                    RandomVerticalFlip, ToTensor)
from tqdm import tqdm

################################################################
################################################################
################################################################
################################################################
################################################################
################################################################
# Use if data csv already cosntructed:


def make_dataframe_with_latent_space(
    model_path="/Users/Happpyyyyyyy/Documents/VisioMel/trained_models/remapped_best_model_loss_tritrain_wh_rotate.pth",
    image_path="/Users/Happpyyyyyyy/Documents/VisioMel/data/images/",
    path_train="data/train_dataframe.csv",
    path_val="data/val_dataframe.csv",
    qupath="/Users/Happpyyyyyyy/Documents/VisioMel/data/qupath/",
    remove_filename=False,
):
    ## Init
    m = DummyModel(model_path=model_path).eval()

    transforms = Compose(
        [ToTensor(), Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])]
    )
    object_dict_train = {
        "filename": [],
        "lymph": [],
        "macro": [],
        "epithelial": [],
        "neutro": [],
    }  # , "inv_lymph": [], "inv_macro": [], "inv_epithelial": [], "inv_neutro": []}
    logits_dict_train = {"filename": [], "latent": []}
    object_dict_val = {
        "filename": [],
        "lymph": [],
        "macro": [],
        "epithelial": [],
        "neutro": [],
    }  # , "inv_lymph": [], "inv_macro": [], "inv_epithelial": [], "inv_neutro": []}
    logits_dict_val = {"filename": [], "latent": []}

    #### Create image datasets :
    train_dataframe = pd.read_csv(path_train)
    val_dataframe = pd.read_csv(path_val)

    train_dataframe = ulceration_breslow_from_relapse_imputer(train_dataframe)
    val_dataframe = ulceration_breslow_from_relapse_imputer(val_dataframe)

    # Process train
    print("\t Processing Train Dataframe :")
    for filename in tqdm(train_dataframe.filename.to_numpy()):
        logits_dict_train["filename"].append(filename)
        slide_path = os.path.join(image_path, filename)

        slide = Image.open(slide_path)
        slide = transforms(slide)

        img = slide.unsqueeze(0)
        with torch.no_grad():
            logits_dict_train["latent"].append(m(img).squeeze().numpy())
        if qupath is not None:
            object_dict_train = update_object_dict(qupath, filename, object_dict_train)

        # Process train
    print("\t Processing Val Dataframe :")
    for filename in tqdm(val_dataframe.filename.to_numpy()):
        logits_dict_val["filename"].append(filename)
        slide_path = os.path.join(image_path, filename)

        slide = Image.open(slide_path)
        slide = transforms(slide)

        img = slide.unsqueeze(0)
        with torch.no_grad():
            logits_dict_val["latent"].append(m(img).squeeze().numpy())
        if qupath is not None:
            object_dict_val = update_object_dict(qupath, filename, object_dict_val)

    # Append elements :
    latent_df_train, latent_df_val = pd.DataFrame.from_dict(
        logits_dict_train
    ), pd.DataFrame.from_dict(logits_dict_val)
    train_dataframe = train_dataframe.merge(latent_df_train, how="inner", on="filename")
    val_dataframe = val_dataframe.merge(latent_df_val, how="inner", on="filename")

    if qupath is not None:
        objects_df_train, objects_df_val = pd.DataFrame.from_dict(
            object_dict_train
        ), pd.DataFrame.from_dict(object_dict_val)
        train_dataframe = train_dataframe.merge(objects_df_train, how="inner", on="filename")
        val_dataframe = val_dataframe.merge(objects_df_val, how="inner", on="filename")

    # Remove columns :
    train_dataframe = train_dataframe.drop(
        columns=[
            "tif_cksum",
            "tif_size",
            "us_tif_url",
            "eu_tif_url",
            "as_tif_url",
            "weights",
            "resolution",
        ],
        errors="ignore",
    )
    val_dataframe = val_dataframe.drop(
        columns=["tif_cksum", "tif_size", "us_tif_url", "eu_tif_url", "as_tif_url", "resolution"],
        errors="ignore",
    )

    if remove_filename:
        train_dataframe = train_dataframe.drop(columns=["filename"], errors="ignore")
        val_dataframe = val_dataframe.drop(columns=["filename"], errors="ignore")

    train_dataframe.to_pickle("train_df_latent_space")
    val_dataframe.to_pickle("val_df_latent_space")
    # train_dataframe.to_pickle("train_df_latent_space", index=False)
    # val_dataframe.to_pickle("val_df_latent_space", index=False)

    return None


def make_dataframe(
    image_path="/Users/Happpyyyyyyy/Documents/VisioMel/data/images/",
    path_train="data/train_dataframe.csv",
    path_val="data/val_dataframe.csv",
    model_path="./trained_models/remapped_best_model_loss_tritrain_wh_rotate.pth",
    qupath="/Users/Happpyyyyyyy/Documents/VisioMel/data/qupath/",
    model_paths=None,
    append_names=None,
    remove_filename=True,
    save=True,
):
    assert (model_paths is None) or (model_path is None), "Not both!"

    # Init
    out_name_train, out_name_val = "data/train_dataframe_", "data/val_dataframe_"

    object_dict_train = {
        "filename": [],
        "lymph": [],
        "macro": [],
        "epithelial": [],
        "neutro": [],
    }  # , "inv_lymph": [], "inv_macro": [], "inv_epithelial": [], "inv_neutro": []}
    logits_dict_train = {
        "filename": [],
        "logit_0": [],
        "logit_1": [],
        "logit_2": [],
        "logit_3": [],
        "logit_4": [],
        "logit_5": [],
        "logit_6": [],
    }
    object_dict_val = {
        "filename": [],
        "lymph": [],
        "macro": [],
        "epithelial": [],
        "neutro": [],
    }  # , "inv_lymph": [], "inv_macro": [], "inv_epithelial": [], "inv_neutro": []}
    logits_dict_val = {
        "filename": [],
        "logit_0": [],
        "logit_1": [],
        "logit_2": [],
        "logit_3": [],
        "logit_4": [],
        "logit_5": [],
        "logit_6": [],
    }

    #### Create image datasets :
    train_dataframe = pd.read_csv(path_train)
    val_dataframe = pd.read_csv(path_val)

    train_dataframe = ulceration_breslow_from_relapse_imputer(train_dataframe)
    val_dataframe = ulceration_breslow_from_relapse_imputer(val_dataframe)

    if model_path is not None:
        model = BreslowUlcerationRelapseModel()
        model.load_state_dict(torch.load(model_path, map_location=torch.device("cpu")))
        model.eval()
    if model_paths is not None:
        model_breslow = torchvision.models.resnet18(pretrained=False)
        model_relapse = torchvision.models.resnet18(pretrained=False)
        model_ulceration = torchvision.models.resnet34(pretrained=False)

        model_breslow.fc = nn.Linear(512, 5)
        model_relapse.fc = nn.Linear(512, 1)
        model_ulceration.fc = nn.Linear(512, 1)

        for pt in model_paths:
            print(pt)
            if "breslow" in pt:
                model_breslow.load_state_dict(torch.load(pt, map_location=torch.device("cpu")))
            elif "ulceration" in pt:
                model_ulceration.load_state_dict(torch.load(pt, map_location=torch.device("cpu")))
            else:
                model_relapse.load_state_dict(torch.load(pt, map_location=torch.device("cpu")))
        model_breslow.eval()
        model_relapse.eval()
        model_ulceration.eval()

    # Get for Train
    print("\t Processing Train Dataframe :")
    for filename in tqdm(train_dataframe.filename.to_numpy()):
        if model_paths is not None:
            logits_dict_train["filename"].append(filename)
            slide_path = os.path.join(image_path, filename)
            logits_dict_train = update_logits_dict_three_models(
                slide_path, logits_dict_train, model_breslow, model_relapse, model_ulceration
            )
        if model_path is not None:
            logits_dict_train["filename"].append(filename)
            slide_path = os.path.join(image_path, filename)
            logits_dict_train = update_logits_dict(slide_path, logits_dict_train, model)
        if qupath is not None:
            object_dict_train = update_object_dict(qupath, filename, object_dict_train)
    for key in logits_dict_train:
        print(key, len(logits_dict_train[key]))
    print("__")
    for key in object_dict_train:
        print(key, len(object_dict_train[key]))

    # Get for Val
    print("\t Processing Val Dataframe :")
    for filename in tqdm(val_dataframe.filename.to_numpy()):
        if model_paths is not None:
            logits_dict_val["filename"].append(filename)
            slide_path = os.path.join(image_path, filename)
            logits_dict_val = update_logits_dict_three_models(
                slide_path, logits_dict_val, model_breslow, model_relapse, model_ulceration
            )
        if model_path is not None:
            logits_dict_val["filename"].append(filename)
            slide_path = os.path.join(image_path, filename)
            logits_dict_val = update_logits_dict(slide_path, logits_dict_val, model)
        if qupath is not None:
            object_dict_val = update_object_dict(qupath, filename, object_dict_val)
    for key in logits_dict_val:
        print(key, len(logits_dict_val[key]))
    print("__")
    for key in object_dict_val:
        print(key, len(object_dict_val[key]))
    # Append to dataframe names and merge dataframes
    print(f"Before: {train_dataframe.shape=}")
    print(f"Before: {val_dataframe.shape=}")
    if model_path is not None:
        logits_df_train, logits_df_val = pd.DataFrame.from_dict(
            logits_dict_train
        ), pd.DataFrame.from_dict(logits_dict_val)
        train_dataframe = train_dataframe.merge(logits_df_train, how="inner", on="filename")
        val_dataframe = val_dataframe.merge(logits_df_val, how="inner", on="filename")
        out_name_train += "logits_"
        out_name_val += "logits_"
    if model_paths is not None:
        logits_df_train, logits_df_val = pd.DataFrame.from_dict(
            logits_dict_train
        ), pd.DataFrame.from_dict(logits_dict_val)
        train_dataframe = train_dataframe.merge(logits_df_train, how="inner", on="filename")
        val_dataframe = val_dataframe.merge(logits_df_val, how="inner", on="filename")
        out_name_train += "logits_three_models_"
        out_name_val += "logits_three_models_"
    print(f"After merge 1: {train_dataframe.shape=}")
    print(f"After merge 1: {val_dataframe.shape=}")
    if qupath is not None:
        objects_df_train, objects_df_val = pd.DataFrame.from_dict(
            object_dict_train
        ), pd.DataFrame.from_dict(object_dict_val)
        train_dataframe = train_dataframe.merge(objects_df_train, how="inner", on="filename")
        val_dataframe = val_dataframe.merge(objects_df_val, how="inner", on="filename")
        out_name_train += "objects"
        out_name_val += "objects"

    if append_names is not None:
        out_name_train += append_names
        out_name_val += append_names
    out_name_train += ".csv"
    out_name_val += ".csv"

    # Remove columns :
    print(f"After merges: {train_dataframe.shape=}")
    print(f"After merges: {val_dataframe.shape=}")
    train_dataframe = train_dataframe.drop(
        columns=[
            "tif_cksum",
            "tif_size",
            "us_tif_url",
            "eu_tif_url",
            "as_tif_url",
            "ulceration",
            "breslow",
            "weights",
            "resolution",
        ],
        errors="ignore",
    )
    val_dataframe = val_dataframe.drop(
        columns=[
            "tif_cksum",
            "tif_size",
            "us_tif_url",
            "eu_tif_url",
            "as_tif_url",
            "ulceration",
            "breslow",
            "resolution",
        ],
        errors="ignore",
    )

    if remove_filename:
        train_dataframe = train_dataframe.drop(columns=["filename"], errors="ignore")
        val_dataframe = val_dataframe.drop(columns=["filename"], errors="ignore")
    if save:
        train_dataframe.to_csv(out_name_train, index=False)
        val_dataframe.to_csv(out_name_val, index=False)

        return out_name_train, out_name_val
    else:
        return train_dataframe, val_dataframe


def update_object_dict(qupath, filename, object_dict):
    tsv_dict = {"lymph": 0, "macro": 0, "epithelial": 0, "neutro": 0}
    total = 0
    num_patchs = 0
    for tsv_path in glob.glob(qupath + filename[:-4] + "_*"):
        num_patchs += 1
        tsv = pd.read_csv(tsv_path, sep="\t")
        values, counts = np.unique(tsv.name.to_numpy(), return_counts=True)
        for label, count in zip(values, counts):
            if label != "nolabe":
                tsv_dict[label] += count
                total += count
    if total > 0:
        normalized = normalize_dict(tsv_dict, total)
        object_dict["filename"].append(filename)
        for key in ["lymph", "macro", "epithelial", "neutro"]:
            val = normalized[key]
            object_dict[key].append(val)
            # if val != 0 :
            #     object_dict["inv_"+key].append(1/val)
            # else:
            #     object_dict["inv_"+key].append(1)
    else:
        object_dict["filename"].append(filename)
        for key in ["lymph", "macro", "epithelial", "neutro"]:
            object_dict[key].append(0)
            # object_dict["inv_"+key].append(0)
    return object_dict


def update_logits_dict_three_models(
    slide_path, logits_dict, model_breslow, model_relapse, model_ulceration
):
    transforms = Compose(
        [ToTensor(), Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])]
    )
    slide = Image.open(slide_path)
    slide = transforms(slide)

    img = slide.unsqueeze(0)
    # print(img.shape)
    with torch.no_grad():
        logits1 = nn.Softmax()(model_breslow(img))
        logits2 = nn.Sigmoid()(model_relapse(img))
        logits3 = nn.Sigmoid()(model_ulceration(img))
    logits1, logits2, logits3 = (
        logits1.squeeze().numpy(),
        logits2.squeeze().numpy(),
        logits3.squeeze().numpy(),
    )
    idx_logit = 0
    for col in range(len(logits1)):
        logits_dict[f"logit_{idx_logit}"].append(logits1[col])
        idx_logit += 1
    logits_dict[f"logit_{idx_logit}"].append(logits2.item())
    logits_dict[f"logit_{idx_logit+1}"].append(logits3.item())
    return logits_dict


def update_logits_dict(slide_path, logits_dict, model):
    transforms = Compose(
        [ToTensor(), Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])]
    )
    slide = Image.open(slide_path)
    slide = transforms(slide)

    img = slide.unsqueeze(0)
    # print(img.shape)
    with torch.no_grad():
        logits1, logits2, logits3 = model(img)
    logits1, logits2, logits3 = (
        logits1.squeeze().numpy(),
        logits2.squeeze().numpy(),
        logits3.squeeze().numpy(),
    )
    idx_logit = 0
    for col in range(len(logits1)):
        logits_dict[f"logit_{idx_logit}"].append(logits1[col])
        idx_logit += 1
    logits_dict[f"logit_{idx_logit}"].append(logits2.item())
    logits_dict[f"logit_{idx_logit+1}"].append(logits3.item())
    return logits_dict


def normalize_dict(d, total):
    for key in d:
        d[key] = d[key] / total
    return d


def append_logits(model, dataset, df):
    logits_all = [[], [], []]
    with torch.no_grad():
        # Fill train dataframe
        for idx, item in enumerate(tqdm(dataset)):
            img = item["slide"].unsqueeze(0)
            # print(img.shape)
            logits1, logits2, logits3 = model(img)
            logits_all[0].append(logits1.squeeze().numpy())
            logits_all[1].append(logits2.squeeze().numpy())
            logits_all[2].append(logits3.squeeze().numpy())
    logits1_array = np.stack(logits_all[0], axis=0)
    logits2_array = np.stack(logits_all[1], axis=0)[:, np.newaxis]
    logits3_array = np.stack(logits_all[2], axis=0)[:, np.newaxis]
    idx_logit = 0
    for col in range(logits1_array.shape[1]):
        df[f"logit_{idx_logit}"] = logits1_array[:, col, np.newaxis]
        idx_logit += 1
    df[f"logit_{idx_logit}"] = logits2_array
    df[f"logit_{idx_logit+1}"] = logits3_array
    return df


def age_val(string):
    a, b = re.findall(r"\d+", string)
    a = int(a)
    b = int(b)
    return (a + b) // 2


def clean_df(df, maps):
    # Replace by tokens
    df = df.replace(maps)
    # Transform age
    df.age = [age_val(string) for string in df.age]
    # # Encode body_Site :
    # df.body_site = bodySiteEnc.transform(df.body_site.to_numpy)
    # Fillna : (There shouldnt be any left)
    df = df.fillna(0)
    return df
