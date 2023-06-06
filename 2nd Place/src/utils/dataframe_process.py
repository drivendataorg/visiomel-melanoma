import os
import re

# Third-party libraries
import pandas as pd

# from torch.utils.data import DataLoader, WeightedRandomSampler
import torch
from PIL import Image
from torchvision.transforms import Compose, Normalize, ToTensor
from tqdm import tqdm

from .imputers import ulceration_breslow_from_relapse_imputer

# Local libraries
from .models import DummyModel


def make_dataframe_with_latent_space(
    model_path="tritrain.pth",
    image_path="data/images/",
    path_train="data/train_dataframe.csv",
    path_val="data/val_dataframe.csv",
    save_path="data/",
    remove_filename=False,
):
    ## Init
    m = DummyModel(model_path=model_path).eval()

    transforms = Compose(
        [ToTensor(), Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])]
    )
    logits_dict_train = {"filename": [], "latent": []}
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
        if not os.path.exists(slide_path):
            slide_path = os.path.join(image_path, filename.replace(".tif", ".png"))

        slide = Image.open(slide_path)
        slide = transforms(slide)

        img = slide.unsqueeze(0)
        with torch.no_grad():
            logits_dict_train["latent"].append(m(img).squeeze().numpy())

        # Process train
    print("\t Processing Val Dataframe :")
    for filename in tqdm(val_dataframe.filename.to_numpy()):
        logits_dict_val["filename"].append(filename)
        slide_path = os.path.join(image_path, filename)
        if not os.path.exists(slide_path):
            slide_path = os.path.join(image_path, filename.replace(".tif", ".png"))

        slide = Image.open(slide_path)
        slide = transforms(slide)

        img = slide.unsqueeze(0)
        with torch.no_grad():
            logits_dict_val["latent"].append(m(img).squeeze().numpy())

    # Append elements :
    latent_df_train, latent_df_val = pd.DataFrame.from_dict(
        logits_dict_train
    ), pd.DataFrame.from_dict(logits_dict_val)
    train_dataframe = train_dataframe.merge(latent_df_train, how="inner", on="filename")
    val_dataframe = val_dataframe.merge(latent_df_val, how="inner", on="filename")

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

    train_dataframe.to_pickle(save_path + "/train_df_latent_space")
    val_dataframe.to_pickle(save_path + "/val_df_latent_space")

    return None


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
    # Fillna : (There shouldnt be any left)
    df = df.fillna(0)
    return df
