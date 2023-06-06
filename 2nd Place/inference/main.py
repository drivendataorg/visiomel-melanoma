import os
import pickle
from pathlib import Path

import numpy as np
import pandas as pd
import pyvips
import torch
import torch.nn as nn
# Local libraries
from assets.src.models import DummyModel, ImageTabularModel
from assets.src.process_dataframe import clean_df
from joblib import dump, load
from PIL import Image
from torchvision.transforms import (ColorJitter, Compose, GaussianBlur,
                                    Normalize, RandomCrop,
                                    RandomHorizontalFlip, RandomRotation,
                                    RandomVerticalFlip, ToTensor)

################################################################
################################################################

DATA_ROOT = Path("/code_execution/data")


def predict(
    device,
    test_dataframe,
    features_names,
    model_type,
    dropout,
    relapse_only,
    ImageTabularModel_path,
):
    model = ImageTabularModel(
        len(features_names), model_type, dropout=dropout, relapse_only=relapse_only
    )
    model.load_state_dict(torch.load(ImageTabularModel_path))  # TODO
    model = model.to(device)
    model = model.eval()

    preds_dict = {"filename": [], "relapse": []}
    for idx in range(len(test_dataframe)):
        sample = test_dataframe.iloc[idx]
        tabular = torch.tensor(sample[features_names], dtype=torch.float32)
        tabular = tabular.unsqueeze(0)
        tabular = tabular.to(device)

        img_embedding = torch.tensor(sample.latent, dtype=torch.float32).to(device)
        img_embedding = img_embedding.unsqueeze(0)
        img_embedding = img_embedding.to(device)

        with torch.no_grad():
            breslow, ulceration, relapse_pred = model(img_embedding, tabular)
        pred = relapse_pred.squeeze().cpu().numpy()

        if sample.age > 0.9:
            pred = min(pred, max(0.05, pred - 0.27))

        preds_dict["filename"].append(sample["filename"])
        preds_dict["relapse"].append(pred)

    return preds_dict


def create_test_df(device, data_root=Path("/code_execution/data")):
    test_dataframe = pd.read_csv(
        data_root / "test_metadata.csv"
    )  # os.path.join(data_root,"/test_metadata.csv"))

    # Remove columns :
    filenames = test_dataframe.filename.to_numpy()
    test_dataframe = test_dataframe.drop(
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
    test_dataframe = test_dataframe.fillna("nan")
    #### Load Model :
    dummy_model_path = "assets/trained_models/remapped_best_model_loss_tritrain_wh_rotate.pth"
    dummy_model = DummyModel(model_path=dummy_model_path)
    dummy_model = dummy_model.to(device)
    dummy_model = dummy_model.eval()

    ##### Construct Clinical data + image embedding
    logits_dict = {"filename": [], "latent": []}
    for filename in filenames:
        logits_dict["filename"].append(filename)
        img = get_image(data_root / filename)  # os.join(data_root,img_path))
        with torch.no_grad():
            logits_dict["latent"].append(
                dummy_model(img.unsqueeze(0).to(device)).cpu().squeeze().numpy()
            )
    latent_df = pd.DataFrame.from_dict(logits_dict)
    test_dataframe = test_dataframe.merge(latent_df, how="inner", on="filename")

    # Clean dataframe :
    # body_sites = test_dataframe.body_site.unique()
    # body_site_mapping = {label: i for i, label in enumerate(body_sites)}
    sex_mapping = {1: 0, 2: 1, "nan": 3}
    melanoma_history_mapping = {"NO": 0, "YES": 1, "nan": 2}
    body_site_mapping = {
        "nan": -1,
        "trunk": 0,
        "arm": 1,
        "seat": 2,
        "head": 3,
        "neck": 3,
        "head/neck": 3,
        "face": 4,
        "trunc": 5,
        "leg": 6,
        "forearm": 7,
        "upper limb": 8,
        "shoulder": 8,
        "upper limb/shoulder": 8,
        "lower limb": 9,
        "hip": 9,
        "lower limb/hip": 9,
        "hand": 10,
        "toe": 11,
        "foot": 12,
        "nail": 12,
        "hand/foot/nail": 12,
        "thigh": 13,
        "sole": 14,
        "finger": 15,
        "scalp": 16,
    }
    maps = {
        "sex": sex_mapping,
        "melanoma_history": melanoma_history_mapping,
        "body_site": body_site_mapping,
    }
    ####### Apply clean func
    test_dataframe = clean_df(test_dataframe, maps)

    return test_dataframe, filenames


def get_image(
    path,
    transforms=Compose(
        [ToTensor(), Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])]
    ),
):
    slide = pyvips.Image.new_from_file(path)
    n = slide.get_n_pages()
    # Height and width of page 0:
    page = 0
    slide = pyvips.Image.new_from_file(path, page=page)

    size = slide.width * slide.height
    ### Decide which page to keep : ###
    while size > 1.3e6:
        if page > n:  # On sait jamais avec ces fous...
            size = -1
            page = -1
            break
        page += 1
        slide = pyvips.Image.new_from_file(path, page=page)
        size = slide.width * slide.height
    # To array :
    img = np.ndarray(
        buffer=slide.write_to_memory(),
        dtype=np.uint8,
        shape=(slide.height, slide.width, slide.bands),
    )
    img = Image.fromarray(img)
    return transforms(img)


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    test_dataframe, filenames = create_test_df(device)

    features_names = list(test_dataframe.columns.values)
    # features_names.remove("relapse")
    features_names.remove("filename")
    features_names.remove("latent")
    # features_names.remove("ulceration")
    # features_names.remove("breslow")
    # features_names.remove("lymph")
    # features_names.remove("macro")
    # features_names.remove("epithelial")
    # features_names.remove("neutro")
    age_denominator = 100
    body_site_denominator = 14
    test_dataframe["age"] = test_dataframe["age"].div(age_denominator).round(2)
    test_dataframe["age"] = test_dataframe["age"].div(age_denominator).round(2)
    test_dataframe["body_site"] = test_dataframe["body_site"].div(body_site_denominator).round(2)
    test_dataframe["body_site"] = test_dataframe["body_site"].div(body_site_denominator).round(2)

    model_type = "FC"
    dropout = 0.2
    relapse_only = False
    ImageTabularModel_path = (
        "assets/trained_models/best_on_val_finetune.pth"  # "assets/trained_models/best_on_val.pth"
    )
    # Get preds :
    preds_dict = predict(
        device,
        test_dataframe,
        features_names,
        model_type,
        dropout,
        relapse_only,
        ImageTabularModel_path,
    )
    submission_format = pd.DataFrame.from_dict(preds_dict)
    # save as "submission.csv" in the root folder, where it is expected
    submission_format.to_csv("submission.csv", index=False)


if __name__ == "__main__":
    main()
