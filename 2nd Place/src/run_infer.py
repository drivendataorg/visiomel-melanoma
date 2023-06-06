import argparse
import os

import numpy as np
import pandas as pd
import pyvips
import torch
from PIL import Image
from torchvision.transforms import Compose, Normalize, ToTensor
from utils.dataframe_process import clean_df

# Local libraries
from utils.models import DummyModel, ImageTabularModel

################################################################


def predict(
    device,
    test_dataframe,
    features_names,
    model_type,
    dropout,
    relapse_only,
    ImageTabularModel_path,
    age_trick=True,
):
    model = ImageTabularModel(
        len(features_names), model_type, dropout=dropout, relapse_only=relapse_only
    )
    model.load_state_dict(torch.load(ImageTabularModel_path, map_location=device))  # TODO
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

        if age_trick and (sample.age > 0.9):
            pred = min(pred, max(0.05, pred - 0.27))

        preds_dict["filename"].append(sample["filename"])
        preds_dict["relapse"].append(pred)

    return preds_dict


def create_test_df(device, args):
    test_dataframe = pd.read_csv(os.path.join(args.data_path, "test_metadata.csv"))

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
    dummy_model_path = args.tritrain_path
    dummy_model = DummyModel(model_path=dummy_model_path)
    dummy_model = dummy_model.to(device)
    dummy_model = dummy_model.eval()

    ##### Construct Clinical data + image embedding
    logits_dict = {"filename": [], "latent": []}
    for filename in filenames:
        logits_dict["filename"].append(filename)
        img = get_image(os.path.join(args.data_path, filename))  # os.join(data_root,img_path))
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
    # TODO.. old or new mapping?????
    # body_site_mapping={'nan': -1, 'trunk': 0, 'arm': 1, 'seat': 2, 'head': 3, 'neck': 3, 'head/neck': 3,
    #                 'face': 4, 'trunc': 5, 'leg': 6, 'forearm': 7, 'upper limb': 8, 'shoulder': 8, 'upper limb/shoulder': 8,
    #                 'lower limb': 9, 'hip': 9, 'lower limb/hip': 9, 'hand': 10, 'toe': 11,
    #                 'foot': 12, 'nail': 12, 'hand/foot/nail': 12, 'thigh': 13, 'sole': 14, 'finger': 15, 'scalp': 16}
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


def main(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    test_dataframe, filenames = create_test_df(device, args)

    # Get name of Tabular features used.
    features_names = list(test_dataframe.columns.values)
    features_names.remove("filename")
    features_names.remove("latent")

    # values to normalize tabular features with
    age_denominator = 100  # See mapping to understand value
    body_site_denominator = 14  # See mapping to understand value

    test_dataframe["age"] = test_dataframe["age"].div(age_denominator).round(2)
    test_dataframe["age"] = test_dataframe["age"].div(age_denominator).round(2)
    test_dataframe["body_site"] = test_dataframe["body_site"].div(body_site_denominator).round(2)
    test_dataframe["body_site"] = test_dataframe["body_site"].div(body_site_denominator).round(2)

    # Model Hyperparams
    dropout = 0.2  # Since in eval mode, not used here
    # Get preds :
    preds_dict = predict(
        device,
        test_dataframe,
        features_names,
        args.model_type,
        dropout,
        args.relapse_only,
        args.model_path,
        age_trick=args.age_trick_bool,
    )
    submission_format = pd.DataFrame.from_dict(preds_dict)
    # save as "submission.csv" in the root folder, where it is expected
    submission_format.to_csv("submission.csv", index=False)


if __name__ == "__main__":
    ##### Parse arguments
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--model_path",
        type=str,
        default="models/after_finetune.pth",
        help="Path to trained model used as classifier.",
    )
    parser.add_argument(
        "--tritrain_path",
        type=str,
        default="models/tritrain.pth",
        help="Path to trained resNet model used as feature extractor.",
    )
    parser.add_argument("--model_type", type=str, default="FC", help='Tpe of model ("FC" or "CNN")')
    parser.add_argument(
        "--relapse_only", type=bool, default=False, help="Describes model outputs (see model class)"
    )

    parser.add_argument(
        "--data_path",
        type=str,
        default="data/",
        help="Path to data, must contain metada.csv and tiff format images.",
    )
    parser.add_argument(
        "--age_trick_bool", type=bool, default=True, help="use age trick boolean at inference"
    )
    args = parser.parse_args()

    main(args)
