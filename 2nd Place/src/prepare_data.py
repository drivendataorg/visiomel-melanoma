####
####
#### Usage : python3 download_data.py (--min_size 1e6)
####
#### Note: data/ folder should exist and contain csv clinical data sheet.
####


import argparse
import os
import subprocess

import pandas as pd
import pyvips
from sklearn.model_selection import train_test_split
from tqdm import tqdm

REGION = "eu"  # "eu", "us", or "as"


def data_split(data_path, root):
    #####
    if not os.path.exists(os.path.join(root, "data/images/")):
        os.makedirs(os.path.join(root, "data/images/"))

    ##### Create train Val split
    metadata = pd.read_csv(os.path.join(data_path, "train_metadata.csv"))
    relapse = pd.read_csv(os.path.join(data_path, "train_labels.csv"))

    metadata = metadata.merge(relapse, on="filename")
    train_data, val_data = train_test_split(
        metadata, test_size=0.3, stratify=metadata.relapse, random_state=12000
    )

    train_data.to_csv(os.path.join(data_path, "train_dataframe.csv"), index=False)
    val_data.to_csv(os.path.join(data_path, "val_dataframe.csv"), index=False)

    return metadata


def main(args):
    # Generate train test split
    root = os.getcwd()
    data_path = os.path.join(root, "data/")
    if args.data_path is not None:
        data_path = args.data_path
    metadata = data_split(data_path, root)

    ##### Download tiffs and save appropriate png
    list_names = metadata.filename.to_numpy()
    list_buckets = metadata[f"{REGION}_tif_url"].to_numpy()

    for name, bucket in tqdm(
        zip(list_names, list_buckets), total=len(list_names)
    ):  # , position=0,leave=True):
        if os.path.exists(os.path.join(root, "data", "images", name.replace(".tif", ".png"))):
            continue

        # If path not given, download tiff
        if args.data_path is None:
            if args.verbose:
                subprocess.run(
                    ["aws", "s3", "cp", bucket, root + "/data/images/", "--no-sign-request"]
                )
            else:
                subprocess.run(
                    ["aws", "s3", "cp", bucket, root + "/data/images/", "--no-sign-request"],
                    capture_output=True,
                    text=True,
                )

        # Num pages:
        slide = pyvips.Image.new_from_file(os.path.join(data_path, "images/", name))
        n = slide.get_n_pages()
        # Height and width of page 0:
        page = 0
        slide = pyvips.Image.new_from_file(os.path.join(data_path, "images/", name), page=page)
        size = slide.width * slide.height
        ### Decide which page to keep : ###
        while size > args.min_size:
            if page > n:
                size = -1
                page = -1
                break
            page += 1
            slide = pyvips.Image.new_from_file(os.path.join(data_path, "images/", name), page=page)
            size = slide.width * slide.height
        # Save page as png :
        slide.write_to_file(os.path.join(data_path, "images/", name[:-4] + ".png"))
        # Delete tiff file if dwonloaded:
        if args.data_path is None:
            os.remove(os.path.join(data_path, "images/", name))


if __name__ == "__main__":
    ##### Parse arguments
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--data_path",
        type=str,
        default=None,
        help='If "None" will download data from aws , else will use given path',
    )
    parser.add_argument("--min_size", type=float, default=1.3e6, help="Minimum resolution to keep")
    parser.add_argument(
        "-v", "--verbose", type=bool, default=False, help="Wether to display aws outputs"
    )

    args = parser.parse_args()

    main(args)
