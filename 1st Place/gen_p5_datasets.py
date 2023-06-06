import argparse
import os
import time
from datetime import timedelta

import cv2
import pandas as pd
import pyvips

parser = argparse.ArgumentParser()
parser.add_argument("--stage", help="train or test", required=True)
parser.add_argument(
    "--raw_img_dir",
    help="Directory containing pyramidal whole slide tif files",
    required=True,
)
parser.add_argument(
    "--meta_csv_path", help="Path to the train or test metadata csv file", required=True
)
args = parser.parse_args()
STAGE = args.stage  #'train',test'
RAW_IMG_DIR = args.raw_img_dir
META_CSV_PATH = args.meta_csv_path

DATA_ROOT = f"./workspace/data/{STAGE}"
meta = pd.read_csv(META_CSV_PATH)

PAGE = 5
IMG_DIR_256 = f"{DATA_ROOT}/resized/256"
IMG_DIR_384 = f"{DATA_ROOT}/resized/384"
os.makedirs(IMG_DIR_256, exist_ok=True)
os.makedirs(IMG_DIR_384, exist_ok=True)

CONFIG = [(256, IMG_DIR_256), (384, IMG_DIR_384)]


def preproc(row):
    filepath = f"{RAW_IMG_DIR}/{row.filename}"
    slide = pyvips.Image.new_from_file(filepath, page=PAGE)
    img = slide.numpy()
    img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
    for SZ, IMG_DIR in CONFIG:
        img1 = cv2.resize(img, (SZ, SZ), interpolation=cv2.INTER_AREA)
        out_path = f"{IMG_DIR}/{row.filename}.png"
        cv2.imwrite(out_path, img1, [int(cv2.IMWRITE_PNG_COMPRESSION), 0])


START_TIME = time.time()
print("##### EXTRACTING PAGE 5 DATASETS #####")
for _, row in meta.iterrows():
    preproc(row)

elapsed = time.time() - START_TIME
print(f"##### DONE EXTRACTING PAGE 5 DATASETS TOTAL TIME: {timedelta(seconds=elapsed)} #####")
