import copy
import gc
import glob
import os
import random
import re
import sys
import time
import warnings

import cv2
import numpy as np
import pandas as pd
import PIL
import pyvips

DATA_ROOT = "/code_execution/data/"
ss = pd.read_csv(f"{DATA_ROOT}/submission_format.csv")
meta = pd.read_csv(f"{DATA_ROOT}/test_metadata.csv")

PAGE = 5
IMG_DIR_256 = f"images/256"
IMG_DIR_384 = f"images/384"
os.makedirs(IMG_DIR_256, exist_ok=True)
os.makedirs(IMG_DIR_384, exist_ok=True)

CONFIG = [(256, IMG_DIR_256), (384, IMG_DIR_384)]


def preproc(row):
    filepath = f"{DATA_ROOT}/{row.filename}"
    slide = pyvips.Image.new_from_file(filepath, page=PAGE)
    # slide = pyvips.Image.new_from_file(filepath,page=PAGE,access='sequential')
    img = slide.numpy()
    img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
    for SZ, IMG_DIR in CONFIG:
        img1 = cv2.resize(img, (SZ, SZ), interpolation=cv2.INTER_AREA)
        out_path = f"{IMG_DIR}/{row.filename}.png"
        cv2.imwrite(out_path, img1, [int(cv2.IMWRITE_PNG_COMPRESSION), 0])


print("EXTRACTING p5 DATASETS")
for _, row in meta.iterrows():
    preproc(row)
