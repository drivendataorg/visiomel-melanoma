import argparse
import logging
import time
from datetime import timedelta

import pandas as pd
import PIL

from wsi import filters, slide

START_TIME = time.time()
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger()
PIL.Image.MAX_IMAGE_PIXELS = None


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

BASE_PAGE = 5
slide.RAW_IMG_DIR = RAW_IMG_DIR
slide.BASE_PAGE = BASE_PAGE

meta = pd.read_csv(META_CSV_PATH)


NAMES = [n.split(".")[0] for n in meta.filename.values]

##Generate masks
print("##### MASKING SLIDES #####")

NAMES = [n.split(".")[0] for n in meta.filename.values]
filters.multiprocess_apply_filters_to_images(image_name_list=NAMES)
elapsed = time.time() - START_TIME
print(f"##### TIME MASKING SLIDES: {timedelta(seconds=elapsed)} #####")
