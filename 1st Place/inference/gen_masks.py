import numpy as np, pandas as pd
import glob,os,sys,shutil,gc,copy,math, warnings,random,string,logging,multiprocessing,subprocess,time
import PIL,pyvips

import skimage.io as sk 
from datetime import timedelta

from wsi import slide,filters,tiles,util

START_TIME = time.time()
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger()
PIL.Image.MAX_IMAGE_PIXELS = None

DATA_ROOT = '/code_execution/data/'
RAW_IMG_DIR = DATA_ROOT #'./'

BASE_PAGE = 5
STAGE = 'test'
slide.RAW_IMG_DIR = RAW_IMG_DIR
slide.BASE_PAGE = BASE_PAGE

meta = pd.read_csv(f"{DATA_ROOT}/test_metadata.csv")

    
RANDOM_STATE = 41
def fix_seed(seed):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
fix_seed(RANDOM_STATE)


DATA_ROOT = '/code_execution/data/'
meta = pd.read_csv(f"{DATA_ROOT}/test_metadata.csv")
NAMES = [n.split('.')[0] for n in meta.filename.values]

##Generate masks
logger.info('************** GENERATING MASKS *********************')

NAMES = [n.split('.')[0] for n in meta.filename.values]
filters.multiprocess_apply_filters_to_images(image_name_list=NAMES)
elapsed = time.time() - START_TIME
gc.collect()
