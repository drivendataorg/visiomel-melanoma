import glob
import os
import time
from datetime import timedelta

import pandas as pd

MAIN_START_TIME = time.time()

os.system("python update.py")
print("##### START GENERATING TILES #####")

os.system("python gen_masks.py")
os.system("python gen_tiles_56_p3_224.py")
os.system("python gen_tiles_40_p2_320.py")
elapsed = time.time() - MAIN_START_TIME
print("##### TIME GENERATING TILES: {timedelta(seconds=elapsed)} #####")

print("##### START GENERATING EMBEDDINGS #####")
START_TIME = time.time()
# base_sz,page,sz,BS (128 for 512)
os.system("python gen_embeddings_res18.py 56 3 224 256")
os.system("python gen_embeddings_res18.py 40 2 320 256")
elapsed = time.time() - START_TIME
elapsed_main = time.time() - MAIN_START_TIME
print(
    f"##### TIME GENERATING EMBEDDINGS: {timedelta(seconds=elapsed)} MAIN TIME: {timedelta(seconds=elapsed_main)} #####"
)

START_TIME = time.time()
print("##### START GENERATING PREDS MIL #####")
# seed,sz/dsamp,embedding size
os.system("python gen_factors.py res18_imnet_56_3_224 41 224 512")
os.system("python gen_factors.py res18_imnet_40_2_320 320 320 512")
os.system("python gen_preds_mil.py res18_imnet_56_3_224")
os.system("python gen_preds_mil.py res18_imnet_40_2_320")
elapsed = time.time() - START_TIME
elapsed_main = time.time() - MAIN_START_TIME
print(
    f"##### TIME GENERATING PREDS MIL: {timedelta(seconds=elapsed)} MAIN TIME: {timedelta(seconds=elapsed_main)} #####"
)

START_TIME = time.time()
print("##### START GENERATING SWIN DATASETS #####")
os.system("python gen_p5_datasets.py")
elapsed = time.time() - START_TIME
elapsed_main = time.time() - MAIN_START_TIME
print(
    f"##### TIME GENERATING SWIN DATASETS: {timedelta(seconds=elapsed)} MAIN TIME: {timedelta(seconds=elapsed_main)} #####"
)


START_TIME = time.time()
print("##### START GENERATING PREDS SWIN #####")
os.system("python gen_preds_swin.py swin_large_patch4_window12_384 384 888")
os.system("python gen_preds_swin.py swinv2_base_window12to16_192to256_22kft1k 256 256")

elapsed = time.time() - START_TIME
elapsed_main = time.time() - MAIN_START_TIME
print(
    f"##### TIME GENERATING PREDS MIL: {timedelta(seconds=elapsed)} MAIN TIME: {timedelta(seconds=elapsed_main)} #####"
)


DATA_ROOT = "/code_execution/data/"
meta = pd.read_csv(f"{DATA_ROOT}/test_metadata.csv")
ss = pd.read_csv(f"{DATA_ROOT}/submission_format.csv")

EXPR = "res18_imnet_56_3_224"
d0 = pd.concat(pd.read_csv(p) for p in glob.glob(f"preds/{EXPR}/pred_*.csv"))
d0 = d0.groupby("filename").mean().reset_index()

EXPR = "res18_imnet_40_2_320"
d1 = pd.concat(pd.read_csv(p) for p in glob.glob(f"preds/{EXPR}/pred_*.csv"))
d1 = d1.groupby("filename").mean().reset_index()

pred_swin256 = pd.read_csv("preds/swin/pred_swin256.csv")  #
pred_swin384 = pd.read_csv("preds/swin/pred_swin384.csv")  #

df_sub = pd.concat([d0, d1, pred_swin256, pred_swin384]).groupby("filename").mean()
df_sub = df_sub.reindex(ss.filename)
print("SUBMITTING")
df_sub.to_csv("submission.csv")
