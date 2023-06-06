import argparse
import glob

import pandas as pd

parser = argparse.ArgumentParser()
parser.add_argument(
    "--submission_format_csv_path", help="Path to the submission_format csv file", required=True
)
args = parser.parse_args()
STAGE = "test"
SUBMISSION_FORMAT_CSV_PATH = args.submission_format_csv_path

ROOT_PRED_DIR = f"./workspace/preds/{STAGE}/"

ss = pd.read_csv(SUBMISSION_FORMAT_CSV_PATH)

EXPR = "expr_56_3_224"
d0 = pd.concat(pd.read_csv(p) for p in glob.glob(f"{ROOT_PRED_DIR}/{EXPR}/pred_*.csv"))
d0 = d0.groupby("filename").mean().reset_index()

EXPR = "expr_40_2_320"
d1 = pd.concat(pd.read_csv(p) for p in glob.glob(f"{ROOT_PRED_DIR}/{EXPR}/pred_*.csv"))
d1 = d1.groupby("filename").mean().reset_index()

pred_swin256 = pd.read_csv(f"{ROOT_PRED_DIR}/pred_swin256.csv")
pred_swin384 = pd.read_csv(f"{ROOT_PRED_DIR}/pred_swin384.csv")

df_sub = pd.concat([d0, d1, pred_swin256, pred_swin384]).groupby("filename").mean()
df_sub = df_sub.reindex(ss.filename)
path = "submission.csv"
print(f"writting submission to {path}")
df_sub.to_csv(path)
