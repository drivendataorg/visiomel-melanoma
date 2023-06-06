import os
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from src.encode import ENCODE_ALL_WSI, ENCODE_TILES
from src.utils import predict_proba


def main():
    test_path = "data/test"
    metadata = pd.read_csv("data/test_metadata.csv")
    resolution = metadata.set_index("filename").to_dict()["resolution"]
    device = "cuda" if torch.cuda.is_available() else "cpu"
    ext = "tif"
    default_proba = 0.16
    out = Path("test_embeddings")
    out.mkdir(exist_ok=True)
    model_giga = "models/gigassl.pth.tar"

    wsi_names = [os.path.basename(x) for x in Path(test_path).glob(f"*.{ext}")]
    if len(wsi_names) == 0:
        raise ValueError(f"No {ext} files found in {test_path}")

    # Encode all the tiles from the WSIs
    ENCODE_TILES(path=test_path, device=device, resolution=resolution, ext=ext, out=out)
    # Encode all the WSIs
    R, empty = ENCODE_ALL_WSI(
        metadata=metadata, model=model_giga, device=device, outputs=out, wsis=wsi_names
    )
    X = np.vstack([x for x in R.values()])
    ids = [x for x in R.keys()] + empty
    C = np.load("models/linear_classifier_coefs.npy")
    I = np.squeeze(np.load("models/linear_classifier_intercepts.npy"))

    y_proba = list(predict_proba(X, C, I).mean(axis=1)) + [default_proba] * len(empty)
    df = pd.DataFrame({"filename": ids, "prediction": y_proba})
    df.to_csv("prediction.csv", index=False)


if __name__ == "__main__":
    main()
