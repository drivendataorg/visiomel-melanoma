"""
Main script to train a bag of classifiers.
2 main steps:
    - Encode the data using the pre-trained models.
    - Train the linear classifiers.
"""
import os
from pathlib import Path

import numpy as np
import pandas as pd
import sklearn
import torch
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold

from src.encode import ENCODE_ALL_WSI, ENCODE_TILES
from src.tilers import TileEncoderFromFolder


def TRAIN_QUALITYCHECK():
    """
    train the logreg for quality check
    """
    root = Path("./qualitycheck")
    out = root / "logreg"
    out.mkdir(exist_ok=True)
    tiler = TileEncoderFromFolder(root / "images", "models/moco.pth.tar", None, out)
    E, Ids = tiler.encode()
    labels = pd.read_csv(root / "labels.csv")
    ids_keep = [o for o, x in enumerate(Ids) if x in labels["filename"].values]
    E, Ids = E[ids_keep], [Ids[o] for o in ids_keep]
    labels = labels.set_index("filename").to_dict()["label"]
    labels = [labels[x] for x in Ids]

    logreg = LogisticRegression(C=7, max_iter=10000, class_weight="balanced")
    logreg.fit(E, labels)
    C = logreg.coef_
    I = logreg.intercept_
    np.save(out / "coefs.npy", C)
    np.save(out / "intercepts.npy", I)


def TRAIN_CLASSIFIERS(R, df_labels, metric=["balanced_accuracy_score", "roc_auc_score"], cv=2):
    """
    Train the linear classifiers.

    Parameters
    ----------
    R : dict = {wsi_name: np.array}
        Dictionary with the embeddings of each WSI.

    df_labels : pd.DataFrame
        Dataframe with the labels.

    metric : str or [str] = ['roc_auc_score', 'f1', 'accuracy_score' ... all sklearn.metrics]
    """
    df_labels = df_labels.set_index("filename").to_dict()["relapse"]
    y = np.array([df_labels[x] for x in R.keys()])
    X = np.vstack([x for x in R.values()])
    skf = StratifiedKFold(n_splits=cv, shuffle=True)
    C = []
    I = []
    if type(metric) == str:
        metric = [metric]
    metrics = {m: [] for m in metric}
    for i, (train_index, test_index) in enumerate(skf.split(X, y)):
        X_train, X_test = X[train_index], X[test_index]
        y_train, y_test = y[train_index], y[test_index]
        logreg = LogisticRegression(C=7, max_iter=10000, class_weight="balanced")
        logreg.fit(X_train, y_train)
        y_pred = logreg.predict(X_test)
        y_proba = logreg.predict_proba(X_test)
        for m in metric:
            if m == "roc_auc_score":
                metrics[m].append(sklearn.metrics.roc_auc_score(y_test, y_proba[:, 1]))
            else:
                metrics[m].append(getattr(sklearn.metrics, m)(y_test, y_pred))
        C.append(logreg.coef_)
        I.append(logreg.intercept_)
    for m in metric:
        print(f"{m} validation mean/std: {np.mean(metrics[m])}, {np.std(metrics[m])}")
    C = np.vstack(C)
    I = np.vstack(I)
    np.save("models/linear_classifier_coefs.npy", C)
    np.save("models/linear_classifier_intercepts.npy", I)


def main():
    train_path = "data/train"
    df_labels = pd.read_csv("data/train_labels.csv")
    metadata_train = pd.read_csv("data/train_metadata.csv")
    resolution = metadata_train.set_index("filename").to_dict()["resolution"]
    device = "cuda" if torch.cuda.is_available() else "cpu"
    ext = "tif"
    out = Path("train_embeddings")
    out.mkdir(exist_ok=True)
    model_giga = "models/gigassl.pth"

    wsi_names = [os.path.basename(x) for x in Path(train_path).glob(f"*.{ext}")]

    # Start with training the qualitycheck logreg.
    TRAIN_QUALITYCHECK()
    # Encode all the tiles from the WSIs
    ENCODE_TILES(path="data/train", device=device, resolution=resolution, ext=ext, out=out)
    # Encode all the WSIs
    R, _ = ENCODE_ALL_WSI(
        metadata=metadata_train, model=model_giga, device=device, outputs=out, wsis=wsi_names
    )
    # Train the linear classifiers
    TRAIN_CLASSIFIERS(R, df_labels)


if __name__ == "__main__":
    main()
