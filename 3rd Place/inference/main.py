import os
from glob import glob

import numpy as np
import pandas as pd
import torch
from joblib import load
from networks_b import ResidualMLP
from sklearn.preprocessing import Normalizer
from tilers import NormalTiler
from tqdm import tqdm


def res_to_level(res):
    """
    gives the level to which downsample given the resolution.
    I want to downsample at 10x in any cases.
    """
    if res < 0.19:
        return 3
    elif res < 0.30:
        return 2
    elif res < 0.6:
        return 1
    else:
        return 2


def TILING(device, tile_encoder, image, n_ensemble, resolution, Nt=5, seed=42, qualitycheck=None):
    """
    Encodes slides by randomly sampling N_ensemble * Nt tiles.
    """
    path_wsi = os.path.join("data", image)
    ext = ".tif"
    normalizer = None  #'macenko'
    path_outputs = os.path.join("./embeddings")
    model_path = tile_encoder
    mask_tolerance = 0.4
    level = res_to_level(resolution)
    size = 224
    tiler = "moco"
    num_workers = 6
    Naug = n_ensemble
    tiler = NormalTiler(
        path_wsi=path_wsi,
        level=level,
        path_outputs=path_outputs,
        size=size,
        device=device,
        tiler=tiler,
        model_path=model_path,
        normalizer=normalizer,
        mask_tolerance=mask_tolerance,
        Naug=Naug,
        Nt=Nt,
        num_workers=num_workers,
        seed=seed,
        qualitycheck=qualitycheck,
    )
    s = tiler.tile_image()
    return s


def load_pretrained_model(model_path):
    ckpt = torch.load(model_path, map_location="cpu")
    config = ckpt["config"]
    config.model.freeze_pooling = 1
    #    if config.model.model_name == 'sparseconvmil':
    #        model = FullSparseConvMIL(config.model)
    #    elif config.model.model_name == 'residualmlp':
    model = ResidualMLP(config.model)
    state_dict = ckpt["state_dict"]
    state_dict = {
        k.replace("backbone.mil.", ""): w
        for k, w in state_dict.items()
        if k.startswith("backbone") and not "classifier" in k
    }
    model.load_state_dict(state_dict, strict=False)
    for name, param in model.named_parameters():
        if "classifier" in name:
            continue
        assert (param == state_dict[name]).all().item(), "Weights not loaded properly"
    print("Loaded the weigths properly.")
    return model, config.dataset.nb_tiles


def prepare_clini_data(clini, stdc):
    clini["age"] = clini["age"].apply(lambda x: int(x[1:3]))
    X = pd.concat([clini[["age", "sex"]], pd.get_dummies(clini["melanoma_history"])], axis=1)
    X = X[["age", "sex", "NO", "YES"]].values
    X = stdc.transform(X)
    return X


def predict_proba(x, C, I):
    """
    x: (n_features,)
    C: (n_classifiers, n_features)
    I: (n_classifiers, )
    """
    z = np.einsum("f,nf->n", x, C) + np.squeeze(I)
    proba = 1 / (1 + np.exp(-z))
    return proba.mean()


def ENCODE_WSI(model, hooker, image, n_ensemble, resolution, device, Nt=5):
    """
    name:
    """
    im = torch.Tensor(
        np.load(
            os.path.join(
                "embeddings_normal",
                f"level_{res_to_level(resolution)}",
                "moco",
                "tiles",
                f"{os.path.splitext(image)[0]}_embeddings.npy",
            )
        )
    )
    xy = torch.Tensor(
        np.load(
            os.path.join(
                "embeddings_normal",
                f"level_{res_to_level(resolution)}",
                "moco",
                "coordinates",
                f"{os.path.splitext(image)[0]}_xy.npy",
            )
        )
    )
    repre = []
    _ = model((im.unsqueeze(0).to(device), xy.unsqueeze(0).to(device)))
    R = hooker.item
    # for rep in range(n_ensemble):
    #    im_s = im[rep*Nt:rep*Nt+Nt, :].unsqueeze(0).to(device)
    #    xy_s = xy[rep*Nt:rep*Nt+Nt, :].unsqueeze(0).to(device)
    #    _ = model((im_s, xy_s))
    #    repre.append(hooker.item)
    # repre = np.vstack([x[0].detach().cpu().numpy() for x in repre])
    # R = repre.mean(0)
    norm = Normalizer()
    R = norm.fit_transform(R.detach().cpu().numpy())
    return np.squeeze(R)


def get_logreg_params(resolution, param_dir):
    dico_centers = {
        0.121399: 0,
        0.194475: 1,
        0.220512: 2,
        0.22649: 3,
        0.226783: 4,
        0.227175: 5,
        0.230213: 6,
        0.242797: 7,
        0.25: 8,
        0.251: 9,
        0.262762: 10,
        0.263223: 11,
        0.264384: 12,
    }  # maps a center towards an int id.
    centers_to_prop = {
        12: 0.9180327868852459,
        3: 0.8333333333333334,
        8: 0.8328767123287671,
        2: 0.6949152542372882,
        4: 0.9,
        10: 0.9,
        0: 0.9222222222222223,
        7: 0.9166666666666666,
        6: 0.8950276243093923,
        9: 0.6666666666666666,
        1: 0.9230769230769231,
        5: 0.75,
        11: 0.6935483870967742,
    }
    if resolution in dico_centers:  # Code degueu berk
        if dico_centers[resolution] in centers_to_prop:
            D = 1 - centers_to_prop[dico_centers[resolution]]
        else:
            D = 0.16
        avails = glob(os.path.join(param_dir, f"*_{dico_centers[resolution]}_*"))
        if len(avails) == 0:
            print("using default coefs")
            C = np.load(os.path.join(param_dir, f"center_default", "coefs.npy"))
            I = np.load(os.path.join(param_dir, f"center_default", "intercepts.npy"))
        elif len(avails) == 1:
            C = np.load(os.path.join(avails[0], "coefs.npy"))
            I = np.load(os.path.join(avails[0], "intercepts.npy"))
        else:
            C = np.load(os.path.join(param_dir, f"center_default", "coefs.npy"))
            I = np.load(os.path.join(param_dir, f"center_default", "intercepts.npy"))
    else:
        print("Using default coefs")
        C = np.load(os.path.join(param_dir, f"center_default", "coefs.npy"))
        I = np.load(os.path.join(param_dir, f"center_default", "intercepts.npy"))
        D = 0.16
    return C, I, D


def main(n_ensemble=400, seed=60):
    # Set a random seed for the entire experiment. Set the seed for torch and numpy
    import time

    TILE_ENCODER = "moco107.pth.tar"
    GIGASSL = "./visiomel_resnetlike_9600.pth"
    PARAMS = "./visiomel_submission_8/"
    QUALITYCHECK = "./QC_submission_8/"
    STDC = load("./standardscale_clini.joblib")
    metadata = pd.read_csv("data/test_metadata.csv")
    X_clini = prepare_clini_data(metadata, STDC)
    filename_to_index = {f: o for o, f in enumerate(metadata["filename"].values)}
    metadata = metadata.set_index("filename")
    resolution = metadata.to_dict()["resolution"]
    tick = time.time()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    wsi_name = [os.path.basename(x) for x in glob("data/*.tif")]
    # Encode wsi tiles
    S = []
    for wsi in wsi_name:
        s = TILING(
            device,
            TILE_ENCODER,
            wsi,
            n_ensemble,
            resolution[wsi],
            5,
            seed,
            qualitycheck=QUALITYCHECK,
        )
        S.append(s)

    tiling_time = time.time() - tick
    tick = time.time()

    # Load Gssl model
    model, _ = load_pretrained_model(GIGASSL)
    model.eval().to(device)

    # hook the last layer of the model
    hooker = type("hooker", (), {"item": None})()

    def hook(m, i, o):
        hooker.item = i[0].cpu()

    handle = model.classifier.register_forward_hook(hook)

    # Encode WSI:
    R = {}
    for wsi in wsi_name:
        try:
            R[wsi] = ENCODE_WSI(model, hooker, wsi, n_ensemble, resolution[wsi], device, 5)
        except:
            print(f"exception at encoding")
            continue

    encoding_time = time.time() - tick
    tick = time.time()

    # Get submission_csv
    sub = pd.read_csv("data/submission_format.csv")

    # Compute probas:
    probas = {}
    for f in sub["filename"].values:
        coefs, intercepts, default_score = get_logreg_params(resolution[f], PARAMS)
        if f in R:
            if coefs.shape[1] > R[f].shape[0]:
                probas[f] = predict_proba(
                    np.hstack([R[f], X_clini[filename_to_index[f]]]), coefs, intercepts
                )
            else:
                print("not using clini")
                probas[f] = predict_proba(R[f], coefs, intercepts)
        else:
            probas[f] = default_score
    sub["relapse"] = sub["filename"].map(probas)
    sub.to_csv("submission.csv", index=False)
    print(f"time elpased tiling: {tiling_time} \n Time elpased giga-ssl encoding {encoding_time}")


if __name__ == "__main__":
    main()
