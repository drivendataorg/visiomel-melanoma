import os
from pathlib import Path

import numpy as np
import torch
from sklearn.preprocessing import Normalizer
from tqdm import tqdm

from src.networks import ResidualMLP
from src.tilers import main as main_tiling
from src.utils import prepare_clini_data, res_to_level


def ENCODE_TILES(path, device, resolution, ext="tif", out=None):
    """
    Encode the data using the pre-trained models.
    """
    assert os.path.exists("qualitycheck/logreg/coefs.npy")
    TILE_ENCODER = "models/moco.pth.tar"
    wsi_names = [os.path.basename(x) for x in Path(path).glob(f"*.{ext}")]
    print("Encoding the tiles.")
    for wsi in tqdm(wsi_names):
        main_tiling(
            path,
            device,
            TILE_ENCODER,
            wsi,
            256,
            resolution[wsi],
            5,
            seed=10,
            qualitycheck="./qualitycheck/logreg",
            path_outputs=out,
            ext=ext,
        )


def encode_wsi(model, hook, image: str, resolution: float, device, out):
    im = torch.Tensor(
        np.load(
            os.path.join(
                out,
                f"level_{res_to_level(resolution)}",
                "moco",
                "tiles",
                f"{os.path.splitext(image)[0]}_embeddings.npy",
            )
        )
    )

    if im.shape[0] == 0:
        warnings.warn(f"No tiles for {im_path}.")
        return

    xy = torch.Tensor(
        np.load(
            os.path.join(
                out,
                f"level_{res_to_level(resolution)}",
                "moco",
                "coordinates",
                f"{os.path.splitext(image)[0]}_xy.npy",
            )
        )
    )
    _ = model((im.unsqueeze(0).to(device), xy.unsqueeze(0).to(device)))
    R = hook.item
    norm = Normalizer()
    R = norm.fit_transform(R.detach().cpu().numpy())
    return np.squeeze(R)


def load_pretrained_model(model_path):
    ckpt = torch.load(model_path, map_location="cpu")
    config = ckpt["config"]
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
    return model


def get_model_and_hook(path):
    model = load_pretrained_model(path)
    # hook the last layer of the model
    hooker = type("hooker", (), {"item": None})()

    def hook(m, i, o):
        hooker.item = i[0].cpu()

    model.classifier.register_forward_hook(hook)
    return model, hooker


def ENCODE_ALL_WSI(metadata, model, device, outputs, wsis):
    """
    Encode the data using the pre-trained models.
    """
    model, hook = get_model_and_hook(model)
    model.eval().to(device)
    resolution = metadata.set_index("filename").to_dict()["resolution"]
    wsi_names = wsis
    X_clini, f_to_i = prepare_clini_data(metadata, train=True)
    R = {}
    empty = []
    print("Encoding the slides.")
    for wsi in wsi_names:
        X = encode_wsi(model, hook, wsi, resolution[wsi], device, out=outputs)
        if X is not None:
            R[wsi] = np.hstack([X, X_clini[f_to_i[wsi]]])
        else:
            empty.append(wsi)

    outputs = Path(outputs / "gigassl")
    outputs.mkdir(exist_ok=True)
    np.save(outputs / "embeddings.npy", np.vstack([x for x in R.values()]))
    np.save(outputs / "filenames.npy", np.array([x for x in R.keys()]))
    return R, empty
