import os
from glob import glob

import numpy as np
import openslide
import torch
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
from torchvision.models import resnet18
from torchvision.transforms import ToTensor
from tqdm import tqdm

from src.utils import (
    get_image,
    make_auto_mask,
    patch_sampling,
    predict_proba,
    res_to_level,
)


def load_moco_model(moco_weights_path, model_name="resnet18"):
    """
    Loads a resnet with moco pretrained weights.
    Args:
        moco_weights_path (str): Path to moco weights.
        model_name (str): Name of the model.
    Returns:
        model (torch.nn.Module): Model with moco weights.
    """
    model = eval(model_name)()
    checkpoint = torch.load(moco_weights_path, map_location="cpu")
    state_dict = checkpoint["state_dict"]
    for k in list(state_dict.keys()):
        if k.startswith("module.encoder_q") and not k.startswith("module.encoder_q.fc"):
            state_dict[k[len("module.encoder_q.") :]] = state_dict[k]
        del state_dict[k]
    model.load_state_dict(state_dict, strict=False)
    for name, param in model.named_parameters():
        if "fc" in name:
            continue
        assert (param == state_dict[name]).all().item(), "Weights not loaded properly"
    return model


class TileEncoderFromFolder:
    """
    input: a folder of tiles, a moco model, normalizer, path_outputs.
    """

    def __init__(self, folder, model, normalizer, path_outputs):
        self.folder = folder
        self.model = load_moco_model(model, model_name="resnet18")
        self.normalizer = self._get_normalizer(normalizer)
        self.preprocess = self._get_transforms()
        self.files = glob(os.path.join(folder, "*.png"))
        self.path_outputs = path_outputs
        os.makedirs(path_outputs, exist_ok=True)
        self.device = "cuda" if torch.cuda.is_available() else "cpu"

    def _get_normalizer(self, normalizer):
        return None

    def _get_transforms(self, shared=True):
        t = transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        return t

    def encode(self):
        """
        Forward pass of the WSI.

        :param model: torch.nn.Module, model to use for the forward pass. This implementation works with resnet18.
        :param param_tiles: list, output of the patch_sampling.
        :param preprocess: kornia.augmentation.AugmentationSequential, preprocessing to apply to the tiles.
        :param Naug: int, number of augmentations per WSI.
        :param Nt: int, number of tiles per augmentation.
        """
        model = self.model
        hook = [0]

        def hook_l3(m, i, o):
            hook[0] = o

        model.layer3.register_forward_hook(hook_l3)
        model.eval()
        model.to(self.device)
        embeddings = []
        ids = []
        for o, path in tqdm(enumerate(self.files)):
            I = np.array(Image.open(path))[:, :, 0:3]
            if self.normalizer:
                I = self.normalizer.transform(I)
            I = ToTensor()(I).to(self.device).unsqueeze(0)
            I = self.preprocess(I)
            with torch.no_grad():
                O = model(I).squeeze().cpu().numpy()
            embeddings.append(torch.mean(hook[0], dim=(2, 3)).squeeze().cpu().numpy())
            ids.append(os.path.basename(path))

        return np.vstack(embeddings), np.stack(ids)


class SharedAugTiler:
    """
    Tiles a WSI and encodes its tiles with a pretrained model.
    Special case with shared augmentations:
    - Naug batches of Nt tiles are sampled from the WSI.
    - Each batch is augmented with the same augmentations.
    - The model is applied to each batch.
    - The features are saved under /path_outputs/level_{level}/{tiler}/tiles/{name_wsi}/{aug_id}.npy
    - The coordinates of the tiles are saved under /path_outputs/level_{level}/{tiler}/coordinates/{name_wsi}/{aug_id}.npy
    """

    def __init__(
        self,
        path_wsi,
        level,
        path_outputs,
        size,
        device,
        tiler,
        model_path=None,
        normalizer=None,
        mask_tolerance=0.5,
        Naug=50,
        Nt=256,
        num_workers=5,
        seed=42,
    ):
        self.Naug = Naug
        self.seed = seed
        self.Nt = Nt
        self.num_workers = num_workers
        self.level = level
        self.device = device
        self.size = (size, size)
        self.path_wsi = path_wsi
        self.model_path = model_path
        self.tiler = tiler
        self.normalize = normalizer is not None
        self.normalizer = self._get_normalizer(normalizer)
        self.name_wsi, self.ext_wsi = os.path.splitext(os.path.basename(self.path_wsi))
        self.outpath = self._set_out_path(
            os.path.join(path_outputs, f"level_{level}", f"{tiler}"), self.name_wsi
        )
        self.slide = openslide.open_slide(self.path_wsi)
        self.mask_tolerance = mask_tolerance
        self.mask_function = lambda x: make_auto_mask(x, mask_level=-1)
        self.preprocess = self._get_transforms(aug=True, imagenet=True)

    def _set_out_path(self, path_outputs, name_wsi):
        outpath = {}
        outpath["tiles"] = os.path.join(path_outputs, "tiles", name_wsi)
        outpath["coordinates"] = os.path.join(path_outputs, "coordinates", name_wsi)
        outpath["example"] = os.path.join(path_outputs, "examples")
        for key in outpath.keys():
            if not os.path.exists(outpath[key]):
                os.makedirs(outpath[key])
        return outpath

    def _write_examples(self, param_tiles):
        sample = np.random.randint(
            1, len(param_tiles), 5
        )  # np.array(param_tiles)[np.random.randint(1, len(param_tiles), 5)]
        for i in sample:
            s = param_tiles[i]
            image = get_image(slide=self.slide, para=s, numpy=False).convert("RGB")
            image = np.array(image)
            if self.normalizer:
                image = self.normalizer.transform(image)
            image = Image.fromarray(image.astype(np.uint8))
            image.save(
                os.path.join(
                    self.outpath["examples"],
                    "{}_i{}_{}_{}_{}_{}_{}.png".format(
                        self.name_wsi, i, s[0], s[1], s[2], s[3], s[4]
                    ),
                )
            )

    def _get_transforms(self, aug=True, imagenet=True, shared=True):
        t = transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        return t

    def _get_normalizer(self, normalizer):
        return None

    def tile_image(self):
        """tile_image.
        Main function of the class. Tiles the WSI and writes the outputs.
        WSI of origin is specified when initializing TileImage.
        """
        sample = patch_sampling(
            slide=self.slide,
            mask_level=-1,
            mask_function=self.mask_function,
            analyse_level=self.level,
            patch_size=self.size,
            mask_tolerance=self.mask_tolerance,
        )
        param_tiles = sample["params"]
        self._write_examples(param_tiles)
        model = self._get_model(self.tiler)
        model.eval()
        model.to(self.device)
        size = self._forward_pass_WSI(model, param_tiles, self.Naug, self.Nt)
        return size

    def _forward_pass_WSI(self, model, param_tiles, Naug, Nt):
        """
        Forward pass of the WSI.

        :param model: torch.nn.Module, model to use for the forward pass. This implementation works with resnet18.
        :param param_tiles: list, output of the patch_sampling.
        :param preprocess: kornia.augmentation.AugmentationSequential, preprocessing to apply to the tiles.
        :param Naug: int, number of augmentations per WSI.
        :param Nt: int, number of tiles per augmentation.
        """
        hook = [0]

        def hook_l3(m, i, o):
            hook[0] = o

        model.layer3.register_forward_hook(hook_l3)
        model.eval()
        model.to(self.device)
        data = WSI_dataset(param_tiles, self.slide, self.normalizer, seed=self.seed)
        data_loader = DataLoader(data, batch_size=Nt, num_workers=self.num_workers)

        for aug, (batch, paras) in enumerate(data_loader):
            paras = np.array(paras)
            batch = batch.to(self.device)
            batch = self.preprocess(batch)
            with torch.no_grad():
                batch = model(batch).squeeze().cpu().numpy()
            embeddings = torch.mean(hook[0], dim=(2, 3)).squeeze().cpu().numpy()
            np.save(os.path.join(self.outpath["tiles"], f"{aug}.npy"), embeddings)
            np.save(os.path.join(self.outpath["coordinates"], f"{aug}.npy"), paras)

    def _get_model(self, tiler):
        """
        Returns a torch.nn.Module object.
        """
        if tiler == "imagenet":
            return resnet18(pretrained=True)
        elif tiler == "moco":
            return load_moco_model(self.model_path, model_name="resnet18")
        else:
            raise ValueError("Tiler not implemented")


class NormalTiler(SharedAugTiler):
    def __init__(
        self,
        path_wsi,
        level,
        path_outputs,
        size,
        device,
        tiler,
        model_path=None,
        normalizer=None,
        mask_tolerance=0.5,
        Naug=50,
        Nt=256,
        num_workers=5,
        seed=42,
        qualitycheck=None,
    ):
        path_outputs = path_outputs
        super().__init__(
            path_wsi,
            level,
            path_outputs,
            size,
            device,
            tiler,
            model_path,
            normalizer,
            mask_tolerance,
            Naug,
            Nt,
            num_workers,
        )
        self.preprocess = self._get_transforms(aug=False, imagenet=True)
        self.qualitycheck = qualitycheck
        if qualitycheck:
            self.coefq = np.load(os.path.join(qualitycheck, "coefs.npy"))
            self.interceptq = np.load(os.path.join(qualitycheck, "intercepts.npy"))

    def _forward_pass_WSI(self, model, param_tiles, Naug, Nt):
        """
        Forward pass of the WSI.

        :param model: torch.nn.Module, model to use for the forward pass. This implementation works with resnet18.
        :param param_tiles: list, output of the patch_sampling.
        :param Naug: int, number of augmentations per WSI.
        :param Nt: int, number of tiles per augmentation.
        """
        hook = [0]

        def hook_l3(m, i, o):
            hook[0] = o

        model.layer3.register_forward_hook(hook_l3)
        model.eval()
        model.to(self.device)
        data = WSI_dataset(param_tiles, self.slide, self.normalizer, seed=self.seed)
        data_loader = DataLoader(
            data, batch_size=Naug, num_workers=self.num_workers, drop_last=False
        )

        E = np.zeros((len(data), 256))
        P = np.zeros((len(data), 2))
        for nt, (batch, paras) in enumerate(data_loader):
            paras = np.array(paras)
            batch = batch.to(self.device)
            batch = self.preprocess(batch)
            with torch.no_grad():
                batch = model(batch).squeeze().cpu().numpy()
            embeddings = torch.mean(hook[0], dim=(2, 3)).squeeze().cpu().numpy()
            E[nt * Naug : min(len(data), (nt + 1) * Naug)] = embeddings
            P[nt * Naug : min(len(data), (nt + 1) * Naug)] = paras

        if self.qualitycheck:
            E, mask = filter_tiles(E, self.coefq, self.interceptq)
            P = P[mask]

        np.save(os.path.join(self.outpath["tiles"], f"{self.name_wsi}_embeddings.npy"), E)
        np.save(os.path.join(self.outpath["coordinates"], f"{self.name_wsi}_xy.npy"), P)
        return len(E)

    def _set_out_path(self, path_outputs, name_wsi):
        outpath = {}
        outpath["tiles"] = os.path.join(path_outputs, "tiles")
        outpath["coordinates"] = os.path.join(path_outputs, "coordinates")
        outpath["examples"] = os.path.join(path_outputs, "examples")
        for key in outpath.keys():
            if not os.path.exists(outpath[key]):
                os.makedirs(outpath[key])
        return outpath


def filter_tiles(mat, c, i, threshold=0.5):
    """
    Filters out tiles that are not likely to be cancerous
    """
    proba = predict_proba(mat, c, i).mean(axis=1)
    mask = proba > threshold
    return mat[mask, :], mask


class WSI_dataset(Dataset):
    def __init__(self, params, slide, name_wsi, normalizer=None, shuffle=False, seed=42):
        self.params = params
        if shuffle:
            np.random.shuffle(self.params)
        self.normalizer = normalizer
        self.slide = slide
        self.name_wsi = name_wsi

    def __len__(self):
        return len(self.params)

    def __getitem__(self, idx):
        params = self.params[idx]
        image = get_image(slide=self.slide, para=params, numpy=True)
        if self.normalizer:
            image = self.normalizer.transform(image)
        image = ToTensor()(image)
        return image, np.array(params[:2])


def main(
    wsi_directory,
    device,
    tile_encoder,
    wsi_name,
    n_ensemble,
    wsi_resolution,
    Nt=5,
    seed=42,
    qualitycheck=None,
    path_outputs=".",
    ext="tif",
):
    path_wsi = os.path.join(wsi_directory, wsi_name)
    ext = f".{ext}"
    normalizer = None
    path_outputs = path_outputs
    model_path = tile_encoder
    mask_tolerance = 0.4
    level = res_to_level(wsi_resolution)
    size = 224
    tiler = "moco"
    num_workers = 5
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
