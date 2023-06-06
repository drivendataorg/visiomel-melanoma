import copy
import gc
import glob
import json
import math
import os
import pickle
import random
import re
import shutil
import sys
import time
import warnings

import cv2
import numpy as np
import pandas as pd
import pytorch_lightning as pl
import torch
import torch.nn.functional as F
import torchvision
import torchvision.transforms as T
from pytorch_lightning import LightningModule, Trainer
from pytorch_lightning.callbacks import EarlyStopping, ModelCheckpoint
from torch import nn
from torch.utils.data import DataLoader, Dataset
from tqdm.auto import tqdm

EXPR = sys.argv[1]
RANDOM_STATE = int(sys.argv[2])
SZ = int(sys.argv[3])
EMBEDDING_SIZE = int(sys.argv[4])  # 512

EXPR_PREDS = f"factors/{EXPR}"
os.makedirs(EXPR_PREDS, exist_ok=True)

MODEL_DIR = f"models/{EXPR}"


def fix_seed(seed):
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    torch.use_deterministic_algorithms(True, warn_only=True)


fix_seed(RANDOM_STATE)
DEVICE = torch.device("cpu")

N_SPLITS = 5

DATA_ROOT = "/code_execution/data/"
meta = pd.read_csv(f"{DATA_ROOT}/test_metadata.csv")
FEATURE_DIR = f"features/{EXPR}"
d_tiles = pd.read_csv(f"{FEATURE_DIR}/tiles.csv")
# d_blur = pd.read_csv(f'{FEATURE_DIR}/blur.csv')
# d_tiles = d_tiles.sort_values(by=['slide','tile_id'])

assert d_tiles.slide.nunique() == len(meta)
d_tiles["filename"] = d_tiles.slide + ".tif"
df = (
    pd.merge(d_tiles, meta, on="filename")
    .sort_values(by=["slide", "tile_id"])
    .reset_index(drop=True)
)

df["path"] = FEATURE_DIR + "/" + df.slide + ".npz"


class VMelTilesDataset(Dataset):
    def __init__(self, df, aug=None):
        self.df = df
        self.aug = aug
        self.slides = df.slide.unique()

    def __len__(self):
        return len(self.slides)

    def __getitem__(self, ix):
        slide = self.slides[ix]
        d = self.df[self.df.slide == slide]  # .set_index('tile_id')
        path = d.path.values[0]
        x = np.load(path)["arr_0"]

        tile_locations = d[["yloc", "xloc"]].values.astype(np.float32)
        tile_locations = torch.from_numpy(tile_locations)

        x = x.astype(np.float32)
        x = torch.from_numpy(x)
        return x, tile_locations  # ,(y,y1,y2)


ds = VMelTilesDataset(df)
x, tile_locations = ds[0]
print("VMelTilesDataset", x.shape, tile_locations.shape, len(ds))

params = {}
params["sparse_conv_n_channels_conv1"] = 128  # 32
params["sparse_conv_n_channels_conv2"] = 128  # 32
params["sparse_map_downsample"] = SZ
params["wsi_embedding_classifier_n_inner_neurons"] = 32
params["batch_size"] = 1


import torchvision.models.resnet as resnet_factory

import sparseconvnet


class SparseConvMIL(nn.Module):
    def __init__(
        self,
        tile_embedder: nn.Module,
        sparse_cnn: nn.Module,
        wsi_embedding_classifier: nn.Module,
        sparse_map_downsample: int,
        tile_coordinates_rotation_augmentation: bool,
        tile_coordinates_flips_augmentation: bool,
        tile_coordinates_resize_augmentation: bool,
    ):
        super().__init__()
        self.tile_embedder = tile_embedder
        self.sparse_cnn = sparse_cnn
        self.wsi_embedding_classifier = wsi_embedding_classifier

        self.sparse_map_downsample = sparse_map_downsample

        # Data augmentation on tiles coordinates
        self.tile_coordinates_rotation_augmentation = tile_coordinates_rotation_augmentation
        self.tile_coordinates_flips_augmentation = tile_coordinates_flips_augmentation
        self.tile_coordinates_resize_augmentation = tile_coordinates_resize_augmentation

    def compute_tile_embeddings(self, tiles):
        """
        Computes concurrent and independent tile embedding with the tile embedder.
        :param tiles: tensor of tiles of expected shape (B_wsi, B_tiles, channels, width, height) with B_wsi equal to
            the number of considered WSI, and B_tiles equal to the number of tiles per considered WSI
        :return: a tensor of tiles embeddings of shape (B_wsi, B_tiles, latent_size)
        """
        # Flatten all tiles across all WSI:
        # (B_wsi, B_tiles, channels, width, height) -> (B_wsi*B_tiles, channels, width, height)
        tiles = tiles.view(tiles.shape[0] * tiles.shape[1], *tiles.shape[2:])
        return self.tile_embedder(tiles)

    @staticmethod
    def post_process_tiles_locations(tiles_locations):
        """
        Reformat the tiles locations into the proper expected format: the sparse-input CNN library sparseconvnet
            expects locations in the format
            [[tile1_loc_x, tile1_loc_y, batch_index_of_tile1],
             [tile2_loc_x, tile2_loc_y, batch_index_of_tile2],
             ...
             [tileN_loc_x, tileN_loc_y, batch_index_of_tileN]]
        :param tiles_locations: locations of sampled tiles with shape (B, n_tiles, 2) with B batch size, n_tiles the
            number of tiles per batch index and the other dimension for both coordinates_x and coordinates_y
        :return: a reformatted tensor of tiles locations with shape (n_tiles, 3)
        """
        device = tiles_locations.device
        reshaped_tiles_locations = tiles_locations.view(
            tiles_locations.shape[0] * tiles_locations.shape[1], -1
        )
        repeated_batch_indexes = torch.tensor(
            [[b] for b in range(tiles_locations.shape[0]) for _ in range(tiles_locations.shape[1])]
        ).to(device)
        return torch.cat((reshaped_tiles_locations, repeated_batch_indexes), dim=1)

    def data_augment_tiles_locations(self, tiles_locations):
        """
        Perform data augmentation of the sparse map of tiles embeddings. First, a matrix of random rotations, flips,
            and resizes is instantiated. Then, a random translation vector is instantiated. The random translation is
            applied on the tiles coordinates, followed by the random rot+flips+resizes.
        :param tiles_locations: matrix of shape (batch_size, n_tiles_per_batch, 2) with tiles coordinates
        :return:
        """
        device = tiles_locations.device

        transform_matrix = torch.eye(2)
        # Random rotations
        if self.tile_coordinates_rotation_augmentation:
            theta = random.uniform(-180.0, 180.0)
            rot_matrix = torch.tensor(
                [[math.cos(theta), -math.sin(theta)], [math.sin(theta), math.cos(theta)]]
            )
            transform_matrix = rot_matrix
        # Random flips
        if self.tile_coordinates_flips_augmentation:
            flip_h = random.choice([-1.0, 1.0])
            flip_v = random.choice([-1.0, 1.0])
            flip_matrix = torch.tensor([[flip_h, 0.0], [0.0, flip_v]])
            transform_matrix = torch.mm(transform_matrix, flip_matrix)
        # Random resizes per axis
        if self.tile_coordinates_resize_augmentation:
            size_factor_h = 0.6 * random.random() + 0.7
            size_factor_v = 0.6 * random.random() + 0.7
            resize_matrix = torch.tensor([[size_factor_h, 0.0], [0.0, size_factor_v]])
            transform_matrix = torch.mm(transform_matrix, resize_matrix)

        # First random translates ids, then apply matrix
        effective_sizes = (
            torch.max(tiles_locations, dim=0)[0] - torch.min(tiles_locations, dim=0)[0]
        )
        random_indexes = [random.randint(0, int(size)) for size in effective_sizes]
        translation_matrix = torch.tensor(random_indexes)
        tiles_locations -= translation_matrix.to(device)
        # Applies transformation
        tiles_locations = torch.mm(tiles_locations.float(), transform_matrix.to(device)).long()

        # Offsets tiles to the leftmost and rightmost
        tiles_locations -= torch.min(tiles_locations, dim=0, keepdim=True)[0]
        return tiles_locations

    def forward(self, tile_embeddings, tiles_original_locations):
        # tile_embeddings = self.compute_tile_embeddings(x)
        tile_embeddings = torch.squeeze(tile_embeddings, 0)
        # print('tile_embeddings', tile_embeddings.shape)
        # print('tiles_original_locations', tiles_original_locations.shape)

        # Builds the sparse map: assign each embedding into its specified location within an empty sparse map
        # First applies downsampling to original tiles locations (see paper)
        tiles_locations = tiles_original_locations / self.sparse_map_downsample
        # Perform data augmentation of the tiles locations, i.e. spatial data augmentation of the sparse map
        # tiles_locations = torch.stack([self.data_augment_tiles_locations(tl) for tl in tiles_locations], dim=0)
        tiles_locations = tiles_locations.to(x.device)
        # Converts tiles locations into the expected format for sparseconvnet
        tiles_locations = self.post_process_tiles_locations(tiles_locations)
        # Instantiates an empty sparse map container for sparseconvnet. Spatial_size is set to the maximum of tiles
        # locations for both axis; mode=4 implies that two embeddings at the same location are averaged elementwise
        input_layer = sparseconvnet.InputLayer(
            dimension=2,
            spatial_size=(
                int(torch.max(tiles_locations[:, 0])) + 1,
                int(torch.max(tiles_locations[:, 1])) + 1,
            ),
            mode=4,
        )
        # Assign each tile embedding to their corresponding post-processed tile location
        sparse_map = input_layer([tiles_locations, tile_embeddings])

        wsi_embedding = self.sparse_cnn(sparse_map)
        wsi_embedding = torch.flatten(wsi_embedding, start_dim=1)

        return self.wsi_embedding_classifier(wsi_embedding)


class SparseAdaptiveAvgPool(nn.AdaptiveAvgPool1d):
    """
    Custom pooling layer that transform a (c, w, h) input sparse tensor into a (c,) output sparse tensor
    """

    def __init__(self, output_size):
        super().__init__(output_size)

    def forward(self, sparse_tensor_input):
        input_features = sparse_tensor_input.features
        input_locations = sparse_tensor_input.get_spatial_locations()

        res = []
        for batch_idx in torch.unique(input_locations[..., 2]):
            pooled = super().forward(
                input_features[input_locations[..., 2] == batch_idx].transpose(0, 1).unsqueeze(0)
            )
            res.append(pooled)

        return torch.cat(res, dim=0)


def get_classifier(input_n_neurons: int, inner_n_neurons: int, n_classes: int):
    """
    Instantiates a ReLU-activated 1-hidden layer MLP.
    :param input_n_neurons: vector size of input data (should be WSI embedding)
    :param inner_n_neurons: number of inner neurons
    :param n_classes: number of output classes
    :return: a Sequential model
    """
    return nn.Sequential(
        nn.Linear(input_n_neurons, inner_n_neurons),
        nn.ReLU(inplace=True),
        nn.Linear(inner_n_neurons, n_classes),
    )


def get_resnet_model(resnet_architecture: str, pretrained: bool):
    """
    Instantiates a ResNet architecture without the finale FC layer.
    :param resnet_architecture: the desired ResNet architecture (e.g. ResNet34 or Wide_Resnet50_2)
    :param pretrained: True to load an architecture pretrained on Imagenet, otherwise standard initialization
    :return: (a Sequential model, number of output channels from the returned model)
    """
    assert resnet_architecture.lower() in resnet_factory.__all__
    resnet_model = getattr(resnet_factory, resnet_architecture.lower())(pretrained, progress=True)
    n_output_channels = resnet_model.fc.in_features
    resnet_model.fc = nn.Sequential()
    return resnet_model, n_output_channels


def get_two_layers_sparse_cnn(
    input_n_channels: int,
    n_out_channels_conv1: int,
    n_out_channels_conv2: int,
    filter_width_conv1: int,
    filter_width_conv2: int,
):
    """
    Instantiates a 2-layers sparse-input ReLU-activated CNN, with a GlobalAveragePooling to reduce spatial
        dimensions to 1.
    :param input_n_channels: vector size of input data (should be the size of each tile embedding)
    :param n_out_channels_conv1: number of output channels for the first convolution
    :param n_out_channels_conv2: number of output channels for the second convolution
    :param filter_width_conv1: width of conv filters for the first convolution
    :param filter_width_conv2: width of conv filters for the second convolution
    :return: a sparseconvnet Sequential model
    """
    return sparseconvnet.Sequential(
        sparseconvnet.SubmanifoldConvolution(
            2, input_n_channels, n_out_channels_conv1, filter_width_conv1, True
        ),
        sparseconvnet.ReLU(),
        sparseconvnet.SubmanifoldConvolution(
            2, n_out_channels_conv1, n_out_channels_conv2, filter_width_conv2, True
        ),
        sparseconvnet.ReLU(),
        SparseAdaptiveAvgPool(1),
    )


def instantiate_sparseconvmil(
    n_out_channels_conv1,
    n_out_channels_conv2,
    filter_width_conv1,
    filter_width_conv2,
    sparse_map_downsample,
    wsi_classifier_input_n_neurons,
    n_classes,
):
    """
    Instantiates a complete SparseConvMIL model:
        1. build a tile embedder (ResNet)
        2. then a pooling function (2-layers sparse-input CNN)
        3. then a classifier (1-hidden layer MLP)
    :param tile_embedder_architecture: resnet architecture of the tile embedder
    :param tile_embedder_pretrained: True to instantiate an Imagenet-pretrained tile embedder
    :param n_out_channels_conv1: number of output channels for the first convolution of the sparse-input pooling
    :param n_out_channels_conv2: number of output channels for the second convolution of the sparse-input pooling
    :param filter_width_conv1: width of conv filters for the first convolution of the sparse-input pooling
    :param filter_width_conv2: width of conv filters for the second convolution of the sparse-input pooling
    :param sparse_map_downsample: downsampling factor applied to the location of the sparse map
    :param wsi_classifier_input_n_neurons: number of inner neurons of the WSI embedding classifier
    :param n_classes: number of output classes
    :return: a Sequential model
    """
    # m = timm.create_model(tile_embedder_architecture,pretrained=tile_embedder_pretrained,num_classes=2,in_chans=3)
    # blocks = [*m.children()]
    # tile_embedder = nn.Sequential(*blocks[:-1])
    tile_embedder = None
    n_output_channels_tile_embedding = EMBEDDING_SIZE
    sparse_input_pooling = get_two_layers_sparse_cnn(
        n_output_channels_tile_embedding,
        n_out_channels_conv1,
        n_out_channels_conv2,
        filter_width_conv1,
        filter_width_conv2,
    )
    wsi_embedding_classifier = get_classifier(
        n_out_channels_conv2, wsi_classifier_input_n_neurons, n_classes
    )

    sparseconvmil_model = SparseConvMIL(
        tile_embedder,
        sparse_input_pooling,
        wsi_embedding_classifier,
        sparse_map_downsample,
        True,
        True,
        True,
    )
    return sparseconvmil_model


NUM_WORKERS = 1
var_configs = [("breslow", 1), ("ulceration", 2)]

var, n_classes = var_configs[0]

sparseconvmil_model = instantiate_sparseconvmil(
    params["sparse_conv_n_channels_conv1"],
    params["sparse_conv_n_channels_conv2"],
    3,
    3,
    params["sparse_map_downsample"],
    params["wsi_embedding_classifier_n_inner_neurons"],
    n_classes=n_classes,
)


class Model(pl.LightningModule):
    def __init__(self, model):
        super().__init__()
        self.model = model

    def forward(self, x: torch.Tensor, locations: torch.Tensor):
        x = self.model(x, locations)
        return x


model = Model(model=sparseconvmil_model)

dfs_preds = []
print(f"GENERATING {EXPR} {var} predictions")

for FOLD in range(N_SPLITS):
    fix_seed(RANDOM_STATE)
    model_path = f"{MODEL_DIR}/f{FOLD}_{var}.pth"
    wts = torch.load(model_path)
    model.load_state_dict(wts)
    model.to(DEVICE)
    model.eval()

    df_test = df.copy()
    ds_test = VMelTilesDataset(df_test)
    dl_test = DataLoader(ds_test, batch_size=1, shuffle=False, num_workers=NUM_WORKERS)
    df_test = df_test.drop_duplicates(subset=["slide"]).copy()

    preds = []
    with torch.no_grad():
        for x, locations in dl_test:
            x = x.to(DEVICE)
            locations = locations.to(DEVICE)
            pr = model(x, locations).cpu()
            preds.append(pr)

    y_pred = torch.concatenate(preds).flatten()
    df_test = df_test[["slide"]].copy()
    df_test[f"{var}_pred"] = y_pred
    dfs_preds.append(df_test)

    # d=df_test.copy();score=skm.mean_squared_error(d[var],d[f'{var}_pred'],squared=False);print(score)
    # d=d[d[var]!=-1];score=skm.mean_squared_error(d[var],d[f'{var}_pred'],squared=False);print(score)

df_preds_breslow = pd.concat(dfs_preds).groupby("slide").mean().reset_index()
df_preds_breslow.to_csv(f"{EXPR_PREDS}/breslow.csv", index=False)
# df_preds_breslow = pd.concat(dfs_preds).sort_values(by='slide').reset_index(drop=True)
# d=df_preds_breslow.copy();score=skm.mean_squared_error(d[var],d[f'{var}_pred'],squared=False);print(score)
# d=d[d[var]!=-1];score=skm.mean_squared_error(d[var],d[f'{var}_pred'],squared=False);print(score)


var, n_classes = var_configs[1]

sparseconvmil_model = instantiate_sparseconvmil(
    params["sparse_conv_n_channels_conv1"],
    params["sparse_conv_n_channels_conv2"],
    3,
    3,
    params["sparse_map_downsample"],
    params["wsi_embedding_classifier_n_inner_neurons"],
    n_classes=n_classes,
)

model = Model(model=sparseconvmil_model)
dfs_preds = []
print(f"GENERATING {EXPR} {var} predictions")
for FOLD in range(N_SPLITS):
    fix_seed(RANDOM_STATE)
    model_path = f"{MODEL_DIR}/f{FOLD}_{var}.pth"
    wts = torch.load(model_path)
    model.load_state_dict(wts)
    model.to(DEVICE)
    model.eval()

    df_test = df.copy()
    ds_test = VMelTilesDataset(df_test)
    dl_test = DataLoader(ds_test, batch_size=1, shuffle=False, num_workers=NUM_WORKERS)
    df_test = df_test.drop_duplicates(subset=["slide"]).copy()

    preds = []
    with torch.no_grad():
        for x, locations in dl_test:
            x = x.to(DEVICE)
            locations = locations.to(DEVICE)
            pr = model(x, locations).cpu()
            preds.append(pr)

    y_pred = torch.concatenate(preds)
    y_pred = F.softmax(y_pred, dim=1).numpy()[:, 1]

    df_test = df_test[["slide"]].copy()
    df_test[f"{var}_pred"] = y_pred
    dfs_preds.append(df_test)

    # d=df_test.copy();score=skm.log_loss(d[var],d[f'{var}_pred'],labels=[0,1]);print(score)
    # d=d[d[var]!=-1];score=skm.log_loss(d[var],d[f'{var}_pred'],labels=[0,1]);print(score)

# df_preds_ulceration = pd.concat(dfs_preds).sort_values(by='slide').reset_index(drop=True)
# d=df_preds_ulceration.copy();score=skm.log_loss(d[var],d[f'{var}_pred'],labels=[0,1]);print(score)
# d=d[d[var]!=-1];score=skm.log_loss(d[var],d[f'{var}_pred'],labels=[0,1]);print(score)
df_preds_ulceration = pd.concat(dfs_preds).groupby("slide").mean().reset_index()
df_preds_ulceration.to_csv(f"{EXPR_PREDS}/ulceration.csv", index=False)
