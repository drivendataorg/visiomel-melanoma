# Solution - Visiomel

Username: trislaz

## Summary

This solution is based on the Giga-SSL representation (https://arxiv.org/abs/2212.03273) and can be divided into three primary components:

1. A WSI preprocessing step that includes tissue segmentation and tile filtering

2. The calculation of pre-trained embeddings for tiles (using a pre-trained MoCo model) and slides (employing Giga-SSL)

3. The linear classifier training step, which produces the final submission output

One of the significant benefits of this approach is its rapid execution, as the inference on the complete test and validation steps can be accomplished in under two hours.

## Whole Slide Image Preprocessing

The whole slide image (WSI) preprocessing step consists of two parts: tissue segmentation and tile filtering.

### Tissue Segmentation

Tissue segmentation, which is the creation of a binary mask that localizes the tissue, is performed on a downsampled version of the WSI at level -2 (with the level referring to the downsampling levels of OpenSlide). To avoid the inclusion of slide artifacts (such as black regions), bands with a width equal to 1/10th of the image are initially excluded from the binary masks. Subsequently, an Otsu thresholding is applied, followed by a morphological opening and closing, to obtain the binary mask. A grid is superimposed on the binary mask, and tiles with less than half of their surface area within the tissue are discarded.

### Tile Filtering

Since a large portion of the tissue is not of interest (e.g., normal epidermis, dermis, adipose tissue, etc.), we filter out the irrelevant tiles. To accomplish this, we randomly sampled 5 tiles per Whole Slide Image (WSI) within the tissue, resulting in a total of 6,696 tiles. Subsequently, we assigned a single label (0 or 1) to each tile based on its relevance, which required 2 hours of manual annotation. We then trained a linear classifier (balanced logistic regression) using the tile encodings. We have provided a small script, `label_tiles.py`, to facilitate tile labeling if desired. By default, the tiling process generates an `examples` folder containing 5 randomly sampled tiles per WSI, which can be used to create a labeled dataset. We have also included the image dataset and corresponding labels used in our study, downloaded with the `downloads.sh` script.

## Computation of Embeddings

### Tile Embeddings

As detailed in the [Giga-SSL paper](https://arxiv.org/abs/2212.03273), we encode the tiles using the first four blocks of a ResNet18 pretrained with MoCo on tiles extracted from The Cancer Genome Atlas (link).

### Slide Embeddings

Tile embeddings are then processed by an aggregator pretrained using the Giga-SSL algorithm. However, unlike the original publication, and due to difficulty in installing the SparseConvNet dependency on the competition Docker image, we utilized a non-spatially aware architecture.

This architecture is entirely composed of Residual MLP layers, mimicking the design described in the [Giga-SSL paper](https://arxiv.org/abs/2212.03273).

## Final Classifier Training

We conduct L2-normalization on the WSI representations and leverage them to train a linear classifier on the training dataset. With 40-fold cross-validation, we train 40 logistic regression models. The final prediction is the average of these 40 model predictions.


# Setup

1. Download the pre-train models and qualitycheck images

```bash
bash downloads.sh
```

2. Install (Anaconda)[https://docs.anaconda.com/free/anaconda/install/index.html]
3. Create the environment:

```bash
conda env create -f requirements.yml
conda activate visiomel3
```

You may want to install the pytorch version appropriate to your cuda version (if so, change it in `requirements.yml`).

4. Put your raw training WSI in the `data/train`folder and your raw test WSI in the `data/test` folder.
5. Put the label csv in `data/train_labels.csv`: this csv contains one `filename` column, containing the slide names (including the extension) and one `label` column, containing the discrete labels.
6. Put the metadata csv in `data/train_metadata.csv`. Same for the test set. The metadata contains 4 columns: `filename`, `age`, `sex` and `melanoma_history`.

Here is a summary of the repo structure:

```diff

./
├── README.md
├── data
│   ├── train
│   │   ├── 0001.svs
│   │   ├── ...svs
│   │   └── 1000.svs
│   ├── test
│   │   ├── 1001.svs
│   │   ├── ...svs
│   │   └── 2000.svs
│   ├── train_labels.csv
│   ├── train_metadata.csv
│   └── test_metadata.csv
├── qualitycheck
│   ├── images
│   │   ├── {x}.png
│   │   ├── ...png
│   │   └── {xn}.png
│   ├── logreg  @@ (Produced by train.py. Needed for predict.py) @@
│   │   ├── coefs.npy
│   │   └── intercepts.npy
│   └── labels.csv
├── models
│   ├── moco.pth.tar
│   ├── gigassl.pth.tar
│   ├── linear_classifier_coefs.npy @@ (Produced by train.py. Needed for predict.py) @@
│   └── linear_classifier_intercepts.npy @@ (Produced by train.py. Needed for predict.py) @@

```

# Hardware

We used a single V100 GPU for the training and inference steps. For both training and inference, the Giga-SSL encoding takes ~10-15 secs per WSI at 10x magnification, on a single V100 GPU, or 1h30 for 500 WSI.

Training and inference of the linear classifiers take <1sec on a CPU.

# Run training

After having set up the data as described in [setup](#setup), run the following command:

```bash
python train.py
```

# Run inference

After having set up the data as described in [setup](#setup), putting the inference data under `data/test`, run the following command:

```bash
python predict.py
```

If no training has been performed, the script will automatically download the pretrained models and perform the inference.

The `inference` directory contains the code that was run in the [competition runtime repo](https://github.com/drivendataorg/visiomel-melanoma-runtime/) to produce the winning submission.
