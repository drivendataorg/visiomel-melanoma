# VisioMel Challenge: Predicting Melanoma Relapse

## Setup Environment

Install gcc, libvips

```bash
sudo apt-get update -y --fix-missing
sudo apt install -y build-essential libvips
```

Install anaconda (https://www.anaconda.com) 

`cd` to `solution` directory and create and activate conda environment

```bash
conda env create -n vmel --file environment.yml`
conda activate vmel
```

Install [sparseconvnet](https://github.com/facebookresearch/SparseConvNet)

`python setup.py develop && python examples/hello-world.py`

## Directory structure

To simplify the pipeline, please setup the train and test datasets in `./workspace` directory as shown

<pre>
./
├── train.ipynb
├── test.ipynb
├── workspace
│   ├── data
│   │   ├── test
│   │   │   ├── submission_format.csv
│   │   │   ├── test_metadata.csv
│   │   │   └── whole_slides
│   │   │       ├── example.tif
│   │   └── train
│   │       ├── train_labels.csv
│   │       ├── train_metadata.csv
│   │       └── whole_slides
│   │           ├── example.tif
</pre>

Files generated during training will be written to `./workspace` directory. 

## Train

Run [train.ipynb](train.ipynb) to train. I've included links to download the pretrained models in the notebooks. The next sections summarize the training steps and model outputs.

### `train_swin`

Train SWIN model to predict relapse (pretrain with breslow target, finetune on relapse) on 5 folds, resulting in 5 models per tile size:

```
workspace/models/swin256
├── swinv2_base_window12to16_192to256_22kft1k_f0_p1.pth
├── swinv2_base_window12to16_192to256_22kft1k_f1_p1.pth
├── swinv2_base_window12to16_192to256_22kft1k_f2_p1.pth
├── swinv2_base_window12to16_192to256_22kft1k_f3_p1.pth
└── swinv2_base_window12to16_192to256_22kft1k_f4_p1.pth

workspace/models/swin384
├── swin_large_patch4_window12_384_f0_p1.pth
├── swin_large_patch4_window12_384_f1_p1.pth
├── swin_large_patch4_window12_384_f2_p1.pth
├── swin_large_patch4_window12_384_f3_p1.pth
└── swin_large_patch4_window12_384_f4_p1.pth
```

### `train_factors`

Trains SparseConv MIL model to predict breslow and ulceration on 5 folds, resulting in 10 models per tile size:

```
workspace/models/expr_40_2_320
├── f0_breslow.pth
├── f0_ulceration.pth
├── f1_breslow.pth
├── f1_ulceration.pth
├── f2_breslow.pth
├── f2_ulceration.pth
├── f3_breslow.pth
├── f3_ulceration.pth
├── f4_breslow.pth
└── f4_ulceration.pth

workspace/models/expr_56_3_224
├── f0_breslow.pth
├── f0_ulceration.pth
├── f1_breslow.pth
├── f1_ulceration.pth
├── f2_breslow.pth
├── f2_ulceration.pth
├── f3_breslow.pth
├── f3_ulceration.pth
├── f4_breslow.pth
└── f4_ulceration.pth
```

### `train_mlp`

Trains a multilayer perceptron to predict relapse from the clinical metadata and outputs of the SWIN and factor models. Trains with 4 random seeds and 5 folds, resulting in 20 models per tile size:

```
workspace/models/expr_40_2_320/mlp
├── r1948_f0.pth
├── r1948_f1.pth
├── ...
├── r888_f3.pth
└── r888_f4.pth
(20 files)

workspace/models/expr_56_3_224/mlp
├── r1948_f0.pth
├── r1948_f1.pth
├── ...
├── r888_f3.pth
└── r888_f4.pth
(20 files)
```

# Test

Run [test.ipynb](test.ipynb) to generate test set predictions.

The `inference` directory contains the code that was run in the [competition runtime repo](https://github.com/drivendataorg/visiomel-melanoma-runtime/) to produce the winning submission.
