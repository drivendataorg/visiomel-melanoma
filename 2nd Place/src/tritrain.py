# Standard libraries
import os
import random
from collections import defaultdict

import numpy as np

# Third-party libraries
import pandas as pd
import torch
import torch.nn as nn
import torchvision
import tqdm
from sklearn.metrics import accuracy_score
from torch.utils.data import DataLoader
from torchvision.transforms import (
    ColorJitter,
    Compose,
    GaussianBlur,
    Normalize,
    RandomCrop,
    RandomHorizontalFlip,
    RandomRotation,
    RandomVerticalFlip,
    ToTensor,
)
from utils.datasets import BreslowUlcerationRelapseDataset
from utils.imputers import ulceration_breslow_from_relapse_imputer

# Local libraries
from utils.models import BreslowUlcerationRelapseModel


NUM_DATA_WORKERS = 30

root = os.getcwd()

transfo_train = Compose(
    [
        RandomHorizontalFlip(),
        RandomVerticalFlip(),
        RandomRotation(
            90, interpolation=torchvision.transforms.InterpolationMode.BILINEAR, expand=True
        ),
        RandomCrop((1024, 1024), pad_if_needed=True),
        ColorJitter(),
        GaussianBlur(3),
        ToTensor(),
        Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ]
)


train_dataframe = pd.read_csv(os.path.join(root, "data", "train_dataframe.csv"))
val_dataframe = pd.read_csv(os.path.join(root, "data", "val_dataframe.csv"))
train_dataframe = train_dataframe[["filename", "relapse", "ulceration", "breslow"]]
val_dataframe = val_dataframe[["filename", "relapse", "ulceration", "breslow"]]
train_dataframe = ulceration_breslow_from_relapse_imputer(train_dataframe)
val_dataframe = ulceration_breslow_from_relapse_imputer(val_dataframe)


def set_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


set_seed(0)

counts = train_dataframe.relapse.value_counts().to_dict()
weights = {k: 1 / v for k, v in counts.items()}

train_dataframe["weights"] = train_dataframe.relapse.apply(lambda x: weights[x])
path = "data"
train_dataset = BreslowUlcerationRelapseDataset(
    train_dataframe, os.path.join(path, "images"), transforms=transfo_train
)
val_dataset = BreslowUlcerationRelapseDataset(val_dataframe, os.path.join(path, "images"), None)

train_loader = DataLoader(
    train_dataset, batch_size=8, num_workers=NUM_DATA_WORKERS, shuffle=True, pin_memory=True
)
val_loader = DataLoader(val_dataset, batch_size=1, num_workers=NUM_DATA_WORKERS, shuffle=False, pin_memory=True)

model = BreslowUlcerationRelapseModel()
alpha = 1
beta = 1
gamma = 1
loss_breslow = torch.nn.CrossEntropyLoss()
loss_ulceration = torch.nn.BCELoss()
loss_relapse = torch.nn.BCELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=1e-3, weight_decay=3e-6)
scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
    optimizer, patience=10, verbose=True, factor=0.1
)
best_loss = 10000
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = nn.DataParallel(model)
model.to(device)
metrics = defaultdict(list)
epochs = 30
for epoch in range(epochs):
    print("Epoch {}/{}".format(epoch + 1, epochs))
    metrics["lr"].append(optimizer.param_groups[0]["lr"])
    model.train()
    labels_r = []
    predictions_r = []
    labels_u = []
    predictions_u = []
    labels_b = []
    predictions_b = []
    running_loss = 0.0
    running_loss_b = 0.0
    running_loss_u = 0.0
    running_loss_r = 0.0
    for i, item in tqdm.tqdm(enumerate(train_loader), desc="Training...", total=len(train_loader)):
        slides = item["slide"].to(device)
        breslow = item["breslow"].to(device)
        ulceration = item["ulceration"].to(device)
        relapse = item["relapse"].to(device)
        outputs = model(slides)
        loss_b = loss_breslow(outputs[0].squeeze(1), breslow)
        loss_u = loss_ulceration(outputs[1].squeeze(1), ulceration)
        loss_r = loss_relapse(outputs[2].squeeze(1), relapse)
        loss = (alpha * loss_b) + (beta * loss_u) + (gamma * loss_r)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        running_loss += loss.item() * slides.size(0)
        running_loss_b += loss_b.item() * slides.size(0)
        running_loss_u += loss_u.item() * slides.size(0)
        running_loss_r += loss_r.item() * slides.size(0)
        labels_u.extend(ulceration.cpu().numpy())
        predictions_u.extend(outputs[1].squeeze(1).detach().cpu().numpy().round())
        labels_r.extend(relapse.cpu().numpy())
        predictions_r.extend(outputs[2].squeeze(1).detach().cpu().numpy().round())
        labels_b.extend(breslow.cpu().numpy())
        predictions_b.extend(outputs[0].detach().argmax(dim=1).cpu().numpy())

    train_loss = running_loss / len(train_loader.dataset)
    train_loss_b = running_loss_b / len(train_loader.dataset)
    train_loss_u = running_loss_u / len(train_loader.dataset)
    train_loss_r = running_loss_r / len(train_loader.dataset)
    train_acc_b = accuracy_score(labels_b, predictions_b)
    train_acc_u = accuracy_score(labels_u, predictions_u)
    train_acc_r = accuracy_score(labels_r, predictions_r)
    print(
        "Train loss: {:.4f} - Train loss breslow: {:.4f} - Train loss ulceration: {:.4f} - Train loss relapse: {:.4f}".format(
            train_loss, train_loss_b, train_loss_u, train_loss_r
        )
    )
    print(
        "Train acc breslow: {:.4f} - Train acc ulceration: {:.4f} - Train acc relapse: {:.4f}".format(
            train_acc_b, train_acc_u, train_acc_r
        )
    )
    metrics["train/loss"].append(train_loss)
    metrics["train/acc_breslow"].append(train_acc_b)
    metrics["train/acc_ulceration"].append(train_acc_u)
    metrics["train/acc_relapse"].append(train_acc_r)
    model.eval()
    labels_b = []
    predictions_b = []
    labels_u = []
    predictions_u = []
    labels_r = []
    predictions_r = []
    running_loss = 0.0
    running_loss_b = 0.0
    running_loss_u = 0.0
    running_loss_r = 0.0
    with torch.no_grad():
        for i, item in tqdm.tqdm(
            enumerate(val_loader), desc="Validation...", total=len(val_loader)
        ):
            slides = item["slide"].to(device)
            breslow = item["breslow"].to(device)
            ulceration = item["ulceration"].to(device)
            relapse = item["relapse"].to(device)
            outputs = model(slides)
            loss_b = loss_breslow(outputs[0].squeeze(1), breslow)
            loss_u = loss_ulceration(outputs[1].squeeze(1), ulceration)
            loss_r = loss_relapse(outputs[2].squeeze(1), relapse)
            loss = (alpha * loss_b) + (beta * loss_u) + (gamma * loss_r)
            running_loss += loss.item() * slides.size(0)
            running_loss_b += loss_b.item() * slides.size(0)
            running_loss_u += loss_u.item() * slides.size(0)
            running_loss_r += loss_r.item() * slides.size(0)
            labels_b.extend(breslow.cpu().numpy())
            predictions_b.extend(outputs[0].argmax(dim=1).cpu().numpy())
            labels_u.extend(ulceration.cpu().numpy())
            predictions_u.extend(outputs[1].squeeze(1).cpu().numpy().round())
            labels_r.extend(relapse.cpu().numpy())
            predictions_r.extend(outputs[2].squeeze(1).cpu().numpy().round())

    val_loss = running_loss / len(val_loader.dataset)
    val_loss_b = running_loss_b / len(val_loader.dataset)
    val_loss_u = running_loss_u / len(val_loader.dataset)
    val_loss_r = running_loss_r / len(val_loader.dataset)
    val_acc_b = accuracy_score(labels_b, predictions_b)
    val_acc_u = accuracy_score(labels_u, predictions_u)
    val_acc_r = accuracy_score(labels_r, predictions_r)
    metrics["val/loss"].append(val_loss)
    metrics["val/acc_breslow"].append(val_acc_b)
    metrics["val/acc_ulceration"].append(val_acc_u)
    metrics["val/acc_relapse"].append(val_acc_r)
    metrics["val/loss_breslow"].append(val_loss_b)
    metrics["val/loss_ulceration"].append(val_loss_u)
    metrics["val/loss_relapse"].append(val_loss_r)

    scheduler.step(val_loss)
    print(
        "Val loss: {:.4f} - Val loss breslow: {:.4f} - Val loss ulceration: {:.4f} - Val loss relapse: {:.4f}".format(
            val_loss, val_loss_b, val_loss_u, val_loss_r
        )
    )
    print(
        "Val acc breslow: {:.4f} - Val acc ulceration: {:.4f} - Val acc relapse: {:.4f}".format(
            val_acc_b, val_acc_u, val_acc_r
        )
    )
    if val_loss_r < best_loss:
        best_loss = val_loss_r
        torch.save(model.module.state_dict(), os.path.join(root, "models", "tritrain.pth"))
        print("Best model saved at epoch {}".format(epoch + 1))
