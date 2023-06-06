import torch
from sklearn.metrics import accuracy_score
from tqdm import tqdm


def clean_state_dict(state_dict):
    new_state_dict = {}
    for key, value in state_dict.items():
        if key.startswith("module."):
            new_state_dict[key[7:]] = value
        else:
            new_state_dict[key] = value
    return new_state_dict


def train_epoch(
    device,
    model,
    optimizer,
    relapse_only,
    loader,
    loss_breslow,
    loss_ulceration,
    loss_relapse,
    alpha,
    beta,
    gamma,
    min_loss_r,
    patches=False,
    clip=None,
    verbose=True,
):
    train_bool = model.training

    train_type = "train"
    if not train_bool:
        train_type = "val"
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

    running_loss_relapse_bce = 0.0  # REMOVE

    for i, (item) in tqdm(
        enumerate(loader), desc=f"{train_type}...", total=len(loader), disable=not verbose
    ):
        if patches:
            (
                img_embedding,
                patch_embedding,
                tabular,
                target_relapse,
                target_breslow,
                target_ulceration,
            ) = item
            patch_embedding = patch_embedding.to(device)
        else:
            img_embedding, tabular, target_relapse, target_breslow, target_ulceration = item
        img_embedding = img_embedding.to(device)
        tabular = tabular.to(device)
        target_relapse = target_relapse.to(device)
        target_breslow = target_breslow.type(torch.LongTensor).to(device)
        target_ulceration = target_ulceration.to(device)
        if patches:
            outputs = model(img_embedding, patch_embedding, tabular)
        else:
            outputs = model(img_embedding, tabular)

        if relapse_only:
            relapse_pred = outputs

            loss_r = loss_relapse(relapse_pred.squeeze(1), target_relapse)
            loss_b = torch.zeros_like(loss_r)
            loss_u = torch.zeros_like(loss_r)

            loss = loss_r  # (alpha * loss_b) + (beta * loss_u) + (gamma * loss_r)
        else:
            breslow, ulceration, relapse_pred = outputs

            loss_b = loss_breslow(breslow.squeeze(1), target_breslow)
            loss_u = loss_ulceration(ulceration.squeeze(1), target_ulceration)
            loss_r = loss_relapse(relapse_pred.squeeze(1), target_relapse)

            loss = (alpha * loss_b) + (beta * loss_u) + (gamma * loss_r)

        if train_bool:
            optimizer.zero_grad()
            loss.backward()
            if clip is not None:
                torch.nn.utils.clip_grad_norm_(model.parameters(), clip)
            optimizer.step()

        with torch.no_grad():
            running_loss_relapse_bce += torch.nn.BCELoss()(
                relapse_pred.squeeze(1), target_relapse
            ).item() * img_embedding.size(
                0
            )  # REMOVE
        running_loss += loss.item() * img_embedding.size(0)
        running_loss_b += loss_b.item() * img_embedding.size(0)
        running_loss_u += loss_u.item() * img_embedding.size(0)
        running_loss_r += loss_r.item() * img_embedding.size(0)

        if not relapse_only:
            labels_u.extend(target_ulceration.cpu().numpy())
            predictions_u.extend(ulceration.squeeze(1).detach().cpu().numpy().round())
            labels_b.extend(target_breslow.cpu().numpy())
            predictions_b.extend(breslow.detach().argmax(dim=1).cpu().numpy())
        labels_r.extend(target_relapse.cpu().numpy())
        predictions_r.extend(relapse_pred.squeeze(1).detach().cpu().numpy().round())

    train_loss = running_loss / len(loader.dataset)
    train_loss_b = running_loss_b / len(loader.dataset)
    train_loss_u = running_loss_u / len(loader.dataset)
    train_loss_r = running_loss_r / len(loader.dataset)

    loss_relapse_bce = running_loss_relapse_bce / len(loader.dataset)
    if relapse_only:
        train_acc_b = 0
        train_acc_u = 0
    else:
        train_acc_b = accuracy_score(labels_b, predictions_b)
        train_acc_u = accuracy_score(labels_u, predictions_u)

    train_acc_r = accuracy_score(labels_r, predictions_r)
    if verbose:
        print(
            "Loss: {:.4f} - Loss breslow: {:.4f} - Loss ulceration: {:.4f} - Loss relapse: {:.4f}".format(
                train_loss, train_loss_b, train_loss_u, train_loss_r
            )
        )
        print(
            "Acc breslow: {:.4f} - Acc ulceration: {:.4f} - Acc relapse: {:.4f}".format(
                train_acc_b, train_acc_u, train_acc_r
            )
        )

    # Update min loss relapse found:
    if train_loss_r < min_loss_r:
        min_loss_r = train_loss_r
    return (
        train_loss,
        labels_r,
        labels_u,
        labels_b,
        predictions_r,
        predictions_u,
        predictions_b,
        min_loss_r,
        train_loss_r,
    )
