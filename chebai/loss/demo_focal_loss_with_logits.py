import torch
from torch import nn


def bce_with_logits(logits, target):
    # PyTorch BCEWithLogitsLoss uses the log-sum-exp trick for numerical stability
    bce_loss = nn.BCEWithLogitsLoss(reduction="none")
    return bce_loss(logits, target)


def bce_naive(logits, target):
    # this is how I would naively implement BCE loss
    sigmoid_logits = torch.sigmoid(logits)
    bce_loss = -(
        target * torch.log(sigmoid_logits)
        + (1 - target) * torch.log(1 - sigmoid_logits)
    )
    return bce_loss


def focal_loss_with_logits(logits, target, gamma=2.0):
    # Focal loss rephrased in terms of BCE with logits
    focal_loss = torch.exp(-bce_with_logits(-logits, target) * gamma) * bce_with_logits(
        logits, target
    )
    return focal_loss


def focal_loss_chebai(logits, target, gamma=2.0):
    # the current chebai implementation does use the BCEwithLogitsLoss, but computes the focal part without log-sum-exp trick
    p_t = torch.where(target == 1, torch.sigmoid(logits), 1 - torch.sigmoid(logits))
    focal_loss = ((1 - p_t) ** gamma) * bce_with_logits(logits, target)
    return focal_loss


def focal_loss_naive(logits, target, gamma=2.0):
    # naive focal loss implementation
    sigmoid_logits = torch.sigmoid(logits)
    p_t = torch.where(target == 1, sigmoid_logits, 1 - sigmoid_logits)
    bce_loss = -(
        target * torch.log(sigmoid_logits)
        + (1 - target) * torch.log(1 - sigmoid_logits)
    )
    focal_loss = ((1 - p_t) ** gamma) * bce_loss
    return focal_loss


if __name__ == "__main__":
    dtype = torch.float32
    y = torch.tensor([1], dtype=dtype)  # label
    print(
        "| x      | Pytorch BCE (stable)   | Naive BCE              | Focal loss (stable)    | Focal loss (Chebai)    | Naive Focal loss       |\n| ------ | ---------------------- | ---------------------- | ---------------------- | ---------------------- | ---------------------- |"
    )
    x = torch.tensor([4.0], dtype=dtype)  # logit
    while x < 200:
        loss = bce_with_logits(x, y)
        loss_naive = bce_naive(x, y)
        floss = focal_loss_with_logits(x, y)
        floss_chebai = focal_loss_chebai(x, y)
        floss_naive = focal_loss_naive(x, y)
        # print as markdown table row
        print(
            f"| {x.item()} | {loss.item()} | {loss_naive.item()} | {floss.item()} | {floss_chebai.item()} | {floss_naive.item()} |"
        )
        x *= 2

    x = torch.tensor([-4.0], dtype=dtype)
    while x > -200:
        loss = bce_with_logits(x, y)
        loss_naive = bce_naive(x, y)
        floss = focal_loss_with_logits(x, y)
        floss_chebai = focal_loss_chebai(x, y)
        floss_naive = focal_loss_naive(x, y)
        # print as markdown table row
        print(
            f"| {x.item()} | {loss.item()} | {loss_naive.item()} | {floss.item()} | {floss_chebai.item()} | {floss_naive.item()} |"
        )
        x *= 2
