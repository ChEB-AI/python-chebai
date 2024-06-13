import logging
import os
import sys

import pytorch_lightning as pl
import torch
import torch.nn.functional as F
from pytorch_lightning import loggers as pl_loggers
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.metrics import F1
from sklearn.metrics import f1_score
from torch import nn
from torch_geometric import nn as tgnn
from torch_geometric.data import DataLoader

from data import ClassificationData, JCIClassificationData

logging.getLogger("pysmiles").setLevel(logging.CRITICAL)


class PartOfNet(pl.LightningModule):
    def __init__(self, in_length, loops=10):
        super().__init__()
        self.loops = loops
        self.left_graph_net = tgnn.GATConv(in_length, in_length)
        self.right_graph_net = tgnn.GATConv(in_length, in_length)
        self.attention = nn.Linear(in_length, 1)
        self.global_attention = tgnn.GlobalAttention(self.attention)
        self.output_net = nn.Sequential(
            nn.Linear(2 * in_length, 2 * in_length),
            nn.Linear(2 * in_length, in_length),
            nn.Linear(in_length, 500),
        )
        self.f1 = F1(1, threshold=0.5)

    def _execute(self, batch, batch_idx):
        pred = self(batch)
        loss = F.binary_cross_entropy_with_logits(pred, batch.label)
        f1 = self.f1(batch.label, torch.sigmoid(pred))
        return loss, f1

    def training_step(self, *args, **kwargs):
        loss, f1 = self._execute(*args, **kwargs)
        self.log(
            "train_loss",
            loss.detach().item(),
            on_step=True,
            on_epoch=True,
            prog_bar=True,
            logger=True,
        )
        self.log(
            "train_f1",
            f1.item(),
            on_step=True,
            on_epoch=True,
            prog_bar=True,
            logger=True,
        )
        return loss

    def validation_step(self, *args, **kwargs):
        with torch.no_grad():
            loss, f1 = self._execute(*args, **kwargs)
            self.log(
                "val_loss",
                loss.detach().item(),
                on_step=True,
                on_epoch=True,
                prog_bar=True,
                logger=True,
            )
            self.log(
                "val_f1",
                f1.item(),
                on_step=True,
                on_epoch=True,
                prog_bar=True,
                logger=True,
            )
            return loss

    def forward(self, x):
        a = self.left_graph_net(x.x_s, x.edge_index_s.long())
        b = self.right_graph_net(x.x_t, x.edge_index_t.long())
        return self.output_net(
            torch.cat(
                [
                    self.global_attention(a, x.x_s_batch),
                    self.global_attention(b, x.x_t_batch),
                ],
                dim=1,
            )
        )

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters())
        return optimizer


class JCINet(pl.LightningModule):
    def __init__(self, in_length, hidden_length, num_classes, loops=10):
        super().__init__()
        self.loops = loops

        self.node_net = nn.Sequential(
            nn.Linear(self.loops * in_length, hidden_length), nn.ReLU()
        )
        self.embedding = torch.nn.Embedding(800, in_length)
        self.left_graph_net = tgnn.GATConv(in_length, in_length, dropout=0.1)
        self.final_graph_net = tgnn.GATConv(in_length, hidden_length, dropout=0.1)
        self.attention = nn.Linear(hidden_length, 1)
        self.global_attention = tgnn.GlobalAttention(self.attention)
        self.output_net = nn.Sequential(
            nn.Linear(hidden_length, hidden_length),
            nn.Linear(hidden_length, num_classes),
        )
        self.f1 = F1(num_classes, threshold=0.5)

    def _execute(self, batch, batch_idx):
        pred = self(batch)
        labels = batch.label.float()
        loss = F.binary_cross_entropy_with_logits(pred, labels)
        f1 = f1_score(
            labels.cpu() > 0.5, torch.sigmoid(pred).cpu() > 0.5, average="micro"
        )
        return loss, f1

    def training_step(self, *args, **kwargs):
        loss, f1 = self._execute(*args, **kwargs)
        self.log(
            "train_loss",
            loss.detach().item(),
            on_step=False,
            on_epoch=True,
            prog_bar=True,
            logger=True,
        )
        self.log(
            "train_f1",
            f1.item(),
            on_step=False,
            on_epoch=True,
            prog_bar=True,
            logger=True,
        )
        return loss

    def validation_step(self, *args, **kwargs):
        with torch.no_grad():
            loss, f1 = self._execute(*args, **kwargs)
            self.log(
                "val_loss",
                loss.detach().item(),
                on_step=False,
                on_epoch=True,
                prog_bar=True,
                logger=True,
            )
            self.log(
                "val_f1",
                f1.item(),
                on_step=False,
                on_epoch=True,
                prog_bar=True,
                logger=True,
            )
            return loss

    def forward(self, x):
        a = self.embedding(x.x)
        l = []
        for _ in range(self.loops):
            a = self.left_graph_net(a, x.edge_index.long())
            l.append(a)
        at = self.global_attention(self.node_net(torch.cat(l, dim=1)), x.x_batch)
        return self.output_net(at)

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters())
        return optimizer


def train(train_loader, validation_loader):
    if torch.cuda.is_available():
        trainer_kwargs = dict(gpus=-1, accelerator="ddp")
    else:
        trainer_kwargs = dict(gpus=0)
    net = JCINet(100, 100, 500)
    tb_logger = pl_loggers.CSVLogger("../../logs/")
    checkpoint_callback = ModelCheckpoint(
        dirpath=os.path.join(tb_logger.log_dir, "checkpoints"),
        filename="{epoch}-{step}-{val_loss:.7f}",
        save_top_k=5,
        save_last=True,
        verbose=True,
        monitor="val_loss",
        mode="min",
    )
    trainer = pl.Trainer(
        logger=tb_logger,
        callbacks=[checkpoint_callback],
        replace_sampler_ddp=False,
        **trainer_kwargs
    )
    trainer.fit(net, train_loader, val_dataloaders=validation_loader)


if __name__ == "__main__":
    batch_size = int(sys.argv[1])
    # vl = ClassificationData("data/full_chebi", split="validation")
    # tr = ClassificationData("data/full_chebi", split="train")
    tr = JCIClassificationData("data/JCI_data", split="train")
    vl = JCIClassificationData("data/JCI_data", split="validation")

    train_loader = DataLoader(
        tr,
        shuffle=True,
        batch_size=batch_size,
        follow_batch=["x", "edge_index", "label"],
    )
    validation_loader = DataLoader(
        vl, batch_size=batch_size, follow_batch=["x", "edge_index", "label"]
    )

    train(train_loader, validation_loader)
