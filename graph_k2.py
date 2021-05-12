import os
from torch import nn
import torch
from sklearn.metrics import f1_score
import torch.nn.functional as F
import pytorch_lightning as pl
from pytorch_lightning import loggers as pl_loggers
from pytorch_lightning.callbacks import ModelCheckpoint, EarlyStopping
from pytorch_lightning.metrics import F1
import torch.nn.functional as F
from torch_geometric import nn as tgnn
from torch_scatter import scatter_mean
from torch_geometric.data import DataLoader
from k_gnn import avg_pool
from data import JCIExtendedGraphData, JCIExtendedGraphTwoData

import logging
import sys

logging.getLogger('pysmiles').setLevel(logging.CRITICAL)



class JCINet(pl.LightningModule):

    def __init__(self, in_length, hidden_length, num_classes, loops=10):
        super().__init__()
        self.embedding = torch.nn.Embedding(800, in_length)

        self.conv1_1 = tgnn.GraphConv(in_length, in_length)
        self.conv1_2 = tgnn.GraphConv(in_length, in_length)
        self.conv1_3 = tgnn.GraphConv(in_length, hidden_length)

        self.conv2_1 = tgnn.GraphConv(in_length, in_length)
        self.conv2_2 = tgnn.GraphConv(in_length, in_length)
        self.conv2_3 = tgnn.GraphConv(in_length, hidden_length)

        self.output_net = nn.Sequential(nn.Linear(hidden_length*2, hidden_length), nn.ELU(),
                                        nn.Linear(hidden_length, hidden_length), nn.ELU(),
                                        nn.Linear(hidden_length, num_classes))

        self.f1 = F1(num_classes, threshold=0.5)
        self.loss = nn.BCEWithLogitsLoss()
        self.f1 = F1(500, threshold=0.5)
        self.dropout = nn.Dropout(0.1)

    def _execute(self, batch, batch_idx):
        pred = self(batch)
        labels = batch.y.float()
        loss = self.loss(pred, labels)
        f1 = f1_score(labels.cpu()>0.5, torch.sigmoid(pred).cpu()>0.5, average="micro")
        return loss, f1

    def training_step(self, *args, **kwargs):
        loss, f1 = self._execute(*args, **kwargs)
        self.log('train_loss', loss.detach().item(), on_step=False, on_epoch=True, prog_bar=True, logger=True)
        self.log('train_f1', f1.item(), on_step=False, on_epoch=True, prog_bar=True, logger=True)
        return loss

    def validation_step(self, *args, **kwargs):
        with torch.no_grad():
            loss, f1 = self._execute(*args, **kwargs)
            self.log('val_loss', loss.detach().item(), on_step=False, on_epoch=True, prog_bar=True, logger=True)
            self.log('val_f1', f1.item(), on_step=False, on_epoch=True, prog_bar=True, logger=True)
            return loss

    def forward(self, x):
        a = self.embedding(x.x)
        a = self.dropout(a)

        a = F.elu(self.conv1_1(a, x.edge_index.long()))
        a = F.elu(self.conv1_2(a, x.edge_index.long()))
        a = F.elu(self.conv1_3(a, x.edge_index.long()))
        a_1 = scatter_mean(a, x.batch, dim=0)

        a = avg_pool(a, x.assignment_index_2)
        a = F.elu(self.conv2_1(a, x.edge_index.long()))
        a = F.elu(self.conv2_2(a, x.edge_index.long()))
        a = F.elu(self.conv2_3(a, x.edge_index.long()))
        a_2 = scatter_mean(a, x.batch_2, dim=0)

        a = torch.cat([a_1, a_2], dim=1)

        a = self.dropout(a)
        return self.output_net(a)

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters())
        return optimizer

def run_graph(batch_size):
    data = JCIExtendedGraphTwoData(batch_size=batch_size)
    data.prepare_data()
    data.setup()
    train_data = data.train_dataloader()
    val_data = data.val_dataloader()
    if torch.cuda.is_available():
        trainer_kwargs = dict(gpus=-1, accelerator="ddp")
    else:
        trainer_kwargs = dict(gpus=0)
    net = JCINet(100, 100, 500)
    tb_logger = pl_loggers.CSVLogger('logs/')
    checkpoint_callback = ModelCheckpoint(
        dirpath=os.path.join(tb_logger.log_dir, "checkpoints"),
        filename="{epoch}-{step}-{val_loss:.7f}",
        save_top_k=5,
        save_last=True,
        verbose=True,
        monitor='val_loss',
        mode='min'
    )
    es = EarlyStopping(monitor='val_loss', patience=10, min_delta=0.00,
       verbose=False,
    )

    trainer = pl.Trainer(logger=tb_logger, min_epochs=100, callbacks=[checkpoint_callback, es], replace_sampler_ddp=False, **trainer_kwargs)
    trainer.fit(net, train_data, val_dataloaders=val_data)


if __name__ == "__main__":
    run_graph(int(sys.argv[1]))
