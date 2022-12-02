"""
Author: Will Connell
Date Initialized: 2021-09-09
Email: connell@keiserlab.org

Models for predicting cell viability from compound features.
"""


###########################################################################################################################################
#        #       #       #       #       #       #       #       #       #       #       #       #       #       #       #       #       #
#                                                                IMPORT MODULES
#    #       #       #       #       #       #       #       #       #       #       #       #       #       #       #       #       #
###########################################################################################################################################

# Data handling
import numpy as np

# Models
import torch
from torch import nn
import torch.nn.functional as F
from torch.optim.lr_scheduler import ReduceLROnPlateau
import pytorch_lightning as pl
from torchmetrics import MetricCollection, MeanSquaredError, R2Score

# Attribution
from captum.attr import IntegratedGradients


###########################################################################################################################################
#        #       #       #       #       #       #       #       #       #       #       #       #       #       #       #       #       #
#                                                              PRIMARY FUNCTIONS
#    #       #       #       #       #       #       #       #       #       #       #       #       #       #       #       #       #
###########################################################################################################################################


def weight_reset(m: nn.Module):
    # - check if the current module has reset_parameters & if it's callabed called it on m
    reset_parameters = getattr(m, "reset_parameters", None)
    if callable(reset_parameters):
        m.reset_parameters()


class LinearBlock(nn.Module):
    def __init__(self, *args, **kwargs):
        super().__init__()
        """
        code borrowed from fastai `categorical` model
        """
        self.block = self.generate_layers(*args, **kwargs)
        self.out_sz = kwargs["out_sz"]

    def generate_layers(self, in_sz, layers, out_sz, ps, use_bn, bn_final):
        if ps is None:
            ps = [0] * len(layers)
        else:
            ps = ps * len(layers)
        sizes = self.get_sizes(in_sz, layers, out_sz)
        actns = [nn.ReLU(inplace=True) for _ in range(len(sizes) - 2)] + [None]
        layers = []
        for i, (n_in, n_out, dp, act) in enumerate(
            zip(sizes[:-1], sizes[1:], [0.0] + ps, actns)
        ):
            layers += self.bn_drop_lin(
                n_in, n_out, bn=use_bn and i != 0, p=dp, actn=act
            )
        if bn_final:
            layers.append(nn.BatchNorm1d(sizes[-1]))
        block = nn.Sequential(*layers)
        return block

    def get_sizes(self, in_sz, layers, out_sz):
        return [in_sz] + layers + [out_sz]

    def bn_drop_lin(
        self,
        n_in: int,
        n_out: int,
        bn: bool = True,
        p: float = 0.0,
        actn: nn.Module = None,
    ):
        "`n_in`->bn->dropout->linear(`n_in`,`n_out`)->`actn`"
        layers = [nn.BatchNorm1d(n_in)] if bn else []
        if p != 0:
            layers.append(nn.Dropout(p))
        layers.append(nn.Linear(n_in, n_out))
        if actn is not None:
            layers.append(actn)
        return layers

    def forward(self, x):
        x = self.block(x)
        return x


class FiLMGenerator(nn.Module):
    def __init__(self, exp, in_sz, out_sz, ps=0.1):
        super().__init__()
        self.exp = exp
        self.gamma = nn.Sequential(nn.Linear(in_sz, in_sz//2), nn.ReLU(), nn.Dropout(ps), nn.Linear(in_sz//2, out_sz))
        self.beta = nn.Sequential(nn.Linear(in_sz, in_sz//2), nn.ReLU(), nn.Dropout(ps), nn.Linear(in_sz//2, out_sz))

    def forward(self, x):
        if self.exp == "scale":
            gamma = self.gamma(x)
            beta = torch.zeros_like(gamma)
        elif self.exp == "shift":
            beta = self.beta(x)
            gamma = torch.ones_like(beta)
        elif self.exp == "film":
            gamma = self.gamma(x)
            beta = self.beta(x)
        else:
            raise ValueError(
                f"Experiment type `{self.exp}` not supported for ConditionalNetwork."
            )
        return gamma, beta


class FiLMLayer(nn.Module):
    def __init__(self, exp, emb_sz, in_sz, out_sz, ps_film, ps_linear):
        super().__init__()
        self.film_generator = FiLMGenerator(exp, emb_sz, in_sz, ps=ps_film)
        self.linear_block = LinearBlock(in_sz=in_sz, layers=[in_sz//2], out_sz=out_sz, ps=[ps_linear], use_bn=True, bn_final=False)

    def forward(self, data):
        cells, cpds = data
        gamma, beta = self.film_generator(cpds)
        cells = gamma * cells + beta
        cells = self.linear_block(cells)
        return cells, cpds


class ChemProbe(pl.LightningModule):
    """
    ChemProbe for predicting cell viability from cell line and compound features using a learned conditional transformation.
    """

    def __init__(
        self,
        exp="film",
        cells_sz=19144,
        cpds_sz=513,
        emb_sz=32,
        n_blocks=2,
        ps_emb=0.2,
        ps_film=0.2,
        ps_linear=0.2,
        lr=1e-3,
        weight_decay=1e-5,
        **kwargs,
    ):
        super().__init__()
        # Save state
        self.exp = exp
        self.emb_sz = emb_sz

        # Metrics
        metrics = MetricCollection([MeanSquaredError(), R2Score()])
        self.train_metrics = metrics.clone(prefix="train_")
        self.valid_metrics = metrics.clone(prefix="val_")
        self.test_metrics = metrics.clone(prefix="test_")

        # Model layers
        self.cells_emb = LinearBlock(
            in_sz=cells_sz,
            layers=[256, 128],
            out_sz=emb_sz,
            ps=[ps_emb],
            use_bn=True,
            bn_final=False,
        )
        self.cpds_emb = LinearBlock(
            in_sz=cpds_sz,
            layers=[256, 128],
            out_sz=emb_sz,
            ps=[ps_emb],
            use_bn=True,
            bn_final=False,
        )
        self.film_blocks = self.build_film_layers(n_blocks=n_blocks, ps_film=ps_film, ps_linear=ps_linear)
        self.save_hyperparameters()

    @staticmethod
    def add_model_specific_args(parent_parser):
        parser = parent_parser.add_argument_group("ConditionalNetwork")
        parser.add_argument("--nblocks", type=int, default=2)
        parser.add_argument("--ps_emb", type=float, default=0.2)
        parser.add_argument("--ps_film", type=float, default=0.2)
        parser.add_argument("--ps_linear", type=float, default=0.2)
        parser.add_argument("--lr", type=float, default=1e-3)
        parser.add_argument("--weight_decay", type=float, default=1e-5)
        return parent_parser

    def build_film_layers(self, n_blocks=2, ps_film=0.2, ps_linear=0.2):
        blocks_sz = [self.emb_sz//i for i in np.arange(2, 2*(n_blocks-1)+2, 2)]
        film_layers_sz = [self.emb_sz] + blocks_sz + [1]
        emb_sz = self.emb_sz
        
        film_layers = []
        for i in range(len(film_layers_sz)-1):
            film_layer = FiLMLayer(
                exp=self.exp,
                emb_sz=emb_sz,
                in_sz=film_layers_sz[i],
                out_sz=film_layers_sz[i+1],
                ps_film=ps_film,
                ps_linear=ps_linear,
            )
            film_layers.append(film_layer)
        
        return nn.Sequential(*film_layers)

    def forward(self, cells, cpds):
        cells_emb = self.cells_emb(cells)
        cpds_emb = self.cpds_emb(cpds)
        target_hat, cpds = self.film_blocks((cells_emb, cpds_emb))
        target_hat = torch.clamp(target_hat, min=0)
        return cells_emb, cpds_emb, target_hat

    def training_step(self, batch, batch_idx):
        cells, cpds, target = batch
        cells_emb, cpds_emb, target_hat = self.forward(cells, cpds)
        metrics = self.train_metrics(target_hat, target)
        self.log_dict(
            metrics,
            on_step=True,
            on_epoch=True,
            prog_bar=True,
            logger=True,
            batch_size=len(batch),
            sync_dist=True,
        )
        return metrics["train_MeanSquaredError"]

    def validation_step(self, batch, batch_idx):
        cells, cpds, target = batch
        cells_emb, cpds_emb, target_hat = self.forward(cells, cpds)
        metrics = self.valid_metrics(target_hat, target)
        self.log_dict(
            metrics,
            on_step=False,
            on_epoch=True,
            prog_bar=True,
            logger=True,
            batch_size=len(batch),
            sync_dist=True,
        )
        return metrics["val_MeanSquaredError"]

    def test_step(self, batch, batch_idx):
        raise NotImplementedError

    def predict_step(self, batch, batch_idx, dataloader_idx=0):
        attributions, delta = None, None
        cells, cpds, _target = batch
        cells_emb, cpds_emb, target_hat = self.forward(cells, cpds)
        
        # apply attribution
        if self.IG:
            baselines = (
                tuple(b.to(self.device) for b in self.baselines)
                if isinstance(self.baselines, tuple)
                else self.baselines
            )
            # IG wrapped in predict_step for multi-GPU scaling
            ig = IntegratedGradients(self.forward_attribute)
            attributions, delta = ig.attribute(
                (cells, cpds[0], cpds[1]),
                baselines=baselines,
                method="gausslegendre",
                return_convergence_delta=True,
            )
        return target_hat, attributions, delta

    def forward_attribute(self, cells, cpds):
        cells_emb, cpds_emb, target_hat = self.forward(cells, cpds)
        return target_hat

    def activate_integrated_gradients(self, baselines=None):
        self.IG = True
        self.baselines = baselines

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(
            self.parameters(),
            lr=self.hparams.lr,
            weight_decay=self.hparams.weight_decay,
        )
        scheduler = ReduceLROnPlateau(optimizer, mode="min", patience=5)
        return {
            "optimizer": optimizer,
            "lr_scheduler": scheduler,
            "monitor": "val_MeanSquaredError",
        }


class ConcatNetwork(pl.LightningModule):
    """
    Concat MLP
    """
    def __init__(
        self,
        exp="concat",
        in_sz=19144 + 513,
        layers=[256, 128, 32, 16, 8],
        ps=0.2,
        lr=1e-3,
        weight_decay=1e-5,
        **kwargs,
    ):
        super().__init__()
        # Save state
        self.exp = exp
        self.in_sz = in_sz
        self.layers = layers
        
        # Metrics
        metrics = MetricCollection([MeanSquaredError(), R2Score()])
        self.train_metrics = metrics.clone(prefix="train_")
        self.valid_metrics = metrics.clone(prefix="val_")
        self.test_metrics = metrics.clone(prefix="test_")

        # Model layers
        self.mlp = LinearBlock(
            in_sz=in_sz,
            layers=self.layers,
            out_sz=1,
            ps=[ps],
            use_bn=True,
            bn_final=False,
        )
        self.save_hyperparameters()

    @staticmethod
    def add_model_specific_args(parent_parser):
        parser = parent_parser.add_argument_group("ConcatNetwork")
        parser.add_argument(
            "--layers", type=list, default=[512, 128, 32, 16, 8]
        )
        parser.add_argument("--ps", type=float, default=0.2)
        parser.add_argument("--lr", type=float, default=1e-3)
        parser.add_argument("--weight_decay", type=float, default=1e-5)
        return parent_parser

    def forward(self, data):
        target_hat = self.mlp(data)
        target_hat = torch.clamp(target_hat, min=0)
        return target_hat

    def training_step(self, batch, batch_idx):
        cells, cpds, target = batch
        data = torch.cat((cells, cpds), dim=1)
        target_hat = self.forward(data)
        metrics = self.train_metrics(target_hat, target)
        self.log_dict(
            metrics,
            on_step=True,
            on_epoch=True,
            prog_bar=True,
            logger=True,
            batch_size=len(batch),
            sync_dist=True,
        )
        return metrics["train_MeanSquaredError"]

    def validation_step(self, batch, batch_idx):
        cells, cpds, target = batch
        data = torch.cat((cells, cpds), dim=1)
        target_hat = self.forward(data)
        metrics = self.valid_metrics(target_hat, target)
        self.log_dict(
            metrics,
            on_step=False,
            on_epoch=True,
            prog_bar=True,
            logger=True,
            batch_size=len(batch),
            sync_dist=True,
        )
        return metrics["val_MeanSquaredError"]

    def test_step(self, batch, batch_idx):
        raise NotImplementedError

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(
            self.parameters(),
            lr=self.hparams.lr,
            weight_decay=self.hparams.weight_decay,
        )
        scheduler = ReduceLROnPlateau(optimizer, mode="min", patience=5)
        return {
            "optimizer": optimizer,
            "lr_scheduler": scheduler,
            "monitor": "val_MeanSquaredError",
        }
