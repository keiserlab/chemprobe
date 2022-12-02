"""
Author: Will Connell
Date Initialized: 2021-09-09
Email: connell@keiserlab.org

Script to predictionsuate new samples.
"""


###########################################################################################################################################
#        #       #       #       #       #       #       #       #       #       #       #       #       #       #       #       #       #
#                                                                IMPORT MODULES
#    #       #       #       #       #       #       #       #       #       #       #       #       #       #       #       #       #
###########################################################################################################################################

# I/O
from pathlib import Path

# Data handling
import numpy as np
import pandas as pd

# Transforms
from sklearn.preprocessing import StandardScaler, OneHotEncoder

# Models
import torch
from torch.utils.data import Dataset, DataLoader, RandomSampler, SequentialSampler, BatchSampler
import pytorch_lightning as pl

# Custom
from chemprobe.chem import generate_doses


###########################################################################################################################################
#        #       #       #       #       #       #       #       #       #       #       #       #       #       #       #       #       #
#                                                              PRIMARY FUNCTIONS
#    #       #       #       #       #       #       #       #       #       #       #       #       #       #       #       #       #
###########################################################################################################################################


class ChemProbeDataset(Dataset):
    """
    Dataset for cell gene expression data and compound features.
    """

    def __init__(
        self,
        metadata,
        cells,
        cpds,
        target="viability",
        batch_size=32
    ):
        self.metadata = metadata
        self.cells = cells
        self.cpds = cpds
        self.target = target
        self.batch_size = batch_size

    def __len__(self):
        return len(self.metadata)

    def __getitem__(self, idx):
        cells = torch.FloatTensor(self.cells.loc[self.metadata.iloc[idx]['ccl_name']].to_numpy())
        cpds = torch.FloatTensor(self.cpds.loc[self.metadata.iloc[idx]['cpd_name']].to_numpy())
        dose = torch.FloatTensor(self.metadata.iloc[idx]['dose'].to_numpy()).reshape(-1,1)
        cpds = torch.concat((cpds, dose), dim=1)
        target = torch.FloatTensor(self.metadata.iloc[idx][self.target].to_numpy())

        return cells, cpds, target


class ChemProbeDataModule(pl.LightningDataModule):
    def __init__(
        self,
        data_path,
        fold,
        batch_size=1024,
        permute_fingerprints=False,
        permute_labels=False,
        num_workers=0,
    ):
        super().__init__()
        self.data_path = Path(data_path)
        self.fold = fold
        self.batch_size = batch_size
        self.permute_fingerprints = permute_fingerprints
        self.permute_labels = permute_labels
        self.num_workers = num_workers
        
    @staticmethod
    def add_argparse_args(parent_parser, **kwargs):
        parser = parent_parser.add_argument_group("CTRPDataModule")
        parser.add_argument("--data_path", type=Path, required=True)
        parser.add_argument("--fold", type=int, required=True)
        parser.add_argument("--batch_size", type=int, default=1024)
        parser.add_argument("--num_workers", type=int, default=0)
        parser.add_argument("--permute_labels", action="store_true")
        parser.add_argument("--permute_fingerprints", action="store_true")
        parser.add_argument(
            "--pred_cells",
            type=Path,
            default=None,
            help="Path to csv of samples for predictions.",
        )
        parser.add_argument(
            "--pred_cpds", nargs="+", default=None, help="Compounds to make predictions for."
        )
        return parent_parser

    # When doing distributed training, Datamodules have two optional arguments for
    # granular control over download/prepare/splitting data:
    def prepare_data(self, stage, pred_cells=None, pred_cpds=None):

        if stage == "fit":
            # read
            self.metadata = pd.read_csv(self.data_path.joinpath("metadata.csv.gz"), index_col=0)
            self.cells = pd.read_csv(self.data_path.joinpath("cells.csv.gz"), index_col=0)
            self.cpds = pd.read_csv(self.data_path.joinpath("cpds.csv.gz"), index_col=0)

            if self.permute_labels:
                np.random.shuffle(self.metadata["viability"].values)

            if self.permute_fingerprints:
                enc = OneHotEncoder(sparse=False, handle_unknown="ignore")
                self.cpds = pd.DataFrame(
                    enc.fit_transform(self.cpds.index.values.reshape(-1, 1)),
                    index=self.cpds.index,
                )

            # preprocess train
            self.train_metadata = self.metadata[self.metadata["fold"] != self.fold]
            self.train_cells = self.cells.loc[self.train_metadata["ccl_name"].unique()]
            scaler = StandardScaler().fit(self.train_cells)
            self.train_cells = pd.DataFrame(
                scaler.transform(self.train_cells),
                index=self.train_cells.index,
                columns=self.train_cells.columns
            )

            # preprocess val
            self.val_metadata = self.metadata[self.metadata["fold"] == self.fold]
            self.val_cells = self.cells.loc[self.val_metadata["ccl_name"].unique()]
            self.val_cells = pd.DataFrame(
                scaler.transform(self.val_cells),
                index=self.val_cells.index,
                columns=self.val_cells.columns
            )

            self.train_dataset = ChemProbeDataset(
                self.train_metadata,
                self.train_cells,
                self.cpds,
            )

            self.val_dataset = ChemProbeDataset(
                self.val_metadata,
                self.val_cells,
                self.cpds,
            )

        if stage == "test":
            raise NotImplementedError

        if stage == "predict":
            # read in custom supplied file
            self.pred_cells = pd.read_csv(pred_cells, index_col=0)

            if pred_cpds is not None:
                self.cpds = self.cpds.loc[pred_cpds]
                print("Predicting on supplied CTRP compounds...")
            else:
                print("Predicting on all CTRP compounds...")
            
            # create samples across different concentrations
            self.pred_metadata = generate_doses(
                self.cpds.index.unique(), self.pred_cells.index.unique()
            )
            self.pred_metadata["viability"] = np.nan  # placeholder

            self.pred_dataset = ChemProbeDataset(
                self.pred_metadata,
                self.pred_cells,
                self.cpds,
            )

    # OPTIONAL, called for every GPU/machine (assigning state is OK)
    def setup(self, stage):

        if stage == "fit":
            return self.train_dataset, self.val_dataset

        if stage == "test":
            raise NotImplementedError

        if stage == "predict":
            return self.pred_dataset

    def train_dataloader(self):
        sampler = BatchSampler(RandomSampler(self.train_dataset), self.batch_size, drop_last=False)
        return DataLoader(
            self.train_dataset,
            sampler=sampler,
            batch_sampler=None,
            batch_size=None,
            num_workers=self.num_workers,
            pin_memory=True,
        )

    def val_dataloader(self):
        sampler = BatchSampler(SequentialSampler(self.val_dataset), self.batch_size, drop_last=False)
        return DataLoader(
            self.val_dataset,
            sampler=sampler,
            batch_sampler=None,
            batch_size=None,
            num_workers=self.num_workers,
            pin_memory=True,
        )

    def test_dataloader(self):
        raise NotImplementedError

    def predict_dataloader(self):
        sampler = BatchSampler(SequentialSampler(self.pred_dataset), self.batch_size, drop_last=False)
        return DataLoader(
            self.pred_dataset,
            sampler=sampler,
            batch_sampler=None,
            batch_size=None,
            num_workers=self.num_workers,
            pin_memory=True,
        )
