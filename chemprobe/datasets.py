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
import importlib.resources as pkg_resources

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
from chemprobe.bio import PROTCODE_GENES


###########################################################################################################################################
#        #       #       #       #       #       #       #       #       #       #       #       #       #       #       #       #       #
#                                                              PRIMARY FUNCTIONS
#    #       #       #       #       #       #       #       #       #       #       #       #       #       #       #       #       #
###########################################################################################################################################


def load_cpds():
    with pkg_resources.path('chemprobe.data', 'cpds.csv.gz') as path:
        cpds = pd.read_csv(path, index_col=0)
    return cpds


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
        onehot_cpds=False,
        permute_labels=False,
        num_workers=0,
    ):
        super().__init__()
        self.data_path = Path(data_path)
        self.fold = fold
        self.batch_size = batch_size
        self.onehot_cpds = onehot_cpds
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
        parser.add_argument("--onehot_cpds", action="store_true", help="One-hot encode compounds")
        return parent_parser

    # When doing distributed training, Datamodules have two optional arguments for
    # granular control over download/prepare/splitting data:
    def prepare_data(self):
        pass

    # OPTIONAL, called for every GPU/machine (assigning state is OK)
    def setup(self, stage):

        if stage == "fit":
            # read
            self.metadata = pd.read_csv(self.data_path.joinpath("metadata.csv.gz"), index_col=0)
            self.cells = pd.read_csv(self.data_path.joinpath("cells.csv.gz"), index_col=0)
            self.cpds = pd.read_csv(self.data_path.joinpath("cpds.csv.gz"), index_col=0)

            if self.permute_labels:
                np.random.shuffle(self.metadata["viability"].values)

            if self.onehot_cpds:
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
            return self.train_dataset, self.val_dataset

        if stage == "test":
            raise NotImplementedError

        if stage == "predict":
            raise NotImplementedError

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
        raise NotImplementedError


class ChemProbePredictDataModule(pl.LightningDataModule):
    def __init__(
        self,
        data_path, 
        cpds=None,
        batch_size=1024,
        num_workers=0,
        pin_memory=True,
    ):
        super().__init__()
        self.data_path = Path(data_path)
        self.cpds_list = cpds  # Store the original list of compound names
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.pin_memory = pin_memory
        self.PROTCODE_GENES = PROTCODE_GENES
        
    @staticmethod
    def add_argparse_args(parent_parser, **kwargs):
        parser = parent_parser.add_argument_group("CTRPDataModule")
        parser.add_argument("--data_path", type=Path, required=True)
        parser.add_argument(
            "--cpds", nargs="+", default=None, help="Compounds to make predictions for."
        )
        parser.add_argument("--batch_size", type=int, default=1024)
        parser.add_argument("--num_workers", type=int, default=0)
        return parent_parser

    def prepare_data(self):
        pass

    def setup(self, stage):
        if stage == "fit":
            raise NotImplementedError

        if stage == "test":
            raise NotImplementedError

        if stage == "predict":
            # Read in custom supplied file
            self.pred_cells = pd.read_csv(self.data_path.joinpath("cells.csv.gz"), index_col=0)
            self.pred_cells = self.impute_genes(self.pred_cells)
            self.pred_cells = self.filter_genes(self.pred_cells)
            self.pred_cells = pd.DataFrame(
                StandardScaler().fit_transform(self.pred_cells),
                index=self.pred_cells.index,
                columns=self.pred_cells.columns
            )

            # Read in CTRP data
            cpds = load_cpds()

            if self.cpds_list is not None:
                available_cpds = cpds.index.intersection(self.cpds_list)
                if available_cpds.empty:
                    raise ValueError(f"None of the supplied compounds {self.cpds_list} are found in the CTRP dataset.")
                else:
                    self.cpds = cpds.loc[available_cpds]
                    print(f"Predicting on available CTRP compounds: {available_cpds.tolist()}")
            else:
                print("Predicting on all CTRP compounds")
                self.cpds = cpds
            
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
            return self.pred_dataset

    def impute_genes(self, df, missing_tolerance=0.15):
        missing = set(self.PROTCODE_GENES) - set(df.columns)
        if missing:
            frac = len(missing) / len(df.columns)
            if frac > missing_tolerance:
                raise ValueError(f"Missing {frac*100:.2f}% of genes, missing tolerance is {missing_tolerance*100:.2f}%")
            else:
                print(f"Missing {frac*100:.2f}% of genes, zero imputing")
                impute_df = pd.DataFrame(
                    np.zeros((len(df), len(missing))), index=df.index, columns=list(missing)
                )
                df = pd.concat((df, impute_df), axis=1)
        return df

    def filter_genes(self, df, filter_by="var"):
        # subset to protcode genes
        df = df[self.PROTCODE_GENES]
        # filter out duplicate genes
        if df.columns.has_duplicates:
            dup = df.columns[df.columns.duplicated()]
            print(f"Duplicate genes: {dup}\n")
            if len(dup) > 100:
                Warning("More than 100 duplicate genes, this may be a problem")
            df = df.T
            if filter_by == "mean":
                print(f"Keeping gene duplicates with highest average expression")
                df[filter_by] = df.mean(axis=1)
            if filter_by == "var":
                print(f"Keeping gene duplicates with highest expression variance")
                df[filter_by] = df.var(axis=1)
            df = df.sort_values(by=filter_by, ascending=False)
            df = df[~df.index.duplicated(keep="first")]
            df = df.drop(columns=[filter_by]).T
        return df
    
    def train_dataloader(self):
        raise NotImplementedError

    def val_dataloader(self):
        raise NotImplementedError

    def test_dataloader(self):
        raise NotImplementedError

    def predict_dataloader(self):
        sampler = BatchSampler(SequentialSampler(self.pred_dataset), self.batch_size, drop_last=False)
        return DataLoader(
            self.pred_dataset,
            sampler=sampler,
            batch_sampler=None,
            batch_size=None,
            shuffle=False,
            num_workers=self.num_workers,
            pin_memory=self.pin_memory,
        )
    
    def attribute_dataloader(self, df, batch_size=128):
        self.attr_metadata = df
        self.attr_dataset = ChemProbeDataset(
            self.attr_metadata,
            self.pred_cells,
            self.cpds,
        )
        sampler = BatchSampler(SequentialSampler(self.attr_dataset), batch_size, drop_last=False)
        return DataLoader(
            self.attr_dataset,
            sampler=sampler,
            batch_sampler=None,
            batch_size=None,
            shuffle=False,
            num_workers=self.num_workers,
            pin_memory=self.pin_memory,
        )