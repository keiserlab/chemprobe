"""
Author: Will Connell
Date Initialized: 2020-09-27
Email: connell@keiserlab.org

Script to preprocess data.
"""


###########################################################################################################################################
#        #       #       #       #       #       #       #       #       #       #       #       #       #       #       #       #       #
#                                                                IMPORT MODULES
#    #       #       #       #       #       #       #       #       #       #       #       #       #       #       #       #       #
###########################################################################################################################################


# I/O
import sys
import argparse
from pathlib import Path
import joblib

# Data handling
import numpy as np
import pandas as pd
from sklearn.model_selection import GroupKFold
from chemprobe.chem import smiles_to_bits

# Transforms
from sklearn.preprocessing import StandardScaler


###########################################################################################################################################
#        #       #       #       #       #       #       #       #       #       #       #       #       #       #       #       #       #
#                                                              PRIMARY FUNCTIONS
#    #       #       #       #       #       #       #       #       #       #       #       #       #       #       #       #       #
###########################################################################################################################################


def process(args, n_splits=5):
    # seed
    np.random.seed(2299)
    
    # cpds, CTRP
    ctrp_cpds = pd.read_csv(
        args.data_path.joinpath("cpds/v20.meta.per_compound.txt"), sep="\t", index_col=0
    )
    ctrp_cl = pd.read_csv(
        args.data_path.joinpath("cpds/v20.meta.per_cell_line.txt"), sep="\t", index_col=0
    )
    ctrp_exp = pd.read_csv(
        args.data_path.joinpath("cpds/v20.meta.per_experiment.txt"), sep="\t", index_col=0
    )
    ctrp_data = pd.read_csv(
        args.data_path.joinpath("cpds/v20.data.per_cpd_post_qc.txt"), sep="\t", index_col=0
    )

    # CCLE sample info
    ccle_samples = pd.read_csv(args.data_path.joinpath("cells/sample_info.csv"))
    # add exp info
    metadata = ctrp_data.join(ctrp_exp["master_ccl_id"]).drop_duplicates()
    # add cell line info
    metadata = metadata.merge(ctrp_cl["ccl_name"], left_on="master_ccl_id", right_index=True)
    # add cpd info
    metadata = metadata.merge(
        ctrp_cpds[["cpd_name", "broad_cpd_id", "cpd_smiles"]],
        left_on="master_cpd_id",
        right_index=True,
    )
    # intersect with CCLE sample metadata
    metadata = metadata.merge(
        ccle_samples[["stripped_cell_line_name", "DepMap_ID"]],
        left_on="ccl_name",
        right_on="stripped_cell_line_name",
        how="inner"
    )
    # format target
    metadata["viability"] = np.clip(metadata["cpd_pred_pv"], a_min=0.0, a_max=None)

    # CCLE
    ccle = (
        pd.read_csv(args.data_path.joinpath("cells/CCLE_expression.csv"), index_col=0)
        .astype(np.float32)
    )
    # clean up gene names
    ccle.columns = [g.split(" ")[0] for g in ccle.columns]
    # filter to cell lines in metadata
    ccle = ccle[ccle.index.isin(metadata['DepMap_ID'].unique())]
    # map to cell line names
    ccle = ccle.rename(index=metadata.set_index('DepMap_ID')['ccl_name'].to_dict())
    ccle = ccle.rename_axis("ccl_name")

    # filter to cell lines with expression data
    cell_lines = set(metadata["ccl_name"].values).intersection(ccle.index.values)
    ccle = ccle[ccle.index.isin(cell_lines)]
    # filter metadata
    metadata = metadata[metadata['ccl_name'].isin(cell_lines)]
    metadata = metadata.reset_index(drop=True)

    # generate fingerprints
    cpd_data = metadata[["cpd_name", "cpd_smiles"]].drop_duplicates()
    fp = smiles_to_bits(cpd_data["cpd_smiles"], nBits=args.nBits)
    fp.index = cpd_data["cpd_name"]

    # generate folds
    metadata["fold"] = -1
    gkf = GroupKFold(n_splits=5)
    for fold, (train_idx, val_idx) in enumerate(
        gkf.split(X=metadata, y=None, groups=metadata["ccl_name"])
    ):
        metadata.loc[val_idx, "fold"] = fold

    # write out data
    args.data_path.joinpath("preprocessed").mkdir(parents=True, exist_ok=True)
    metadata.to_csv(args.data_path.joinpath(f"preprocessed/metadata.csv.gz"))
    fp.to_csv(args.data_path.joinpath(f"preprocessed/cpds.csv.gz"))
    ccle.to_csv(args.data_path.joinpath(f"preprocessed/cells.csv.gz"))

    return f"Completed preprocessing, saved to {args.data_path.joinpath('preprocessed')}"


###########################################################################################################################################
#        #       #       #       #       #       #       #       #       #       #       #       #       #       #       #       #       #
#                                                                    CLI
#    #       #       #       #       #       #       #       #       #       #       #       #       #       #       #       #       #
###########################################################################################################################################


def main():
    """Parse Arguments"""
    desc = "Script for preprocessing data for reproducibility."
    parser = argparse.ArgumentParser(
        description=desc, formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    # positional
    parser.add_argument(
        "--data_path", type=Path, required=True, help="Directory to write processed data."
    )
    parser.add_argument(
        "--nBits", type=int, default=512, help="Number of bits for fingerprints."
    )
    args = parser.parse_args()

    return process(args)


if __name__ == "__main__":  # pragma: no cover
    sys.exit(main())