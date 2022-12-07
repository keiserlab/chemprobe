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
import sys
import argparse
from pathlib import Path
import warnings
warnings.filterwarnings("ignore")
from string import ascii_uppercase

# Data handling
import numpy as np
import pandas as pd

# Pytorch
import torch
from pytorch_lightning import Trainer

# dose-response
from thunor.io import read_vanderbilt_hts
from thunor.viability import viability
from thunor.curve_fit import fit_params

# Custom
from chemprobe.models import ChemProbeEnsemble
from chemprobe.datasets import ChemProbePredictDataModule


###########################################################################################################################################
#        #       #       #       #       #       #       #       #       #       #       #       #       #       #       #       #       #
#                                                              PRIMARY FUNCTIONS
#    #       #       #       #       #       #       #       #       #       #       #       #       #       #       #       #       #
###########################################################################################################################################


def format_vanderbilt_hts(predictions):

    # melt
    predictions = predictions.drop(columns=["viability"])
    predictions = predictions.melt(
        id_vars=["cpd_name", "ccl_name", "cpd_conc_umol", "dose"],
        var_name="fold",
        value_name="pred_viability",
    )

    # metadata
    nfolds = predictions["fold"].nunique()
    ndoses = predictions["dose"].nunique()

    # create controls
    vhts = []
    for upid, (grp, data) in enumerate(predictions.groupby(["ccl_name", "cpd_name"])):
        # create a control "row" per "plate"
        ctrls = pd.DataFrame.from_dict(
            {
                "ccl_name": np.repeat(grp[0], ndoses),
                "cpd_conc_umol": np.repeat(0, ndoses),
                "dose": np.repeat(0, ndoses),
                "cpd_name": np.repeat(grp[1], ndoses),
                "pred_viability": np.repeat(1, ndoses),
                "fold": np.repeat(-1, ndoses),
            },
            orient="index",
        ).T
        data = pd.concat([data, ctrls])
        data["well"] = [
            f"{row}{str(col).zfill(2)}"
            for row in ascii_uppercase[: nfolds + 1]
            for col in range(1, ndoses + 1)
        ]
        data["upid"] = upid
        vhts.append(data)
    vhts = pd.concat(vhts)

    # data
    vhts["time"] = 72
    vhts["cell.line"] = vhts["ccl_name"]
    vhts["cell.count"] = (vhts["pred_viability"] * 1e3).astype(int)
    vhts["drug1"] = vhts["cpd_name"]
    vhts["drug1.conc"] = vhts["cpd_conc_umol"] / 1e6
    vhts["drug1.units"] = "M"
    
    return vhts


def process(args):

    # model
    models = list(args.model_path.glob("*.ckpt"))
    print(f"Found models: {models}")
    ensemble = ChemProbeEnsemble(models, attribute=args.attribute)
    ensemble.eval()

    # data
    dm = ChemProbePredictDataModule.from_argparse_args(args)

    # trainer
    trainer = Trainer.from_argparse_args(
        args,
        profiler=None,
        logger=False,
        precision=16,
        replace_sampler_ddp=False,
    )

    # predict
    values = trainer.predict(ensemble, dm)
    predictions = torch.cat([batch[0] for batch in values]).numpy()
    predictions = pd.DataFrame(
        predictions, columns=np.arange(predictions.shape[1])
    )
    predictions = pd.concat(
        (dm.pred_metadata.reset_index(drop=True), predictions), axis=1
    )

    # format into vanderbilt hts structure
    vhts = format_vanderbilt_hts(predictions)
    vhts.to_csv(args.data_path.joinpath("predictions_vhts.csv"), index=False)

    # read into thunor
    plate_height = vhts["fold"].nunique() * 2
    plate_width = vhts["dose"].nunique()
    print(f"\nPlate height: {plate_height}")
    print(f"Plate width: {plate_width}\n")
    vhts = read_vanderbilt_hts(
        args.data_path.joinpath("predictions_vhts.csv"),
        plate_height=plate_height,
        plate_width=plate_width,
        sep=",",
    )

    # fit dose-response curves and write to file
    res, ctrl = viability(vhts)
    params = fit_params(ctrl, res)
    params.to_pickle(args.data_path.joinpath("params.pkl"))


def main():
    """Parse Arguments"""
    desc = "Script for predicting cellular viability."
    parser = argparse.ArgumentParser(
        description=desc, formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument(
        "--model_path", type=Path, default="../data/trained_models", help="Directory of models."
    )

    # Model args
    parser = ChemProbeEnsemble.add_model_specific_args(parser)
    # Data args
    parser = ChemProbePredictDataModule.add_argparse_args(parser)
    # Trainer args
    parser = Trainer.add_argparse_args(parser)

    args = parser.parse_args()

    return process(args)


if __name__ == "__main__":  # pragma: no cover
    sys.exit(main())
