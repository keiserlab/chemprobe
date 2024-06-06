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
import scanpy as sc
from kneed import KneeLocator

# Pytorch
import torch
from pytorch_lightning import Trainer

# dose-response
from thunor.io import read_vanderbilt_hts
from thunor.viability import viability
from thunor.curve_fit import fit_params

# Custom
from chemprobe.bio import PROTCODE_GENES
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
    ensemble = torch.hub.load("keiserlab/chemprobe", model="ChemProbeEnsemble", source="github", attribute=False)
    ensemble.eval()

    # data
    dm = ChemProbePredictDataModule.from_argparse_args(args)

    # trainer
    trainer = Trainer.from_argparse_args(
        args,
        profiler=None,
        logger=False,
        precision=32,
        replace_sampler_ddp=False,
    )

    # predict
    results = trainer.predict(ensemble, dm)
    predictions = []
    for batch in results:
        batch_results = [fold['target_hat'] for fold in batch.values()]
        predictions.append(torch.cat(batch_results, dim=1))
    predictions = torch.cat(predictions, dim=0).numpy()
    del results
    predictions = pd.DataFrame(
        predictions, columns=np.arange(predictions.shape[1])
    )
    predictions = pd.concat(
        (dm.pred_metadata.reset_index(drop=True), predictions), axis=1
    )
    
    # format predictinos into vanderbilt hts format
    vhts = format_vanderbilt_hts(predictions)
    vhts.to_csv(args.data_path.joinpath("predictions_vhts.csv.gz"), index=False)

    # read into thunor
    print("Fitting dose-response curves...")
    plate_height = vhts["fold"].nunique() * 2
    plate_width = vhts["dose"].nunique()
    print(f"\nPlate height: {plate_height}")
    print(f"Plate width: {plate_width}\n")
    vhts = read_vanderbilt_hts(
        args.data_path.joinpath("predictions_vhts.csv.gz"),
        plate_height=plate_height,
        plate_width=plate_width,
        sep=",",
    )

    # fit dose-response curves
    res, ctrl = viability(vhts)
    params = fit_params(ctrl, res)
    
    # filter params
    params = params.dropna(subset='ic50')
    x = np.arange(100+1)
    y = [np.percentile(params['auc'], i) for i in x]
    kl = KneeLocator(x, y, curve="convex")
    params = params[params['auc'] < np.percentile(params['auc'], kl.knee-2)]

    # writeout
    params.to_pickle(args.data_path.joinpath("params.pkl"))

    # attribute at IC50
    if args.attribute:        
        print("Attributing at predicted IC50s...")
        query = params.reset_index()[['cell_line', 'drug', 'ic50']]
        query = query.rename(columns={'cell_line': 'ccl_name', 'drug': 'cpd_name', 'ic50': 'dose'})
        query['dose'] = query['dose'] * 1e6
        query['viability'] = np.nan
        
        # trainer with inference_mode=False
        trainer = Trainer.from_argparse_args(
            args,
            profiler=None,
            logger=False,
            precision=32,
            replace_sampler_ddp=False,
            inference_mode=False
        )
        dm.setup("attribute")
        dl = dm.attribute_dataloader(query, batch_size=128)
        ensemble.activate_integrated_gradients()
        results = trainer.predict(ensemble, dataloaders=dl)

        # write each fold attributions
        for fold in range(5):
            attributions = []
            for batch in results:
                attributions.append(batch[fold]['attributions'])
            attributions = torch.abs(torch.cat(attributions, dim=0)).numpy()
            attributions = pd.DataFrame(attributions, columns=PROTCODE_GENES)
            attributions = sc.AnnData(attributions, obs=query.reset_index(drop=True), var=pd.DataFrame(index=PROTCODE_GENES))
            attributions.write(args.data_path.joinpath(f"attributions_fold={fold}.h5ad"))

        # averaged attributions across folds for each gene
        attributions = []
        for batch in results:
            a = torch.cat([batch[fold]['attributions'].unsqueeze(0) for fold in range(5)], dim=0)
            a = torch.mean(torch.abs(a), dim=0)
            attributions.append(a)
        attributions = torch.cat(attributions, dim=0).numpy()
        attributions = pd.DataFrame(attributions, columns=PROTCODE_GENES)
        attributions = sc.AnnData(attributions, obs=query.reset_index(drop=True), var=pd.DataFrame(index=PROTCODE_GENES))
        attributions.write(args.data_path.joinpath(f"attributions_avg.h5ad"))

    print(f"Data written to {args.data_path}")


def main():
    """Parse Arguments"""
    desc = "Script for predicting cellular viability."
    parser = argparse.ArgumentParser(
        description=desc, formatter_class=argparse.ArgumentDefaultsHelpFormatter
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
