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


# I/O
import sys
from pathlib import Path
import argparse
from datetime import datetime

# Data handling

# Modeling
from pytorch_lightning import Trainer, seed_everything
from pytorch_lightning.callbacks import EarlyStopping, ModelCheckpoint
from pytorch_lightning.loggers.tensorboard import TensorBoardLogger
from pytorch_lightning.plugins import DDPPlugin
import optuna

# Custom
from chemprobe.datasets import ChemProbeDataModule
from chemprobe.models import (
    ChemProbe,
    ConcatNetwork,
    weight_reset,
)


###########################################################################################################################################
#        #       #       #       #       #       #       #       #       #       #       #       #       #       #       #       #       #
#                                                              PRIMARY FUNCTIONS
#    #       #       #       #       #       #       #       #       #       #       #       #       #       #       #       #       #
###########################################################################################################################################


def parse_optuna_study(study_path, exp, folds=[0, 1, 2, 3, 4]):
    """Parse Optuna study for best model checkpoints."""
    trial_paths = []
    for fold in folds:
        name = f"exp={exp}-fold={fold}"
        study_stem = Path(f"{study_path}/{name}/")
        study_db = Path(f"{study_path}/{name}/{name}.db")
        if Path(study_db).exists():
            study = optuna.load_study(name, f"sqlite:////{str(study_db)}").trials_dataframe()
            study = study[study["state"].isin(["COMPLETE", "PRUNED"])][
                ["user_attrs_fold", "datetime_start", "number", "value"]
            ]
            number = study.nlargest(columns="value", n=1)["number"].item()
            study_stem = study_stem.joinpath(f"model_logs/exp={exp}-fold={fold}-trial={number}/")
        path = list(
            study_stem
            .joinpath("checkpoints")
            .glob("epoch*.ckpt")
        )[0]
        trial_paths.append(path)
    if len(trial_paths) == 1:
        trial_paths = trial_paths[0]
    else:
        raise ValueError("More than one trial path found.")
    return trial_paths


def process(args):
    # reproducibility
    seed_everything(2299)
    dict_args = vars(args)

    # name conventions
    exp = args.exp
    if args.permute_labels:
        study_name = args.study_path.stem
        exp = f"{study_name}-PERMUTED"

    # data
    dm = ChemProbeDataModule.from_argparse_args(args)
    dm.prepare_data(stage="fit")
    
    # model
    if args.exp in ["concat", "id"]:
        model = ConcatNetwork(**dict_args)
    elif args.permute_labels:
        model = parse_optuna_study(args.study_path, args.exp, folds=[args.fold])
        print(f"Loading {model}")
        model = ChemProbe.load_from_checkpoint(model)
        print("Resetting weights")
        model.apply(weight_reset)
    else:
        model = ChemProbe(**dict_args)
    
    # callbacks
    logger = TensorBoardLogger(
        save_dir=args.data_path.parent.joinpath("lightning_logs"), name=f"{args.name}-{exp}", version=f"fold={args.fold}",
    )
    early_stop = EarlyStopping(
        monitor="val_MeanSquaredError", min_delta=1e-4, patience=12, verbose=False, mode="min"
    )
    checkpoint_callback = ModelCheckpoint(
        monitor="val_MeanSquaredError", mode="min", save_last=True
    )
    
    # trainer
    start = datetime.now()
    trainer = Trainer.from_argparse_args(
        args,
        strategy="ddp_find_unused_parameters_false",
        default_root_dir=logger.log_dir,
        logger=logger,
        callbacks=[early_stop, checkpoint_callback],
        profiler="simple",
        replace_sampler_ddp=False,
    )
    trainer.fit(model, dm)
    print("Completed fold {} in {}".format(args.fold, str(datetime.now() - start)))
    print(f"Model saved at {checkpoint_callback.best_model_path}")
    return


def main():
    """Parse Arguments"""
    desc = "Script for training multiple methods of conditional featurization for prediction of cell viability."
    parser = argparse.ArgumentParser(
        description=desc, formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    # Experiment args
    parser.add_argument(
        "--name",
        type=str,
        required=True,
    )
    parser.add_argument(
        "--exp",
        type=str,
        choices=["concat", "id", "shift", "scale", "film"],
        help="Model type.",
    )
    temp_args, _ = parser.parse_known_args()

    # Model args
    if temp_args.exp in ["concat", "id"]:
        parser = ConcatNetwork.add_model_specific_args(parser)
    elif temp_args.exp in ["shift", "scale", "film"]:
        parser = ChemProbe.add_model_specific_args(parser)

    # Data args
    parser = ChemProbeDataModule.add_argparse_args(parser)
    temp_args, _ = parser.parse_known_args()
    if temp_args.permute_labels:
        parser.add_argument(
            "--study_path", type=Path, help="Directory of model or optimized models."
        )

    # Trainer args
    parser = Trainer.add_argparse_args(parser)

    args = parser.parse_args()

    return process(args)


if __name__ == "__main__":  # pragma: no cover
    sys.exit(main())
