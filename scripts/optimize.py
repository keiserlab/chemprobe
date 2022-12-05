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


# Modeling
import torch
from pytorch_lightning import Trainer, seed_everything
from pytorch_lightning.callbacks import EarlyStopping, ModelCheckpoint
from pytorch_lightning.loggers.tensorboard import TensorBoardLogger
from pytorch_lightning.plugins import DDPPlugin

# Optuna
import optuna
from optuna.integration import PyTorchLightningPruningCallback

# Custom
from chemprobe.datasets import ChemProbeDataModule
from chemprobe.models import ChemProbe, ConcatNetwork

# Misc
import sys
from datetime import datetime
from pathlib import Path
import argparse


###########################################################################################################################################
#        #       #       #       #       #       #       #       #       #       #       #       #       #       #       #       #       #
#                                                              PRIMARY FUNCTIONS
#    #       #       #       #       #       #       #       #       #       #       #       #       #       #       #       #       #
###########################################################################################################################################


def activate_study(study_path, pruner):
    study_path.mkdir(parents=True, exist_ok=True)
    study_name = study_path.stem
    study_db = study_path.joinpath(f"{study_name}.db")
    storage_name = f"sqlite:///{study_db}"

    print(f"Initializing study: {study_db}")
    study = optuna.create_study(
        direction="maximize",
        pruner=pruner,
        study_name=study_name,
        storage=storage_name,
        load_if_exists=True,
    )
    return study


class Objective:
    def __init__(self, args) -> None:
        self.args = args
        self.logs = args.study_path
        self.exp = args.exp

    def suggestions(self, trial):
        kwargs = {}

        if self.exp in ["concat", "id"]:
            # n_blocks or film layers
            kwargs["n_blocks"] = trial.suggest_int("n_blocks", 1, 3)
            # embeddings
            kwargs["layers"] = [
                trial.suggest_int(f"layer-{i}", 16, 4096) for i in range(5)
            ]
            # dropout
            kwargs["ps"] = trial.suggest_uniform(f"ps", 0.0, 0.5)

        elif self.exp in ["shift", "scale", "film"]:
            # cells and cpds embeddings size
            kwargs["emb_sz"] = trial.suggest_int("emb_sz", 16, 512)
            # n_blocks or film layers
            kwargs["n_blocks"] = trial.suggest_int("n_blocks", 1, 5)
            # dropout
            kwargs["ps_emb"] = trial.suggest_uniform("ps_emb", 0.0, 0.5)
            kwargs["ps_film"] = trial.suggest_uniform("ps_film", 0.0, 0.5)
            kwargs["ps_linear"] = trial.suggest_uniform("ps_linear", 0.0, 0.5)

        # learning rate
        kwargs["learning_rate"] = trial.suggest_uniform("learning_rate", 1e-5, 1e-3)
        # weight decat
        kwargs["weight_decay"] = trial.suggest_uniform("weight_decay", 1e-5, 1e-3)

        return kwargs

    def __call__(self, trial):
        # torch.cuda.empty_cache()

        # Suggestions
        start = datetime.now()
        print(f"Training on fold {self.args.fold}")
        trial.set_user_attr("fold", self.args.fold)
        kwargs = self.suggestions(trial)

        # Force model params
        kwargs["fold"] = self.args.fold
        kwargs["exp"] = self.exp

        # DataModule
        dm = ChemProbeDataModule.from_argparse_args(self.args)

        # Model
        if self.exp in ["concat", "id"]:
            model = ConcatNetwork(**kwargs)
        elif self.exp in ["shift", "scale", "film"]:
            model = ChemProbe(**kwargs)

        # Callbacks
        logger = TensorBoardLogger(
            save_dir=self.logs,
            name=f"exp={args.exp}-fold={args.fold}",
            version=f"trial={trial.number}",
        )
        early_stop = EarlyStopping(
            monitor="val_R2Score",
            min_delta=1e-4,
            patience=12,
            verbose=False,
            mode="max",
        )
        checkpoint_callback = ModelCheckpoint(
            monitor="val_R2Score", mode="max", save_last=True
        )

        # Trainer
        trainer = Trainer.from_argparse_args(
            self.args,
            strategy="ddp_find_unused_parameters_false",
            replace_sampler_ddp=False,
            default_root_dir=logger.log_dir,
            logger=logger,
            callbacks=[
                PyTorchLightningPruningCallback(trial, monitor="val_R2Score"),
                early_stop,
                checkpoint_callback,
            ],
            profiler=None,
            deterministic=True,
            precision=16,
        )
        trainer.fit(model, dm)

        # save and clean up gpu
        val_R2Score = trainer.callback_metrics["val_R2Score"].item()
        del dm, model, trainer
        torch.cuda.empty_cache()

        print(
            "Completed fold {} in {}".format(
                self.args.fold, str(datetime.now() - start)
            )
        )
        print(f"Fold val_R2Score: {val_R2Score}")

        return val_R2Score


def process(args):
    pruner: optuna.pruners.BasePruner = (
        optuna.pruners.MedianPruner(
            n_startup_trials=5, n_warmup_steps=10, n_min_trials=3
        )
        if args.prune
        else optuna.pruners.NopPruner()
    )

    # Create study
    # args.study_path = args.study_path.joinpath(f"exp={args.exp}_fold={args.fold}")
    study = activate_study(args.study_path, pruner)
    objective = Objective(args)
    study.optimize(objective, n_trials=args.ntrials)

    print("Number of finished trials: {}".format(len(study.trials)))

    print("Best trial:")
    trial = study.best_trial

    print("  Value: {}".format(trial.value))

    print("  Params: ")
    for key, value in trial.params.items():
        print("    {}: {}".format(key, value))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Optuna hyperparam optimization.")
    parser.add_argument(
        "--study_path",
        type=Path,
        help="Path to SQLite database location (must be on disk - no NFS).",
    )
    parser.add_argument(
        "--ntrials",
        type=int,
        default=2,
        help="Number of trials to run on objective function.",
    )
    parser.add_argument(
        "--prune",
        "-p",
        action="store_true",
        help="Activate the pruning feature. `MedianPruner` stops unpromising "
        "trials at the early stages of training.",
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

    # Trainer args
    parser = Trainer.add_argparse_args(parser)

    # Parse
    args = parser.parse_args()

    sys.exit(process(args))
