#
# SPDX-FileCopyrightText: Copyright © 2024 Idiap Research Institute <contact@idiap.ch>
#
# SPDX-FileContributor: Fabio Fehr <fabio.fehr@idiap.ch>
#
# SPDX-License-Identifier: GPL-3.0-only
#

# Load any environment variables
from dotenv import load_dotenv

load_dotenv()

import argparse
import os
from datetime import datetime

import lightning.pytorch as pl
import torch
from pytorch_lightning.loggers import WandbLogger

from data_modules.SummarisationDataModule import SummarisationDataModule
from models_pl.bart_lightning import BartLightning
from models_pl.nvibart_lightning import NviBartLightning
from utils import create_or_load_model


def load_empirical_distribution(args):
    """Load empirical distribution for the NVIB summarisation model."""

    if args.emp_data is None:
        args.emp_data = "gaussian"  # No empirical distribution
    # Load empirical distribution
    empirical_distribution_path = os.path.join(
        args.output_dir,
        args.project_name,
        "empirical_priors",
        args.emp_data
        + "_"
        + args.model.replace("NVIB", "")
        + "_train_perc"
        + str(args.emp_perc)
        + "_"
        + "embedding_stats.pt",
    )
    print(empirical_distribution_path)
    if os.path.exists(empirical_distribution_path):
        print("Loading empirical distribution from: ", empirical_distribution_path)
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        empirical_distribution = torch.load(empirical_distribution_path, map_location=device)

        # Encoder
        args.prior_mus_encoder = empirical_distribution["encoder_means"]
        args.prior_vars_encoder = empirical_distribution["encoder_var"]
        args.prior_log_alphas_encoder = empirical_distribution["mean_of_encoder_scaled_l2norm2"]
        args.prior_log_alpha_stdevs_encoder = empirical_distribution["log_alpha_encoder_std"]

        # Decoder
        args.prior_mus_decoder = empirical_distribution["decoder_means"]
        args.prior_vars_decoder = empirical_distribution["decoder_var"]
        args.prior_log_alphas_decoder = empirical_distribution["mean_of_decoder_scaled_l2norm2"]
        args.prior_log_alpha_stdevs_decoder = empirical_distribution["log_alpha_decoder_std"]

        # Cross
        args.prior_mus_cross = empirical_distribution["cross_means"]
        args.prior_vars_cross = empirical_distribution["cross_var"]
        args.prior_log_alphas_cross = empirical_distribution["mean_of_cross_scaled_l2norm2"]
        args.prior_log_alpha_stdevs_cross = empirical_distribution["log_alpha_cross_std"]

    else:
        print("No empirical distribution")


def main(args):
    """Main function to evaluate summarisation models."""

    START_TIME = datetime.now().replace(microsecond=0)
    dict_args = vars(args)
    pl.seed_everything(args.seed)
    OUTPUT_PATH = os.path.join(args.output_dir, args.project_name, args.experiment_name)
    args.output_path = OUTPUT_PATH

    # Explicit model path for evaluation - if None then use instantiated model
    MODEL_PATH = args.model_path

    # Select model
    model = {
        "BART-LARGE-CNN": BartLightning,
        "BART-LARGE-XSUM": BartLightning,
        "NVIBBART-LARGE-CNN": NviBartLightning,
        "NVIBBART-LARGE-XSUM": NviBartLightning,
    }[args.model]

    # If NVIB model then load empirical distribution
    if "NVIB" in args.model:
        load_empirical_distribution(args)

    # Load best model
    model, wandb_id = create_or_load_model(OUTPUT_PATH, MODEL_PATH, model, args)

    # Make data module
    dm = SummarisationDataModule(
        model, fp16=True if args.quantisation == "16bit" else False, **dict_args
    )

    # WandB logger
    wandb_logger = WandbLogger(project=args.project_name, id=wandb_id, log_model="None")
    wandb_logger.log_hyperparams(args)

    # Trainer
    trainer = pl.Trainer(
        # limit_val_batches=1,
        # limit_test_batches=1,
        # deterministic=True,
        accelerator="auto",
        logger=wandb_logger,
        precision=16 if args.quantisation == "16bit" else 32,
    )

    # Evaluate model
    model.eval()

    # validation
    START_TIME = datetime.now().replace(microsecond=0)
    if not args.test:
        print("Start validation")
        trainer.validate(model, datamodule=dm)
        val_time = (datetime.now().replace(microsecond=0) - START_TIME).total_seconds()
        print("Validation: ", datetime.now().replace(microsecond=0) - START_TIME)
        wandb_logger.log_metrics({"validation_time": val_time})
    # test
    else:
        trainer.test(model, datamodule=dm)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    # Paths + Naming
    parser.add_argument(
        "--experiment_name",
        default="initial_experiment",
        type=str,
        help="Experiment name",
    )
    parser.add_argument(
        "--project_name",
        default="local_experiments",
        type=str,
        help="Project name for wandb",
    )
    parser.add_argument("--output_dir", default="outputs", type=str, help="Output directory")
    parser.add_argument(
        "--model_path", default=None, type=str, help="Path to specific model checkpoint"
    )

    # Data
    parser.add_argument(
        "--data",
        type=str,
        choices=["cnn_dailymail", "xsum", "wikihow", "curation", "samsum"],
        help="Dataset options",
    )
    parser.add_argument(
        "--emp_data",
        type=str,
        default="gaussian",
        choices=[
            "cnn_dailymail",
            "xsum",
            "gaussian",
            "wikihow",
            "curation",
            "samsum",
        ],
        help="Empirical dataset options",
    )
    parser.add_argument(
        "--emp_perc",
        type=float,
        default=1.0,
        help="Amount of empirical data to use",
    )

    parser.add_argument(
        "--num_workers", type=int, default=4, help="Number of workers for processing"
    )
    parser.add_argument(
        "--val_perc",
        type=float,
        default=1,
    )

    # Model
    parser.add_argument(
        "--model",
        default="BART-LARGE-XSUM",
        help="Model selection",
    )
    # Quantisation
    parser.add_argument(
        "--quantisation",
        default=None,
        help="quantisation selection, 4bit, 8bit, 16bit",
    )

    # NVIB
    # Not used in this work
    parser.add_argument("--kld_lambda", type=float, default=1, help="KL dirichlet lambda")
    parser.add_argument("--klg_lambda", type=float, default=1, help="KL gaussian lambda")
    parser.add_argument(
        "--mu_tau",
        type=float,
        default=1,
        help="1 is the posterior, 0 is the empirical prior.",
    )

    # Used in this work
    parser.add_argument(
        "--alpha_tau_e",
        type=float,
        default=10,
        help="Alpha tau encoder controls the influence of the empirical prior on the posterior.",
    )
    parser.add_argument(
        "--alpha_tau_c",
        type=float,
        default=10,
        help="Alpha tau cross controls the influence of the empirical prior on the posterior.",
    )
    parser.add_argument(
        "--alpha_tau_d",
        type=float,
        default=10,
        help="Alpha tau decoder controls the influence of the empirical prior on the posterior.",
    )
    parser.add_argument(
        "--stdev_tau_e",
        type=float,
        default=0,
        help="Stdev tau encoder controls the influence of the interpolation between query and value.",
    )
    parser.add_argument(
        "--stdev_tau_c",
        type=float,
        default=0,
        help="Stdev tau cross controls the influence of the interpolation between query and value.",
    )
    parser.add_argument(
        "--stdev_tau_d",
        type=float,
        default=0,
        help="Stdev tau decoder controls the influence of the interpolation between query and value.",
    )

    # Evaluation
    parser.add_argument("--seed", default=42, type=int)
    parser.add_argument(
        "--fast_dev_run",
        action="store_true",
        help="Touches all train, validation and test scripts for debugging",
    )
    parser.add_argument("--test", action="store_true", help="Flag for testing")
    parser.add_argument("--batch_size", default=1, type=int)

    args = parser.parse_args()

    main(args)
