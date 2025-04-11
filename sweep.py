#
# SPDX-FileCopyrightText: Copyright © 2024 Idiap Research Institute <contact@idiap.ch>
#
# SPDX-FileContributor: Fabio Fehr <fabio.fehr@idiap.ch>
#
# SPDX-License-Identifier: GPL-3.0-only
#

from dotenv import load_dotenv

load_dotenv()
import argparse
import os

import lightning.pytorch as pl
import torch
import yaml
from pytorch_lightning.loggers import WandbLogger

import wandb
from data_modules.SummarisationDataModule import SummarisationDataModule
from data_modules.TranslationDataModule import TranslationDataModule
from models_pl.bart_lightning import BartLightning
from models_pl.marian_lightning import MarianLightning
from models_pl.nvibart_lightning import NviBartLightning
from models_pl.nvibmarian_lightning import NvibMarianLightning
from utils import create_or_load_model


def main(args):
    """Main function for hyperparameter sweeps."""

    OUTPUT_PATH = os.path.join(args["output_dir"], args["project_name"], args["experiment_name"])
    args["output_path"] = OUTPUT_PATH

    if not os.path.exists(OUTPUT_PATH):
        os.makedirs(OUTPUT_PATH)

    # If sweep id exists then load it
    if os.path.exists(os.path.join(OUTPUT_PATH, "sweep_id.txt")):
        print("Sweep id exists, loading sweep", os.path.join(OUTPUT_PATH, "sweep_id.txt"))
        with open(os.path.join(OUTPUT_PATH, "sweep_id.txt")) as f:
            sweep_id = f.read()
    else:
        print("Sweep id does not exist, creating sweep")

        # Sweep config
        parms = ("method", "metric", "parameters", "entity")
        sweep_config = {k: args[k] for k in parms if k in args}

        # Create sweep
        sweep_id = wandb.sweep(
            sweep=sweep_config, project=args["project_name"], entity=args["entity"]
        )

        # Save sweep id
        with open(os.path.join(OUTPUT_PATH, "sweep_id.txt"), "w") as f:
            f.write(sweep_id)

    # Local scope but define the objective function
    def objective():
        """Objective function for hyperparameter sweeps."""
        # Use global arguments
        global args
        # Initialise wandb - This gives us our sweep parameters
        wandb.init(project=args["project_name"], entity=args["entity"])
        # update args with wandb config
        args.update(wandb.config)

        # Update experiment name with wandb config
        for key, value in wandb.config.items():
            args["experiment_name"] = args["experiment_name"] + "_" + key + str(round(value, 2))
        args["output_path"] = os.path.join(
            args["output_dir"], args["project_name"], args["experiment_name"]
        )

        # Set seed
        pl.seed_everything(args["seed"])
        # Explicit model path for evaluation - if None then use instantiated model
        MODEL_PATH = args["model_path"]

        if "MARIAN" in args["model"]:
            translation_flag = True
        else:
            translation_flag = False
        # Select model
        model = {
            "BART-LARGE-CNN": BartLightning,
            "BART-LARGE-XSUM": BartLightning,
            "NVIBBART-LARGE-CNN": NviBartLightning,
            "NVIBBART-LARGE-XSUM": NviBartLightning,
            "MARIAN": MarianLightning,
            "NVIBMARIAN": NvibMarianLightning,
        }[args["model"]]

        # If NVIB model then load empirical distribution
        if "NVIB" in args["model"]:
            if args["emp_data"] is None:
                args["emp_data"] = "gaussian"  # No empirical distribution
            # Load empirical distribution
            if translation_flag:
                empirical_distribution_path = os.path.join(
                    args["output_dir"],
                    args["project_name"],
                    "empirical_priors",
                    args["emp_data"]
                    + "_"
                    + args["model"].replace("NVIB", "")
                    + "_train_perc"
                    + str(args["emp_perc"])
                    + "_"
                    + args["src_lang"]
                    + "_"
                    + args["tgt_lang"]
                    + "_"
                    + "embedding_stats.pt",
                )
            else:
                empirical_distribution_path = os.path.join(
                    args["output_dir"],
                    args["project_name"],
                    "empirical_priors",
                    args["emp_data"]
                    + "_"
                    + args["model"].replace("NVIB", "")
                    + "_"
                    + "train_perc"
                    + str(args["emp_perc"])
                    + "_"
                    + "embedding_stats.pt",
                )
            if os.path.exists(empirical_distribution_path):
                print("Loading empirical distribution from: ", empirical_distribution_path)
                device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
                empirical_distribution = torch.load(
                    empirical_distribution_path, map_location=device
                )

                # Encoder
                args["prior_mus_encoder"] = empirical_distribution["encoder_means"]
                args["prior_vars_encoder"] = empirical_distribution["encoder_var"]
                args["prior_log_alphas_encoder"] = empirical_distribution[
                    "mean_of_encoder_scaled_l2norm2"
                ]
                args["prior_log_alpha_stdevs_encoder"] = empirical_distribution[
                    "log_alpha_encoder_std"
                ]

                # Decoder
                args["prior_mus_decoder"] = empirical_distribution["decoder_means"]
                args["prior_vars_decoder"] = empirical_distribution["decoder_var"]
                args["prior_log_alphas_decoder"] = empirical_distribution[
                    "mean_of_decoder_scaled_l2norm2"
                ]
                args["prior_log_alpha_stdevs_decoder"] = empirical_distribution[
                    "log_alpha_decoder_std"
                ]

                # Cross
                args["prior_mus_cross"] = empirical_distribution["cross_means"]
                args["prior_vars_cross"] = empirical_distribution["cross_var"]
                args["prior_log_alphas_cross"] = empirical_distribution[
                    "mean_of_cross_scaled_l2norm2"
                ]
                args["prior_log_alpha_stdevs_cross"] = empirical_distribution[
                    "log_alpha_cross_std"
                ]

            else:
                print("No empirical distribution")

        # Load best model
        model, wandb_id = create_or_load_model(
            args["output_path"], MODEL_PATH, model, argparse.Namespace(**args)
        )
        wandb_logger = WandbLogger(project=args["project_name"], id=wandb_id, log_model="None")
        wandb_logger.log_hyperparams(args)

        # Make data module
        if translation_flag:
            dm = TranslationDataModule(
                model, fp16=True if args["quantisation"] == "16bit" else False, **args
            )
        else:
            dm = SummarisationDataModule(
                model, fp16=True if args["quantisation"] == "16bit" else False, **args
            )

        # Trainer
        trainer = pl.Trainer(
            # limit_val_batches=0.1,
            # limit_test_batches=1,
            # deterministic=True,
            accelerator="auto",
            logger=wandb_logger,
            precision=16 if args["quantisation"] == "16bit" else 32,
        )

        # Evaluate model
        model.eval()
        trainer.validate(model, datamodule=dm)
        trainer.test(model, datamodule=dm)

    # Define the agent
    wandb.agent(
        sweep_id,
        entity=args["entity"],
        project=args["project_name"],
        function=objective,
        count=args["count"],
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    # Config
    parser.add_argument(
        "--config_path",
        default="sweep_configs/sweep_config.yml",
        type=str,
        help="Sweep name",
    )
    args = parser.parse_args()

    # Load the sweep configuration
    with open(args.config_path) as file:
        args = yaml.safe_load(file)

    main(args)
