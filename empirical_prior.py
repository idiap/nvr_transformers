#
# SPDX-FileCopyrightText: Copyright © 2024 Idiap Research Institute <contact@idiap.ch>
#
# SPDX-FileContributor: Fabio Fehr <fabio.fehr@idiap.ch>
#
# SPDX-License-Identifier: GPL-3.0-only
#

# This script calculates the empirical mean and variance of of the latent vectors of the encoder

from dotenv import load_dotenv

load_dotenv()
import argparse
import math
import os

import lightning.pytorch as pl
import torch
from tqdm import tqdm

from data_modules.SummarisationDataModule import SummarisationDataModule
from data_modules.TranslationDataModule import TranslationDataModule
from models_pl.bart_lightning import BartLightning
from models_pl.marian_lightning import MarianLightning
from utils import create_or_load_model


def batch_token_sum(hidden_states, attention_mask, device):
    """
    Sum over the token dim and mask then sum over batch dim
    hidden_states: (batch_size, seq_len, hidden_size)
    attention_mask: (batch_size, seq_len) 1s and 0s
    """
    return torch.sum(
        torch.sum(
            hidden_states.masked_fill_((attention_mask == 0).to(device).unsqueeze(-1), 0),
            dim=1,
        ),
        0,
    )


# Notice not exp - overflow.
def scaled_l2norm2_of_mean(embedding_mean, head_dim):
    """
    The scaled L2 norm squared of the mean
    embedding_mean: (hidden_size)
    head_dim: (1) scalar
    return: (1) scalar
    """
    return torch.sum(embedding_mean**2) / (2 * math.sqrt(head_dim))


def exp_scaled_l2norm2_sum(hidden_states, attention_mask, head_dim, device):
    """Calculate the exponential of the scaled L2 norm squared and sum over the batch, sequence
    length and hidden dimensions."""

    # Sum over sequence length and batches (all tokens)
    return torch.sum(
        # Exponential
        torch.exp(
            # Square, scale and sum over hidden dimensions
            torch.sum(
                (hidden_states.masked_fill((attention_mask == 0).to(device).unsqueeze(-1), 0))
                ** 2,
                dim=-1,
            )
            / (2 * math.sqrt(head_dim)),
        )
    )


def scaled_l2norm2_sum(hidden_states, attention_mask, head_dim, device):
    """Calculate the scaled L2 norm squared and sum over the batch, sequence length and hidden
    dimensions."""
    # Sum over sequence length and batches (all tokens)
    return torch.sum(
        # Square, scale and sum over hidden dimensions
        torch.sum(
            (hidden_states.masked_fill((attention_mask == 0).to(device).unsqueeze(-1), 0)) ** 2,
            dim=-1,
        )
        / (2 * math.sqrt(head_dim)),
    )


def variance(hidden_states, attention_mask, embedding_mean, device):
    """
    Variance over the token dim and mask then sum over batch dim
    hidden_states: (batch_size, seq_len, hidden_size)
    attention_mask: (batch_size, seq_len) 1s and 0s
    """
    return torch.sum(
        torch.sum(
            ((hidden_states - embedding_mean) ** 2).masked_fill_(
                (attention_mask == 0).to(device).unsqueeze(-1), 0
            ),
            dim=1,
        ),
        0,
    )


def l2_norm_variance(hidden_states, attention_mask, embedding_mean, head_dim, device):
    """
    Variance over the token dim and mask then sum over batch dim
    hidden_states: (batch_size, seq_len, hidden_size)
    attention_mask: (batch_size, seq_len) 1s and 0s
    return: (1) scalar
    """
    # Calculate the scaled L2 norm [batch_size, seq_len]
    l2_norm = torch.sum(
        (hidden_states.masked_fill((attention_mask == 0).to(device).unsqueeze(-1), 0)) ** 2,
        dim=-1,
    ) / (2 * math.sqrt(head_dim))

    return torch.sum(
        ((l2_norm - embedding_mean) ** 2).masked_fill((attention_mask == 0).to(device), 0)
    )


def main(args):
    """Calculate the empirical mean and variance and psuedo counts of the latent vectors in the
    encoder and decoder."""

    OUTPUT_PATH = os.path.join(args.output_dir, args.project_name, args.experiment_name)
    args.output_path = OUTPUT_PATH
    dict_args = vars(args)
    pl.seed_everything(args.seed)
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    # Explicit model path for evaluation - if None then use instantiated model
    MODEL_PATH = args.model_path

    # Select model
    model = {
        "BART-LARGE-XSUM": BartLightning,
        "BART-LARGE-CNN": BartLightning,
        "PARABART": BartLightning,
        "MARIAN": MarianLightning,
    }[args.model]

    # Load best model
    model, _ = create_or_load_model(OUTPUT_PATH, MODEL_PATH, model, args)

    # Compute in parallel
    # model = torch.nn.DataParallel(model, device_ids=list(range(torch.cuda.device_count())))
    model.to(device)

    summarisation_flag = False
    translation_flag = False
    paraphrase_flag = False

    # Make data module
    if args.data in [
        "xsum",
        "cnn_dailymail",
        "wikihow",
        "samsum",
        "curation",
    ]:
        summarisation_flag = True
        dm = SummarisationDataModule(
            model, fp16=True if args.quantisation == "16bit" else False, **dict_args
        )
    elif args.data in [
        "iwslt2017",
        "ted_talks_iwslt",
        "bible_para",
        "opus100",
    ]:
        translation_flag = True
        dm = TranslationDataModule(
            model, fp16=True if args.quantisation == "16bit" else False, **dict_args
        )
    else:
        raise ValueError("Data not recognised")

    dm.prepare_data()
    dm.setup("fit")
    if paraphrase_flag:
        dm.setup("empirical_prior")

    # Evaluate model - NB for dropout to be disabled
    model.eval()

    # Running sums
    running_encoder_sum = [
        torch.zeros(model.config.d_model, device=device)
        for _ in range(model.config.encoder_layers)
    ]
    running_decoder_sum = [
        torch.zeros(model.config.d_model, device=device)
        for _ in range(model.config.decoder_layers)
    ]
    running_cross_sum = torch.zeros(model.config.d_model, device=device)

    # Running sum for scaled_l2norm2
    running_encoder_scaled_l2norm2_sum = [
        torch.zeros(1, device=device) for _ in range(model.config.encoder_layers)
    ]
    running_decoder_scaled_l2norm2_sum = [
        torch.zeros(1, device=device) for _ in range(model.config.decoder_layers)
    ]
    running_cross_scaled_l2norm2_sum = torch.zeros(1, device=device)

    running_encoder_lengths = 0
    running_decoder_lengths = 0
    running_encoder_var = [
        torch.zeros(model.config.d_model, device=device)
        for _ in range(model.config.encoder_layers)
    ]
    running_decoder_var = [
        torch.zeros(model.config.d_model, device=device)
        for _ in range(model.config.decoder_layers)
    ]
    running_cross_var = torch.zeros(model.config.d_model, device=device)

    running_encoder_scaled_l2norm2_var_sum = [
        torch.zeros(1, device=device) for _ in range(model.config.encoder_layers)
    ]
    running_decoder_scaled_l2norm2_var_sum = [
        torch.zeros(1, device=device) for _ in range(model.config.decoder_layers)
    ]
    running_cross_scaled_l2norm2_var_sum = torch.zeros(1, device=device)

    # Paths
    # Summarisation paths
    if summarisation_flag:
        mean_path = os.path.join(
            args.output_dir,
            args.project_name,
            args.experiment_name,
            args.data + "_" + args.model + "_train_perc" + str(args.train_perc) + "_" + "mean.pt",
        )

        total_path = os.path.join(
            args.output_dir,
            args.project_name,
            args.experiment_name,
            args.data
            + "_"
            + args.model
            + "_train_perc"
            + str(args.train_perc)
            + "_"
            + "embedding_stats.pt",
        )
    elif paraphrase_flag:
        mean_path = os.path.join(
            args.output_dir,
            args.project_name,
            args.experiment_name,
            "paws_quora"
            + "_"
            + args.model
            + "_train_perc"
            + str(args.train_perc)
            + "_"
            + "mean.pt",
        )
        total_path = os.path.join(
            args.output_dir,
            args.project_name,
            args.experiment_name,
            "paws_quora"
            + "_"
            + args.model
            + "_train_perc"
            + str(args.train_perc)
            + "_"
            + "embedding_stats.pt",
        )
    elif translation_flag:
        mean_path = os.path.join(
            args.output_dir,
            args.project_name,
            args.experiment_name,
            args.data
            + "_"
            + args.model
            + "_train_perc"
            + str(args.train_perc)
            + "_"
            + args.src_lang
            + "_"
            + args.tgt_lang
            + "_"
            + "mean.pt",
        )
        total_path = os.path.join(
            args.output_dir,
            args.project_name,
            args.experiment_name,
            args.data
            + "_"
            + args.model
            + "_train_perc"
            + str(args.train_perc)
            + "_"
            + args.src_lang
            + "_"
            + args.tgt_lang
            + "_"
            + "embedding_stats.pt",
        )

    if os.path.exists(total_path):
        print("Loading embedding statistics from file")
        # Torch load
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        total_dict = torch.load(total_path, map_location=device)
        encoder_means = total_dict["encoder_means"]
        cross_means = total_dict["cross_means"]
        decoder_means = total_dict["decoder_means"]
        cross_means = total_dict["cross_means"]
        running_encoder_lengths = total_dict["encoder_lengths"]
        running_decoder_lengths = total_dict["decoder_lengths"]
        encoder_var = total_dict["encoder_var"]
        decoder_var = total_dict["decoder_var"]
        cross_var = total_dict["cross_var"]
        encoder_scaled_l2norm2_of_means = total_dict["encoder_scaled_l2norm2_of_means"]
        decoder_scaled_l2norm2_of_means = total_dict["decoder_scaled_l2norm2_of_means"]
        cross_scaled_l2norm2_of_mean = total_dict["cross_scaled_l2norm2_of_mean"]
        log_alpha_encoder_std = total_dict["log_alpha_encoder_std"]
        log_alpha_decoder_std = total_dict["log_alpha_decoder_std"]
        log_alpha_cross_std = total_dict["log_alpha_cross_std"]

    else:
        print("Creating embedding statistics")

        if os.path.exists(mean_path):
            print("Loading mean from file")
            # Torch load
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            mean_dict = torch.load(mean_path, map_location=device)
            encoder_means = mean_dict["encoder_means"]
            decoder_means = mean_dict["decoder_means"]
            cross_means = mean_dict["cross_means"]
            mean_of_encoder_scaled_l2norm2 = mean_dict["mean_of_encoder_scaled_l2norm2"]
            mean_of_decoder_scaled_l2norm2 = mean_dict["mean_of_decoder_scaled_l2norm2"]
            mean_of_cross_scaled_l2norm2 = mean_dict["mean_of_cross_scaled_l2norm2"]
            running_encoder_lengths = mean_dict["encoder_lengths"]
            running_decoder_lengths = mean_dict["decoder_lengths"]

        else:
            print("Creating mean statistics")
            with torch.no_grad():
                for batch in tqdm(dm.train_dataloader()):
                    batch.to(device)

                    # Forward pass
                    out = model.model(**batch)

                    # Total lengths (where not pads)
                    decoder_attention_mask = (batch["labels"] != model.tokenizer.pad_token_id) * 1
                    running_encoder_lengths += (
                        batch["input_ids"] != model.tokenizer.pad_token_id
                    ).sum()
                    running_decoder_lengths += (
                        batch["labels"] != model.tokenizer.pad_token_id
                    ).sum()

                    # Sum over all tokens and all batches
                    # Encoder layers
                    current_encoder_sum = list(
                        map(
                            batch_token_sum,
                            out.encoder_hidden_states[: model.config.encoder_layers],
                            [batch["attention_mask"]] * model.config.encoder_layers,
                            [device] * model.config.encoder_layers,
                        )
                    )
                    running_encoder_sum = [
                        running_encoder_sum[i] + current_encoder_sum[i]
                        for i in range(len(running_encoder_sum))
                    ]

                    # Decoder layers
                    current_decoder_sum = list(
                        map(
                            batch_token_sum,
                            out.decoder_hidden_states[: model.config.decoder_layers],
                            [decoder_attention_mask] * model.config.decoder_layers,
                            [device] * model.config.decoder_layers,
                        )
                    )
                    running_decoder_sum = [
                        running_decoder_sum[i] + current_decoder_sum[i]
                        for i in range(len(running_decoder_sum))
                    ]

                    # Cross attention encoder layers
                    current_cross_sum = batch_token_sum(
                        out.encoder_last_hidden_state, batch["attention_mask"], device
                    )
                    running_cross_sum += current_cross_sum

                    ############################################################
                    # This is the expectation of the scaled L2 norm
                    ############################################################
                    # Encoder scaled_l2norm2_sum
                    current_encoder_scaled_l2norm2_sum = list(
                        map(
                            scaled_l2norm2_sum,
                            out.encoder_hidden_states[: model.config.encoder_layers],
                            [batch["attention_mask"]] * model.config.encoder_layers,
                            [model.config.d_model / model.config.encoder_attention_heads]
                            * model.config.encoder_layers,
                            [device] * model.config.encoder_layers,
                        )
                    )

                    running_encoder_scaled_l2norm2_sum = [
                        running_encoder_scaled_l2norm2_sum[i]
                        + current_encoder_scaled_l2norm2_sum[i]
                        for i in range(len(running_encoder_scaled_l2norm2_sum))
                    ]

                    # Decoder scaled_l2norm2_sum
                    current_decoder_scaled_l2norm2_sum = list(
                        map(
                            scaled_l2norm2_sum,
                            out.decoder_hidden_states[: model.config.decoder_layers],
                            [decoder_attention_mask] * model.config.decoder_layers,
                            [model.config.d_model / model.config.decoder_attention_heads]
                            * model.config.decoder_layers,
                            [device] * model.config.decoder_layers,
                        )
                    )
                    running_decoder_scaled_l2norm2_sum = [
                        running_decoder_scaled_l2norm2_sum[i]
                        + current_decoder_scaled_l2norm2_sum[i]
                        for i in range(len(running_decoder_scaled_l2norm2_sum))
                    ]

                    # Cross attention scaled_l2norm2_sum
                    current_cross_scaled_l2norm2_sum = scaled_l2norm2_sum(
                        out.encoder_last_hidden_state,
                        batch["attention_mask"],
                        model.config.d_model / model.config.decoder_attention_heads,
                        device,
                    )
                    running_cross_scaled_l2norm2_sum += current_cross_scaled_l2norm2_sum
                    # break

                # Calculate mean
                encoder_means = [
                    encoder_sum / running_encoder_lengths for encoder_sum in running_encoder_sum
                ]
                decoder_means = [
                    decoder_sum / running_decoder_lengths for decoder_sum in running_decoder_sum
                ]
                cross_means = running_cross_sum / running_encoder_lengths

                # Calculate mean_of_scaled_l2norm2
                mean_of_encoder_scaled_l2norm2 = [
                    encoder_sum / running_encoder_lengths
                    for encoder_sum in running_encoder_scaled_l2norm2_sum
                ]
                mean_of_decoder_scaled_l2norm2 = [
                    decoder_sum / running_decoder_lengths
                    for decoder_sum in running_decoder_scaled_l2norm2_sum
                ]
                mean_of_cross_scaled_l2norm2 = (
                    running_cross_scaled_l2norm2_sum / running_encoder_lengths
                )

            mean_dict = {
                "encoder_means": encoder_means,
                "decoder_means": decoder_means,
                "cross_means": cross_means,
                "mean_of_encoder_scaled_l2norm2": mean_of_encoder_scaled_l2norm2,
                "mean_of_decoder_scaled_l2norm2": mean_of_decoder_scaled_l2norm2,
                "mean_of_cross_scaled_l2norm2": mean_of_cross_scaled_l2norm2,
                "encoder_lengths": running_encoder_lengths,
                "decoder_lengths": running_decoder_lengths,
            }

            torch.save(mean_dict, mean_path)

        ############################################################
        # This is the scaled L2 norm of the expectation
        ############################################################
        encoder_scaled_l2norm2_of_means = list(
            map(
                scaled_l2norm2_of_mean,
                encoder_means,
                [model.config.d_model / model.config.encoder_attention_heads] * len(encoder_means),
            )
        )
        decoder_scaled_l2norm2_of_means = list(
            map(
                scaled_l2norm2_of_mean,
                decoder_means,
                [model.config.d_model / model.config.decoder_attention_heads] * len(decoder_means),
            )
        )
        cross_scaled_l2norm2_of_mean = scaled_l2norm2_of_mean(
            cross_means, model.config.d_model / model.config.decoder_attention_heads
        )

        ############################################################
        # This is the variance
        ############################################################

        with torch.no_grad():
            print("Calculating variance")
            for batch in tqdm(dm.train_dataloader()):
                batch.to(device)
                # Forward pass
                out = model.model(**batch)

                decoder_attention_mask = (batch["labels"] != model.tokenizer.pad_token_id) * 1

                # Encoder layers

                # variance(out.encoder_hidden_states[: model.config.encoder_layers][0], batch["attention_mask"], encoder_means[0], device)
                current_encoder_var_sum = list(
                    map(
                        variance,
                        out.encoder_hidden_states[: model.config.encoder_layers],
                        [batch["attention_mask"]] * model.config.encoder_layers,
                        encoder_means,
                        [device] * model.config.encoder_layers,
                    )
                )
                running_encoder_var = [
                    running_encoder_var[i] + current_encoder_var_sum[i]
                    for i in range(len(running_encoder_var))
                ]

                # Decoder layers
                current_decoder_var_sum = list(
                    map(
                        variance,
                        out.decoder_hidden_states[: model.config.decoder_layers],
                        [decoder_attention_mask] * model.config.decoder_layers,
                        decoder_means,
                        [device] * model.config.decoder_layers,
                    )
                )
                running_decoder_var = [
                    running_decoder_var[i] + current_decoder_var_sum[i]
                    for i in range(len(running_decoder_var))
                ]

                # Cross attention encoder layers
                current_cross_var_sum = variance(
                    out.encoder_last_hidden_state,
                    batch["attention_mask"],
                    cross_means,
                    device,
                )
                running_cross_var += current_cross_var_sum

                ############################################################
                # Log alpha variance calculation
                ############################################################

                # Encoder variance between log alpha and expected log alpha
                current_encoder_scaled_l2norm2_var_sum = list(
                    map(
                        l2_norm_variance,
                        out.encoder_hidden_states[: model.config.encoder_layers],
                        [batch["attention_mask"]] * model.config.encoder_layers,
                        mean_of_encoder_scaled_l2norm2,
                        [model.config.d_model / model.config.decoder_attention_heads]
                        * model.config.encoder_layers,
                        [device] * model.config.encoder_layers,
                    )
                )
                # Encoder running variance sum
                running_encoder_scaled_l2norm2_var_sum = [
                    running_encoder_scaled_l2norm2_var_sum[i]
                    + current_encoder_scaled_l2norm2_var_sum[i]
                    for i in range(len(current_encoder_scaled_l2norm2_var_sum))
                ]

                # Decoder variance between log alpha and expected log alpha
                current_decoder_scaled_l2norm2_var_sum = list(
                    map(
                        l2_norm_variance,
                        out.decoder_hidden_states[: model.config.decoder_layers],
                        [decoder_attention_mask] * model.config.decoder_layers,
                        mean_of_decoder_scaled_l2norm2,
                        [model.config.d_model / model.config.decoder_attention_heads]
                        * model.config.decoder_layers,
                        [device] * model.config.decoder_layers,
                    )
                )
                # Decoder running variance sum
                running_decoder_scaled_l2norm2_var_sum = [
                    running_decoder_scaled_l2norm2_var_sum[i]
                    + current_decoder_scaled_l2norm2_var_sum[i]
                    for i in range(len(current_decoder_scaled_l2norm2_var_sum))
                ]

                # Cross attention layers
                # Cross variance between log alpha and expected log alpha
                current_cross_scaled_l2norm2_var_sum = l2_norm_variance(
                    out.encoder_last_hidden_state,
                    batch["attention_mask"],
                    mean_of_cross_scaled_l2norm2,
                    model.config.d_model / model.config.decoder_attention_heads,
                    device,
                )
                # Cross running variance sum
                running_cross_scaled_l2norm2_var_sum = (
                    running_cross_scaled_l2norm2_var_sum + current_cross_scaled_l2norm2_var_sum
                )
                # break
        total_dict = {
            "encoder_means": encoder_means,
            "decoder_means": decoder_means,
            "cross_means": cross_means,
            "encoder_lengths": running_encoder_lengths,
            "decoder_lengths": running_decoder_lengths,
            "encoder_var": [
                encoder_var / (running_encoder_lengths - 1) for encoder_var in running_encoder_var
            ],
            "decoder_var": [
                decoder_var / (running_decoder_lengths - 1) for decoder_var in running_decoder_var
            ],
            "cross_var": running_cross_var / (running_encoder_lengths - 1),
            "mean_of_encoder_scaled_l2norm2": mean_of_encoder_scaled_l2norm2,
            "mean_of_decoder_scaled_l2norm2": mean_of_decoder_scaled_l2norm2,
            "mean_of_cross_scaled_l2norm2": mean_of_cross_scaled_l2norm2,
            "encoder_scaled_l2norm2_of_means": encoder_scaled_l2norm2_of_means,
            "decoder_scaled_l2norm2_of_means": decoder_scaled_l2norm2_of_means,
            "cross_scaled_l2norm2_of_mean": cross_scaled_l2norm2_of_mean,
            "log_alpha_encoder_std": [
                torch.sqrt(log_alpha_encoder_var / (running_encoder_lengths - 1))
                for log_alpha_encoder_var in running_encoder_scaled_l2norm2_var_sum
            ],
            "log_alpha_decoder_std": [
                torch.sqrt(log_alpha_decoder_var / (running_decoder_lengths - 1))
                for log_alpha_decoder_var in running_decoder_scaled_l2norm2_var_sum
            ],
            "log_alpha_cross_std": torch.sqrt(
                running_cross_scaled_l2norm2_var_sum / (running_encoder_lengths - 1)
            ),
        }
        print("Saving embedding stats to: ", total_path)
        torch.save(total_dict, total_path)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    # Paths + Naming
    parser.add_argument(
        "--experiment_name",
        default="empirical_priors",
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
    parser.add_argument("--data", type=str, default="xsum", help="Dataset")
    parser.add_argument(
        "--num_workers", type=int, default=4, help="Number of workers for processing"
    )
    parser.add_argument(
        "--train_perc",
        type=float,
        default=1.0,
    )
    parser.add_argument("--src_lang", type=str, default="en", help="Source language")
    parser.add_argument("--tgt_lang", type=str, default="de", help="Target language")

    # Model
    parser.add_argument(
        "--model",
        default="BART-LARGE-XSUM",
        help="Model selection",
    )
    parser.add_argument("--quantisation", default=None, help="Quantisation method")

    # Evaluation
    parser.add_argument("--seed", default=42, type=int)
    parser.add_argument("--batch_size", default=1, type=int)
    parser.add_argument(
        "--output_hidden_states", action="store_true", help="Output the hidden states"
    )

    args = parser.parse_args()

    main(args)
