#
# SPDX-FileCopyrightText: Copyright © 2024 Idiap Research Institute <contact@idiap.ch>
#
# SPDX-FileContributor: Fabio Fehr <fabio.fehr@idiap.ch>
#
# SPDX-License-Identifier: GPL-3.0-only
#

import glob
import math
import os
import pickle

import matplotlib.pyplot as plt
import numpy as np
import plotly.express as px
import torch
import wandb
from scipy.spatial.distance import jensenshannon as js_distance


def get_checkpoint_path(path):
    """Get the checkpoint path from the directory."""
    ckpt_lst = glob.glob(os.path.join(path, "epoch*step*.ckpt"))
    if len(ckpt_lst) != 0:
        checkpoint_path = ckpt_lst[0]
    else:
        checkpoint_path = None

    return checkpoint_path


def get_best_model_path(path):
    """Get the best model checkpoint path from the directory."""
    ckpt_lst = glob.glob(os.path.join(path, "best_model.ckpt"))
    if len(ckpt_lst) != 0:
        checkpoint_path = ckpt_lst[0]
    else:
        checkpoint_path = None

    return checkpoint_path


def load_model(checkpoint_path, model_name, args):
    """Load model from checkpoint."""
    print("Loading model: ", checkpoint_path)
    model = model_name.load_from_checkpoint(checkpoint_path, args=args, strict=False)
    return model


def create_or_load_model(output_path, checkpoint_path, model_name, args):
    """Create or load model from checkpoint.

    Return model and wandb_id.
    """
    # Create model and load from checkpoint
    if checkpoint_path is not None:
        print("Loading model")
        model = load_model(checkpoint_path, model_name, args=args)
    else:
        print("Creating model")
        if not os.path.isdir(output_path):
            os.makedirs(output_path)
        model = model_name(args)

    if os.path.exists(os.path.join(output_path, "wandb_id.txt")):
        print("Loading W&B ID")
        wandb_id = open(os.path.join(output_path, "wandb_id.txt")).read()
    else:
        print("Creating W&B ID")
        if not os.path.isdir(output_path):
            os.makedirs(output_path)
        wandb_id = wandb.util.generate_id()
        with open(os.path.join(output_path, "wandb_id.txt"), "w") as f:
            f.write(wandb_id)

    return model, wandb_id


def show_attention(
    batch,
    batch_item,
    tokenizer,
    attentions_all,
    attention_type,
    logger,
    val_or_test,
    zmax=None,
    prior=None,
    num_heads=1,
    num_layers=1,
    batch_idx=None,
    observation=None,
):
    """Show attention maps."""
    # num_layers = len(model_outputs["cross_attentions"])
    # num_heads = model_outputs["cross_attentions"][0].shape[1]

    attentions = attentions_all[attention_type]

    # Cross attention, first layer, pooled across all heads
    # size: bsz, num_heads, tgt_len, src_len

    # INPUT SENTENCE
    if attention_type == "encoder_attentions" or attention_type == "cross_attentions":
        input_sentence = tokenizer.batch_decode(
            batch["input_ids"][batch_item, :], skip_special_tokens=True
        )
        # Strip "" which are skipped tokens
        input_sentence = [x for x in input_sentence if x != ""]
        input_sentence = ["<s>"] + input_sentence + [r"<\s>"]

    # attention_type == "decoder_attentions"
    else:
        input_sentence = tokenizer.batch_decode(
            batch["labels"][batch_item, :], skip_special_tokens=True
        )
        # Strip "" which are skipped tokens
        input_sentence = [x for x in input_sentence if x != ""]
        input_sentence = ["<s>"] + input_sentence + [r"<\s>"]

    # OUTPUT SENTENCE
    # Self attention
    if attention_type == "encoder_attentions" or attention_type == "decoder_attentions":
        output_sentence = input_sentence

    # Cross attention
    else:
        output_sentence = tokenizer.batch_decode(
            batch["labels"][batch_item, :], skip_special_tokens=True
        )
        # Strip "" which are skipped tokens
        output_sentence = [x for x in output_sentence if x != ""]

    # Add prior for NVIB models
    if prior is not None:
        input_sentence = [prior] + input_sentence

    # Make the tokens unique by adding a count
    count = 1
    for i in range(0, len(input_sentence)):
        token = input_sentence[i]
        if (token != "<s>") or (token != r"<\s>") or (token != prior):
            input_sentence[i] = token + "_" + str(count)
            count += 1
    count = 1
    for i in range(0, len(output_sentence)):
        token = output_sentence[i]
        if (token != "<s>") or (token != r"<\s>"):
            output_sentence[i] = token + "_" + str(count)
            count += 1

    # Plot layers and heads
    for layer in range(num_layers):
        for head in range(num_heads):
            if num_heads == 1:
                cross_attentions = attentions[layer][batch_item, :, :, :].mean(dim=0)
            else:
                cross_attentions = attentions[layer][batch_item, head, :, :]
            fig = px.imshow(
                cross_attentions[: len(output_sentence), : len(input_sentence)].detach().cpu(),
                labels=dict(x="K", y="Q", color="Score"),
                zmax=zmax,
                zmin=0,
                x=input_sentence,
                y=output_sentence,
            )
            # Access the wandb logger to log a figure
            if num_heads == 1:
                logger.experiment.log(
                    {
                        f"{val_or_test} - {attention_type} Map - Layer {layer} - Pooled Heads - Batch {batch_idx} - Obs {observation}": wandb.Plotly(
                            fig
                        )
                    }
                )
            else:
                logger.experiment.log(
                    {
                        f"{val_or_test} - Attention Map - Layer {layer} - Head {head} - Batch {batch_idx} - Obs {observation}": wandb.Plotly(
                            fig
                        )
                    }
                )


def batched_jensen_shannon_divergence(x, y):
    """Compute the Jensen-Shannon divergence between two distributions in a batched format."""

    # Calculate the batch size and tensor dimensions
    batch_size, head_dim, tgt_dim, src_dim = x.shape

    # Flatten the distributions
    x = x.reshape(batch_size * head_dim * tgt_dim, src_dim).to("cpu")
    y = y.reshape(batch_size * head_dim * tgt_dim, src_dim).to("cpu")

    lengths = torch.count_nonzero(y, dim=1) + 1

    jsd = list(
        map(
            lambda i: js_distance(x[i, 0 : lengths[i]], y[i, 0 : lengths[i]]),
            range(len(lengths)),
        )
    )

    # Convert nans to zero and square the values for the Jensen-Shannon divergence
    jsd = [0 if (math.isnan(x) or math.isinf(x)) else x ** 2 for x in jsd]

    # Compute the average Jensen-Shannon divergence over heads and batchs
    return sum(jsd) / len(jsd)


def calculate_js_div(path1, path2):
    """Calculate the Jensen-Shannon divergence between two attention distributions."""

    with open(path1, "rb") as f:
        attn1 = pickle.load(f)
    with open(path2, "rb") as f:
        attn2 = pickle.load(f)

    if attn1[0][0].shape != attn2[0][0].shape:
        prior_matrix = torch.zeros(attn1[0][0].shape, device=attn1[0][0].device)[
            :, :, :, 0
        ].unsqueeze(-1)

    result = list(
        map(
            lambda b: list(
                map(
                    lambda lam: batched_jensen_shannon_divergence(
                        attn1[b][lam],
                        torch.cat((prior_matrix, attn2[b][lam][:, :, :, :]), dim=-1),
                    ),
                    range(len(attn1[b])),
                )
            ),
            range(len(attn1)),
        )
    )[0]

    # return average js divergence
    return (sum(result) / len(result)).item()
