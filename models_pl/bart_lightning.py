#
# SPDX-FileCopyrightText: Copyright © 2024 Idiap Research Institute <contact@idiap.ch>
#
# SPDX-FileContributor: Fabio Fehr <fabio.fehr@idiap.ch>
#
# SPDX-License-Identifier: GPL-3.0-only
#

import os

from transformers import AutoConfig, AutoTokenizer

from models_hf.modeling_bart import BartForConditionalGeneration
from models_pl.seq2seq_lightning import Seq2SeqLightning


def clean_bart_tokens(tokens_list):
    """Clean BART tokens."""
    return [token.replace("Ġ", "").replace("Ċ", "<n>") for token in tokens_list]


class BartLightning(Seq2SeqLightning):
    """Lighting module for BART, inherits from a base sequence-to-sequence model."""

    def __init__(self, args, **kwargs):
        super().__init__(args, **kwargs)

        # Select model
        if args.model == "BART-LARGE-CNN":
            model_name = "facebook/bart-large-cnn"
            self.log_rouge = True

        elif args.model == "BART-LARGE-XSUM":
            model_name = "facebook/bart-large-xsum"
            self.log_rouge = True
        else:
            model_name = "facebook/bart-base"

        self.config = AutoConfig.from_pretrained(
            model_name,
        )

        self.config.output_path = args.output_path

        # Output attention score and hidden states for plotting
        self.config.output_attentions = True
        if hasattr(args, "output_hidden_states"):
            self.config.output_hidden_states = True
        else:
            self.config.output_hidden_states = False

        # Load tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(
            model_name,
        )

        # Load model in 4bit or 8bit or 32bit
        if args.quantisation == "4bit":
            self.model = BartForConditionalGeneration.from_pretrained(
                model_name, config=self.config, load_in_4bit=True
            )
        elif args.quantisation == "8bit":
            self.model = BartForConditionalGeneration.from_pretrained(
                model_name, config=self.config, load_in_8bit=True
            )
        else:
            self.model = BartForConditionalGeneration.from_pretrained(
                model_name, config=self.config
            )

        # Save predictions path
        self.save_predictions_path = os.path.join(args.output_path, "predictions")

        self.save_hyperparameters()
