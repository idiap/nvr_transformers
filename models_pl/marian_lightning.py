#
# SPDX-FileCopyrightText: Copyright © 2024 Idiap Research Institute <contact@idiap.ch>
#
# SPDX-FileContributor: Fabio Fehr <fabio.fehr@idiap.ch>
#
# SPDX-License-Identifier: GPL-3.0-only
#

import os

from transformers import AutoConfig, AutoTokenizer

from models_hf.modeling_marian import MarianMTModel
from models_pl.seq2seq_lightning import Seq2SeqLightning


class MarianLightning(Seq2SeqLightning):
    """Lighting module for Marian, inherits from a base sequence-to-sequence model."""

    def __init__(self, args, **kwargs):
        super().__init__(args, **kwargs)

        # Load config
        self.config = AutoConfig.from_pretrained(
            f"Helsinki-NLP/opus-mt-{args.src_lang}-{args.tgt_lang}"
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
            f"Helsinki-NLP/opus-mt-{args.src_lang}-{args.tgt_lang}",
            src_lang=args.src_lang,
            tgt_lang=args.tgt_lang,
        )

        # Load model in 4bit or 8bit or 32bit
        if args.quantisation == "4bit":
            self.model = self.model = MarianMTModel.from_pretrained(
                f"Helsinki-NLP/opus-mt-{args.src_lang}-{args.tgt_lang}",
                config=self.config,
                load_in_4bit=True,
            )
        elif args.quantisation == "8bit":
            self.model = MarianMTModel.from_pretrained(
                f"Helsinki-NLP/opus-mt-{args.src_lang}-{args.tgt_lang}",
                config=self.config,
                load_in_8bit=True,
            )
        else:
            self.model = MarianMTModel.from_pretrained(
                f"Helsinki-NLP/opus-mt-{args.src_lang}-{args.tgt_lang}",
                config=self.config,
            )

        # Logging metrics
        self.log_bleu = True
        self.log_chrf = False

        # Save predictions path
        self.save_predictions_path = os.path.join(args.output_path, "predictions")

        self.save_hyperparameters()
