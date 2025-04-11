#
# SPDX-FileCopyrightText: Copyright © 2024 Idiap Research Institute <contact@idiap.ch>
#
# SPDX-FileContributor: Fabio Fehr <fabio.fehr@idiap.ch>
#
# SPDX-License-Identifier: GPL-3.0-only
#

import os

from transformers import AutoConfig, AutoTokenizer

from models_nvib.modeling_nvibmarian import MarianMTModelNVIB
from models_pl.seq2seq_lightning import Seq2SeqLightning


class NvibMarianLightning(Seq2SeqLightning):
    """Lighting module for Marian with Nvib, inherits from a base sequence-to-sequence model."""

    def __init__(self, args, **kwargs):
        super().__init__(args, **kwargs)

        self.lambda_kld = args.kld_lambda
        self.lambda_klg = args.klg_lambda

        # Load pretrained model + Tokenizer
        self.config = AutoConfig.from_pretrained(
            f"Helsinki-NLP/opus-mt-{args.src_lang}-{args.tgt_lang}"
        )
        self.config.output_attentions = True
        self.config.output_path = args.output_path

        # Encoder
        self.config.prior_mus_encoder = (
            args.prior_mus_encoder if hasattr(args, "prior_mus_encoder") else None
        )
        self.config.prior_vars_encoder = (
            args.prior_vars_encoder if hasattr(args, "prior_vars_encoder") else None
        )
        self.config.prior_log_alphas_encoder = (
            args.prior_log_alphas_encoder if hasattr(args, "prior_log_alphas_encoder") else None
        )
        self.config.prior_log_alpha_stdevs_encoder = (
            args.prior_log_alpha_stdevs_encoder
            if hasattr(args, "prior_log_alpha_stdevs_encoder")
            else None
        )

        # Decoder
        self.config.prior_mus_decoder = (
            args.prior_mus_decoder if hasattr(args, "prior_mus_decoder") else None
        )
        self.config.prior_vars_decoder = (
            args.prior_vars_decoder if hasattr(args, "prior_vars_decoder") else None
        )
        self.config.prior_log_alphas_decoder = (
            args.prior_log_alphas_decoder if hasattr(args, "prior_log_alphas_decoder") else None
        )
        self.config.prior_log_alpha_stdevs_decoder = (
            args.prior_log_alpha_stdevs_decoder
            if hasattr(args, "prior_log_alpha_stdevs_decoder")
            else None
        )

        # Cross
        self.config.prior_mus_cross = (
            args.prior_mus_cross if hasattr(args, "prior_mus_cross") else None
        )
        self.config.prior_vars_cross = (
            args.prior_vars_cross if hasattr(args, "prior_vars_cross") else None
        )
        self.config.prior_log_alphas_cross = (
            args.prior_log_alphas_cross if hasattr(args, "prior_log_alphas_cross") else None
        )
        self.config.prior_log_alpha_stdevs_cross = (
            args.prior_log_alpha_stdevs_cross
            if hasattr(args, "prior_log_alpha_stdevs_cross")
            else None
        )

        self.config.delta = 1
        self.config.kappa = 1
        self.config.mu_tau = args.mu_tau

        # Individual encoder, cross and decoder NVIB parameters
        self.config.alpha_tau_e = args.alpha_tau_e
        self.config.alpha_tau_c = args.alpha_tau_c
        self.config.alpha_tau_d = args.alpha_tau_d
        self.config.stdev_tau_e = args.stdev_tau_e
        self.config.stdev_tau_c = args.stdev_tau_c
        self.config.stdev_tau_d = args.stdev_tau_d

        self.tokenizer = AutoTokenizer.from_pretrained(
            f"Helsinki-NLP/opus-mt-{args.src_lang}-{args.tgt_lang}",
            src_lang=args.src_lang,
            tgt_lang=args.tgt_lang,
        )
        self.model = MarianMTModelNVIB.from_pretrained(
            f"Helsinki-NLP/opus-mt-{args.src_lang}-{args.tgt_lang}", config=self.config
        )
        # Reinitialise the NVIB parameters
        self.model.model.nvib_layer.init_parameters()
        # Initialise the NVIB parameters per layer
        for layer in self.model.model.encoder.layers:
            # if layer has attribute sa_nvib_layer, then init_parameters
            if hasattr(layer, "nvib_sa_layer"):
                layer.nvib_sa_layer.init_parameters()

        for layer in self.model.model.decoder.layers:
            # if layer has attribute sa_nvib_layer, then init_parameters
            if hasattr(layer, "nvib_causal_sa_layer"):
                layer.nvib_causal_sa_layer.init_parameters()

        # Logging
        self.log_bleu = True
        self.is_nvib = True
        # Save predictions path
        self.save_predictions_path = os.path.join(args.output_path, "predictions")

        self.baseline_rouge_path = os.path.join(
            args.output_dir,
            args.project_name,
            "baseline_"
            + args.model.replace("NVIB", "")
            + "_"
            + args.data
            + "_"
            + args.src_lang
            + args.tgt_lang,
        )

        self.baseline_bleu_path = os.path.join(
            args.output_dir,
            args.project_name,
            "baseline_"
            + args.model.replace("NVIB", "")
            + "_"
            + args.data
            + "_"
            + args.src_lang
            + args.tgt_lang,
        )

        # PLOT SELF ATTENTION
        # self.plot_encoder_attention = True
        # self.plot_cross_attention = True

        self.save_hyperparameters()
