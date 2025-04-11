#
# SPDX-FileCopyrightText: Copyright © 2024 Idiap Research Institute <contact@idiap.ch>
#
# SPDX-FileContributor: Fabio Fehr <fabio.fehr@idiap.ch>
#
# SPDX-License-Identifier: GPL-3.0-only
#

import os
from itertools import chain

import lightning.pytorch as pl
import torch
import torch.nn as nn
from torchmetrics.text.bleu import BLEUScore

# import evaluate
from torchmetrics.text.rouge import ROUGEScore
from transformers import get_cosine_schedule_with_warmup

from utils import show_attention


class Seq2SeqLightning(pl.LightningModule):
    """Base lightning module for sequence to sequence models."""

    def __init__(self, args, **kwargs):
        super().__init__()

        # Training parameters
        self.learning_rate = args.learning_rate if hasattr(args, "learning_rate") else None
        self.learning_rate_scheduler = (
            args.lr_scheduler if hasattr(args, "lr_scheduler") else False
        )
        self.perc_warmup = args.perc_warmup if hasattr(args, "perc_warmup") else None

        self.max_length = args.max_length if hasattr(args, "max_length") else None

        # For NVIB models
        self.is_nvib = False

        # Logging metrics
        self.log_bleu = False
        self.log_rouge = False

        # Save predictions
        self.save_predictions_path = None
        self.baseline_rouge_path = None
        self.baseline_bleu_path = None

        # Plotting attention
        self.plot_cross_attention = False
        self.plot_encoder_attention = False
        self.plot_decoder_attention = False

    def training_step(self, batch, batch_idx):
        """Training step in the lightning trainer for sequence to sequence models."""

        # print(f"training step batch_idx {batch_idx}, items: {batch['input_ids']}")

        # Forward pass
        model_outputs = self.model(**batch)

        # Get loss and ignore the pads
        criterion = nn.CrossEntropyLoss(ignore_index=self.tokenizer.pad_token_id, reduction="mean")
        # Transform targets
        targets = torch.flatten(batch["labels"])  # [Nt x B]
        # Transform vocabulary
        logits = torch.flatten(model_outputs["logits"], start_dim=0, end_dim=1)  # [Nt x B, V]
        # Calculates loss
        cross_entropy_loss = criterion(logits, targets)  # [Nt x B]

        if self.is_nvib:
            kld = torch.mean(model_outputs.kl_dirichlet) * self.lambda_kld
            klg = torch.mean(model_outputs.kl_gaussian) * self.lambda_klg

            loss = cross_entropy_loss + kld + klg

            # Log things
            self.log("train_kld", kld)
            self.log("train_klg", klg)
        else:
            loss = cross_entropy_loss

        self.log("train_loss", loss)
        self.log("train_cross_entropy", cross_entropy_loss)

        return loss

    def on_validation_epoch_start(self) -> None:
        """Before the validation epoch starts, reset the lists."""
        self.val_inputs = []
        self.val_preds = []
        self.val_tgt = []
        self.val_ce = []

    def validation_step(self, batch, batch_idx):
        """Validation step in the lightning trainer for sequence to sequence models."""

        # Forward pass
        model_outputs = self.model(**batch)

        # Get loss and ignore the pads
        criterion = nn.CrossEntropyLoss(ignore_index=self.tokenizer.pad_token_id, reduction="mean")
        # Transform targets
        targets = torch.flatten(batch["labels"])  # [Nt x B]
        # Transform vocabulary
        logits = torch.flatten(model_outputs["logits"], start_dim=0, end_dim=1)  # [Nt x B, V]
        # Calculates loss
        cross_entropy_loss = criterion(logits, targets)  # [Nt x B]
        # val_cross_entropy_loss = model_outputs.loss  # [Nt x B]

        if self.is_nvib:
            kld = torch.mean(model_outputs.kl_dirichlet) * self.lambda_kld
            klg = torch.mean(model_outputs.kl_gaussian) * self.lambda_klg

            loss = cross_entropy_loss + kld + klg

            # Log things
            self.log("val_kld", kld)
            self.log("val_klg", klg)
        else:
            loss = cross_entropy_loss

        self.log("val_loss", loss)
        self.log("val_cross_entropy", cross_entropy_loss)

        # Autoregressive prediction
        generated_ids = self.model.generate(
            batch["input_ids"],
            max_new_tokens=self.max_length,
        )
        batch_predictions = self.tokenizer.batch_decode(generated_ids, skip_special_tokens=True)

        tgt = self.tokenizer.batch_decode(
            batch["labels"],
            skip_special_tokens=True,
        )

        # Here we save locally and not keep in memory
        self.val_preds.append(batch_predictions)
        self.val_tgt.append(tgt)
        self.val_ce.append(cross_entropy_loss)

        if self.plot_cross_attention:
            if batch_idx == 0:
                # Only the first batch and the first item of the batch
                for i in range(1):
                    batch_item = i
                    # Plot attention
                    show_attention(
                        batch,
                        batch_item,
                        self.tokenizer,
                        model_outputs,
                        "cross_attentions",
                        self.logger,
                        "Validation",
                        # zmax=1,
                        prior="<PRIOR>" if self.is_nvib else None,
                        # num_heads=self.config.decoder_attention_heads,
                        num_heads=1,
                        num_layers=self.config.decoder_layers,
                        batch_idx=batch_idx,
                        observation=batch_item,
                    )
        if self.plot_encoder_attention:
            if batch_idx == 0:
                # Only the first batch and the first item of the batch
                for i in range(1):
                    batch_item = i
                    # Plot attention
                    show_attention(
                        batch,
                        batch_item,
                        self.tokenizer,
                        model_outputs,
                        "encoder_attentions",
                        self.logger,
                        "Validation",
                        # zmax=1,
                        prior="<PRIOR>" if self.is_nvib else None,
                        # num_heads=self.config.encoder_attention_heads,
                        num_heads=1,  # Pool over heads
                        num_layers=self.config.encoder_layers,
                        batch_idx=batch_idx,
                        observation=batch_item,
                    )
        if self.plot_decoder_attention:
            if batch_idx == 0:
                # Only the first batch and the first item of the batch
                for i in range(1):
                    batch_item = i
                    # Plot attention
                    show_attention(
                        batch,
                        batch_item,
                        self.tokenizer,
                        model_outputs,
                        "decoder_attentions",
                        self.logger,
                        "Validation",
                        # zmax=1,
                        prior="<PRIOR>" if self.is_nvib else None,
                        # num_heads=self.config.encoder_attention_heads,
                        num_heads=1,  # Pool over heads
                        num_layers=self.config.decoder_layers,
                        batch_idx=batch_idx,
                        observation=batch_item,
                    )

    def on_validation_epoch_end(self):
        """After the validation epoch ends, calculate the metrics and log them."""
        # First load the data only if youre gpu 1
        # Gather from all GPUs
        preds = self.all_gather(self.val_preds)
        targets = self.all_gather(self.val_tgt)
        cross_entropy_loss = self.all_gather(self.val_ce)

        if self.save_predictions_path is not None:
            # Save predictions
            torch.save(preds, self.save_predictions_path + "_val.pt")
            # Save txt
            with open(self.save_predictions_path + "_val.txt", "w") as f:
                for pred in preds:
                    for item in pred:
                        f.write("%s\n" % item)

        # Calculate score
        preds = list(chain.from_iterable(self.val_preds))
        targets = list(chain.from_iterable(self.val_tgt))
        cross_entropy_loss = torch.mean(torch.stack(self.val_ce))
        # Average of all cross entropy losses from teacher forcing
        self.log("ce_val", cross_entropy_loss, rank_zero_only=True)

        if self.log_rouge:
            score = ROUGEScore()
            results = score(preds, targets)
            self.log("rouge1_val", results["rouge1_fmeasure"] * 100)
            self.log("rouge2_val", results["rouge2_fmeasure"] * 100)
            self.log("rougeL_val", results["rougeL_fmeasure"] * 100)

        if self.baseline_rouge_path is not None and os.path.exists(
            self.baseline_rouge_path + "/predictions_val.pt"
        ):
            # load baseline_targets
            baseline_targets = torch.load(self.baseline_rouge_path + "/predictions_val.pt")
            baseline_targets = list(chain.from_iterable(baseline_targets))
            score = ROUGEScore()
            results = score(preds, baseline_targets)
            self.log("baseline_rouge1_val", results["rouge1_fmeasure"] * 100)
            self.log("baseline_rouge2_val", results["rouge2_fmeasure"] * 100)
            self.log("baseline_rougeL_val", results["rougeL_fmeasure"] * 100)
        if self.log_bleu:
            score = BLEUScore()
            results = score(preds, [[ref] for ref in targets])
            self.log("bleu_val", results.item() * 100, rank_zero_only=True)
        if self.baseline_bleu_path is not None and os.path.exists(
            self.baseline_bleu_path + "/predictions_val.pt"
        ):
            # load baseline_targets
            baseline_targets = torch.load(self.baseline_bleu_path + "/predictions_val.pt")
            baseline_targets = list(chain.from_iterable(baseline_targets))
            score = BLEUScore()
            results = score(preds, [[ref] for ref in baseline_targets])
            self.log("baseline_bleu_val", results.item() * 100, rank_zero_only=True)

    def on_test_epoch_start(self) -> None:
        """Before the test epoch starts, reset the lists."""

        self.test_inputs = []
        self.test_preds = []
        self.test_tgt = []
        self.test_ce = []

    def test_step(self, batch, batch_idx):
        """Test step in the lightning trainer for sequence to sequence models."""

        # Forward pass
        model_outputs = self.model(**batch)

        # Get loss and ignore the pads
        criterion = nn.CrossEntropyLoss(ignore_index=self.tokenizer.pad_token_id, reduction="mean")
        # Transform targets
        targets = torch.flatten(batch["labels"])  # [Nt x B]
        # Transform vocabulary
        logits = torch.flatten(model_outputs["logits"], start_dim=0, end_dim=1)  # [Nt x B, V]
        # Calculates loss
        cross_entropy_loss = criterion(logits, targets)  # [Nt x B]
        # val_cross_entropy_loss = model_outputs.loss  # [Nt x B]

        if self.is_nvib:
            kld = torch.mean(model_outputs.kl_dirichlet) * self.lambda_kld
            klg = torch.mean(model_outputs.kl_gaussian) * self.lambda_klg

            loss = cross_entropy_loss + kld + klg

            # Log things
            self.log("test_kld", kld)
            self.log("test_klg", klg)
        else:
            loss = cross_entropy_loss

        self.log("test_loss", loss)
        self.log("test_cross_entropy", cross_entropy_loss)

        # Autoregressive prediction
        generated_ids = self.model.generate(
            batch["input_ids"],
            max_new_tokens=self.max_length,
        )
        batch_predictions = self.tokenizer.batch_decode(generated_ids, skip_special_tokens=True)

        tgt = self.tokenizer.batch_decode(
            batch["labels"],
            skip_special_tokens=True,
        )

        # Here we save locally and not keep in memory
        self.test_preds.append(batch_predictions)
        self.test_tgt.append(tgt)
        self.test_ce.append(cross_entropy_loss)

    def on_test_epoch_end(self):
        """After the test epoch ends, calculate the metrics and log them."""

        # First load the data only if youre gpu 1
        # Gather from all GPUs
        preds = self.all_gather(self.test_preds)
        targets = self.all_gather(self.test_tgt)
        cross_entropy_loss = self.all_gather(self.test_ce)

        if self.save_predictions_path is not None:
            # Save predictions
            torch.save(preds, self.save_predictions_path + "_test.pt")
            # Save txt
            with open(self.save_predictions_path + "_test.txt", "w") as f:
                for pred in preds:
                    for item in pred:
                        f.write("%s\n" % item)

        # Calculate score
        preds = list(chain.from_iterable(self.test_preds))
        targets = list(chain.from_iterable(self.test_tgt))
        cross_entropy_loss = torch.mean(torch.stack(self.test_ce))
        # Average of all cross entropy losses from teacher forcing
        self.log("ce_test", cross_entropy_loss, rank_zero_only=True)

        if self.baseline_rouge_path is not None and os.path.exists(
            self.baseline_rouge_path + "/predictions_test.pt"
        ):
            # load baseline_targets
            baseline_targets = torch.load(self.baseline_rouge_path + "/predictions_test.pt")
            baseline_targets = list(chain.from_iterable(baseline_targets))
            score = ROUGEScore()
            results = score(preds, baseline_targets)
            self.log("baseline_rouge1_test", results["rouge1_fmeasure"] * 100)
            self.log("baseline_rouge2_test", results["rouge2_fmeasure"] * 100)
            self.log("baseline_rougeL_test", results["rougeL_fmeasure"] * 100)
        if self.log_rouge:
            score = ROUGEScore()
            results = score(preds, targets)
            self.log("rouge1_test", results["rouge1_fmeasure"] * 100)
            self.log("rouge2_test", results["rouge2_fmeasure"] * 100)
            self.log("rougeL_test", results["rougeL_fmeasure"] * 100)
        if self.log_bleu:
            score = BLEUScore()
            results = score(preds, [[ref] for ref in targets])
            self.log("bleu_test", results.item() * 100, rank_zero_only=True)
        if self.baseline_bleu_path is not None and os.path.exists(
            self.baseline_bleu_path + "/predictions_test.pt"
        ):
            # load baseline_targets
            baseline_targets = torch.load(self.baseline_bleu_path + "/predictions_test.pt")
            baseline_targets = list(chain.from_iterable(baseline_targets))
            score = BLEUScore()
            results = score(preds, [[ref] for ref in baseline_targets])
            self.log("baseline_bleu_test", results.item() * 100, rank_zero_only=True)


def configure_optimizers(self):
    """Configure the optimizers and learning rate schedulers."""

    optimizer = torch.optim.RAdam(self.parameters(), lr=self.learning_rate)

    if self.learning_rate_scheduler:
        lr_scheduler = {
            "scheduler": get_cosine_schedule_with_warmup(
                optimizer,
                num_training_steps=self.trainer.max_steps,
                num_warmup_steps=self.perc_warmup * self.trainer.max_steps,
            ),
            "name": "learning_rate_scheduler",
            "interval": "step",
        }

        return [optimizer], [lr_scheduler]
    else:
        return [optimizer]
