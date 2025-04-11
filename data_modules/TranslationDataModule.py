#
# SPDX-FileCopyrightText: Copyright © 2024 Idiap Research Institute <contact@idiap.ch>
#
# SPDX-FileContributor: Fabio Fehr <fabio.fehr@idiap.ch>
#
# SPDX-License-Identifier: GPL-3.0-only
#

import os

import lightning.pytorch as pl

# Should support switches between my translation datasets
import torch
from datasets import concatenate_datasets, load_dataset
from torch.utils.data import DataLoader, Dataset
from transformers.data import DataCollatorForSeq2Seq


class DatasetWrapper(Dataset):
    """Dataset wrapper to set the __getitem__ function."""

    def __init__(self, dataset_dict):
        self.dataset_dict = dataset_dict

    def __len__(self):
        return len(self.dataset_dict["labels"])

    def __getitem__(self, idx):
        return {
            "input_ids": self.dataset_dict["input_ids"][idx],
            "labels": self.dataset_dict["labels"][idx],
            "attention_mask": self.dataset_dict["attention_mask"][idx],
        }


def process_dataset(dataset, even_split=False):
    """We split the dataset into train, validation and test.

    Put in a function for using with maps.
    """

    if even_split:
        # Split the dataset into validation and test evenly
        train_split = dataset["train"].train_test_split(train_size=0.33)
        dataset["train"] = train_split["train"]
        val_split = train_split["test"].train_test_split(test_size=0.5)
        dataset["validation"] = val_split["train"]
        dataset["test"] = val_split["test"]
    else:
        # Split the dataset into validation and test train/val/test = rest/10K/10K
        val_split = dataset["train"].train_test_split(test_size=10000)
        dataset["validation"] = val_split["test"]
        test_split = val_split["train"].train_test_split(test_size=10000)
        dataset["test"] = test_split["test"]
        dataset["train"] = test_split["train"]

    return dataset


def prepare_tokenization(
    dataset,
    dataset_name,
    name,
    model_name,
    src_label,
    tgt_label,
    tokenizer,
    max_length=None,
    number_of_workers=1,
):
    """Tokenize the dataset and save to disk."""

    path = f"data/tokenized_{model_name}_{dataset_name}_{src_label}_{tgt_label}_{name}.pkl"
    # Train, validation or test
    dataset = dataset[name]
    column_names = dataset.column_names

    # if path does not exist
    if not os.path.exists(path):
        print("Tokenizing and saving to ", path)

        # Inbedded function for parallelization
        def tokenize_per_example(examples):
            src = [ex[src_label] for ex in examples["translation"]]
            tgt = [ex[tgt_label] for ex in examples["translation"]]

            # Tokenize into lists
            model_inputs = tokenizer(src, truncation=True, max_length=max_length)
            labels = tokenizer(text_target=tgt, truncation=True, max_length=max_length)

            model_inputs["labels"] = labels["input_ids"]
            return model_inputs

        # Run function in parallel
        model_inputs = dataset.map(
            tokenize_per_example,
            batched=True,
            num_proc=number_of_workers,
            remove_columns=column_names,
            load_from_cache_file=False,  # This can be helpful but sometimes it gets stuck!
        )

        # save
        torch.save(model_inputs, path)


def load_prepared_data(name, data_name, model_name, src_label, tgt_label):
    """Load the prepared data from disk."""

    path = f"data/tokenized_{model_name}_{data_name}_{src_label}_{tgt_label}_{name}.pkl"
    # Load
    model_inputs = torch.load(path)

    return model_inputs


class TranslationDataModule(pl.LightningDataModule):
    """Data module for translation tasks."""

    def __init__(
        self,
        pl_model,
        batch_size,
        src_lang,
        tgt_lang,
        data,
        num_workers,
        fp16,
        prefix="",
        train_perc=1,
        val_perc=1,
        **kwargs,
    ):
        super().__init__()
        self.tokenizer = pl_model.tokenizer

        # Handles the padding and right shifting of decoder inputs
        self.collator = DataCollatorForSeq2Seq(
            tokenizer=pl_model.tokenizer,
            model=pl_model,
            label_pad_token_id=self.tokenizer.pad_token_id,
            pad_to_multiple_of=8 if fp16 else None,
        )

        self.batch_size = batch_size
        self.src_lang = src_lang
        self.tgt_lang = tgt_lang
        self.data = data
        self.num_workers = num_workers
        self.max_length = pl_model.config.max_length
        self.prefix = prefix
        self.model_name = os.path.basename(pl_model.config._name_or_path)
        self.train_perc = train_perc
        self.val_perc = val_perc

    def prepare_data(self):
        """Download data or load data HuggingFace."""
        # Download data or load data HuggingFace
        if self.data == "iwslt2017":
            self.dataset = load_dataset(
                self.data, self.data + "-" + self.src_lang + "-" + self.tgt_lang
            )
        elif self.data == "ted_talks_iwslt":
            self.dataset = load_dataset(
                self.data, language_pair=(self.src_lang, self.tgt_lang), year="2014"
            )
        elif self.data == "bible_para":
            # This one we need to go en-zh
            if self.tgt_lang == "de":
                self.dataset = load_dataset(self.data, lang1=self.tgt_lang, lang2=self.src_lang)
            else:
                self.dataset = load_dataset(self.data, lang1=self.src_lang, lang2=self.tgt_lang)
        elif self.data == "opus100":
            if self.tgt_lang == "de":
                self.dataset = load_dataset(self.data, self.tgt_lang + "-" + self.src_lang)
            else:
                self.dataset = load_dataset(self.data, self.src_lang + "-" + self.tgt_lang)
        else:
            raise NotImplementedError(f"{self.data} not implemented")

        # Split the dataset into validation and test
        if not self.data == "opus100":
            if self.data == "ted_talks_iwslt":
                self.dataset = process_dataset(self.dataset, even_split=True)
            # Allow for more data in iwslt2017 validation
            elif self.data == "iwslt2017":
                val_split = self.dataset["train"].train_test_split(test_size=9100)
                # concat the validations
                self.dataset["validation"] = concatenate_datasets(
                    [self.dataset["validation"], val_split["test"]]
                )
                self.dataset["train"] = val_split["train"]
            else:
                # Split the dataset into validation and test train/val/test = rest/10K/10K
                self.dataset = process_dataset(self.dataset, even_split=False)

        # Tokenize here and save to disk
        prepare_tokenization(
            dataset=self.dataset,
            dataset_name=self.data,
            name="train",
            model_name=self.model_name,
            src_label=self.src_lang,
            tgt_label=self.tgt_lang,
            tokenizer=self.tokenizer,
            max_length=self.max_length,
            number_of_workers=self.num_workers,
        )
        prepare_tokenization(
            dataset=self.dataset,
            dataset_name=self.data,
            name="validation",
            model_name=self.model_name,
            src_label=self.src_lang,
            tgt_label=self.tgt_lang,
            tokenizer=self.tokenizer,
            max_length=self.max_length,
            number_of_workers=self.num_workers,
        )
        prepare_tokenization(
            dataset=self.dataset,
            dataset_name=self.data,
            name="test",
            model_name=self.model_name,
            src_label=self.src_lang,
            tgt_label=self.tgt_lang,
            tokenizer=self.tokenizer,
            max_length=self.max_length,
            number_of_workers=self.num_workers,
        )

    def setup(self, stage=None):
        """Setup data for train, val and test.

        Only run just before the trainer uses it. More efficient
        """

        # Assign train/val datasets for use in dataloaders
        if stage == "fit" or stage is None:
            self.train_inputs = load_prepared_data(
                name="train",
                data_name=self.data,
                model_name=self.model_name,
                src_label=self.src_lang,
                tgt_label=self.tgt_lang,
            )

            # Subset train data
            if self.train_perc < 1:
                self.train_inputs = self.train_inputs.train_test_split(
                    train_size=self.train_perc, seed=42
                )["train"]

            self.validation_inputs = load_prepared_data(
                name="validation",
                # name="train",  # For overfitting
                data_name=self.data,
                model_name=self.model_name,
                src_label=self.src_lang,
                tgt_label=self.tgt_lang,
            )

        # Assign validation dataset for use in dataloader(s)
        if stage == "validate":

            self.validation_inputs = load_prepared_data(
                name="validation",
                data_name=self.data,
                model_name=self.model_name,
                src_label=self.src_lang,
                tgt_label=self.tgt_lang,
            )
            # Subset for validation
            if self.val_perc < 1:
                self.validation_inputs = self.validation_inputs.train_test_split(
                    train_size=self.val_perc, seed=42
                )["train"]

        # Assign test dataset for use in dataloader(s)
        if stage == "test":
            self.test_inputs = load_prepared_data(
                name="test",
                data_name=self.data,
                model_name=self.model_name,
                src_label=self.src_lang,
                tgt_label=self.tgt_lang,
            )

    def train_dataloader(self):
        """Training dataloader."""
        return DataLoader(
            self.train_inputs,
            shuffle=True,
            pin_memory=True,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            collate_fn=self.collator,
        )

    def val_dataloader(self):
        """Validation dataloader."""
        return DataLoader(
            self.validation_inputs,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            collate_fn=self.collator,
            shuffle=False,
        )

    def test_dataloader(self):
        """Test dataloader."""
        return DataLoader(
            self.test_inputs,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            collate_fn=self.collator,
        )
