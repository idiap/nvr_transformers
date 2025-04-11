#
# SPDX-FileCopyrightText: Copyright © 2024 Idiap Research Institute <contact@idiap.ch>
#
# SPDX-FileContributor: Fabio Fehr <fabio.fehr@idiap.ch>
#
# SPDX-License-Identifier: GPL-3.0-only
#

import os

import lightning.pytorch as pl
import torch
from datasets import load_dataset
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


def process_curation_dataset(dataset):
    """Process the curation dataset to remove empty summaries and articles that are shorter than
    the summary.

    We split the dataset into train, validation and test. Put in a function for using with maps.
    """

    # Mapping function
    def filter_curation(examples):
        if examples["summary"] and examples["article_content"]:
            return len(examples["summary"]) > 0 and len(examples["article_content"]) > len(
                examples["summary"]
            )

    # Filter dataset
    dataset = dataset.filter(lambda example: filter_curation(example))
    # Split into train, validation and test
    val_split = dataset["train"].train_test_split(test_size=0.5)
    test_split = val_split["test"].train_test_split(test_size=0.5)
    dataset["train"] = val_split["train"]
    dataset["validation"] = test_split["train"]
    dataset["test"] = test_split["test"]

    return dataset


def prepare_tokenization(
    dataset,
    dataset_name,
    name,
    model_name,
    tokenizer,
    # max_length=None,
    prefix="",
    number_of_workers=1,
):
    """Tokenize the dataset and save to disk.

    :param dataset: downloaded dataset from hugging face
    :param tokenizer: the tokenizer
    :param padding: Boolean to pad or not to pad
    :return: tokenized and padded model input
    """
    path = f"data/tokenized_{model_name}_{dataset_name}_{name}.pkl"
    # Train, validation or test
    dataset = dataset[name]
    column_names = dataset.column_names
    if dataset_name == "cnn_dailymail":
        doc_name = "article"
        summ_name = "highlights"
    elif dataset_name == "xsum":
        doc_name = "document"
        summ_name = "summary"
    elif dataset_name == "curation":
        doc_name = "article_content"
        summ_name = "summary"
    elif dataset_name == "wikihow":
        doc_name = "text"
        summ_name = "headline"
    elif dataset_name == "samsum":
        doc_name = "dialogue"
        summ_name = "summary"
    else:
        raise ValueError("Dataset not supported")

    # if path does not exist
    if not os.path.exists(path):
        print("Tokenizing and saving to ", path)

        # Inbedded function for parallelization
        def tokenize_per_example(examples):
            doc = examples[doc_name]
            summ = examples[summ_name]
            # Add prefix for models
            doc = [prefix + doc_ex for doc_ex in doc]

            # Tokenize into lists
            model_inputs = tokenizer(
                doc,
                truncation=True,
                max_length=1024,  # BARTs context is 1024
            )
            labels = tokenizer(
                text_target=summ,
                truncation=True,
                max_length=1024,  # BARTs context is 1024
            )

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


def load_prepared_data(name, data_name, model_name):
    """Load the prepared data from disk."""
    path = f"data/tokenized_{model_name}_{data_name}_{name}.pkl"
    model_inputs = torch.load(path)
    return model_inputs


class SummarisationDataModule(pl.LightningDataModule):
    """Data module for summarisation tasks."""

    def __init__(
        self,
        pl_model,
        batch_size,
        data,
        num_workers,
        fp16,
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
        self.data = data
        self.num_workers = num_workers
        self.model_name = os.path.basename(pl_model.config._name_or_path)
        self.train_perc = train_perc
        self.val_perc = val_perc

    def prepare_data(self):
        """Prepare data - download - tokenize - save to disk."""
        # Download data or load data
        if self.data == "wikihow":
            self.dataset = load_dataset(self.data, "sep", "data")
        elif self.data == "cnn_dailymail":
            self.dataset = load_dataset(self.data, "3.0.0")
        elif self.data == "curation":
            self.dataset = load_dataset("d0rj/curation-corpus")
            self.dataset = process_curation_dataset(self.dataset)

        else:
            self.dataset = load_dataset(self.data)

        # Tokenize here and save to disk
        prepare_tokenization(
            dataset=self.dataset,
            dataset_name=self.data,
            name="train",
            model_name=self.model_name,
            tokenizer=self.tokenizer,
            # max_length=self.max_length,
            number_of_workers=self.num_workers,
            prefix="",
        )
        prepare_tokenization(
            dataset=self.dataset,
            dataset_name=self.data,
            name="validation",
            model_name=self.model_name,
            tokenizer=self.tokenizer,
            # max_length=self.max_length,
            number_of_workers=self.num_workers,
            prefix="",
        )
        prepare_tokenization(
            dataset=self.dataset,
            dataset_name=self.data,
            name="test",
            model_name=self.model_name,
            tokenizer=self.tokenizer,
            # max_length=self.max_length,
            number_of_workers=self.num_workers,
            prefix="",
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
            )

            # Subset train data
            if self.train_perc < 1:
                self.train_inputs = self.train_inputs.train_test_split(
                    train_size=self.train_perc, seed=42
                )["train"]

            self.validation_inputs = load_prepared_data(
                name="validation",
                data_name=self.data,
                model_name=self.model_name,
            )

        # Assign validation dataset for use in dataloader(s)
        if stage == "validate":
            self.validation_inputs = load_prepared_data(
                name="validation",
                data_name=self.data,
                model_name=self.model_name,
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
            )

    def train_dataloader(self):
        """Train dataloader."""
        return DataLoader(
            self.train_inputs,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            collate_fn=self.collator,
            shuffle=True,
            pin_memory=True,
        )

    def val_dataloader(self):
        """Validation dataloader."""
        return DataLoader(
            self.validation_inputs,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            collate_fn=self.collator,
            shuffle=False,
            pin_memory=True,
        )

    def test_dataloader(self):
        """Test dataloader."""
        return DataLoader(
            self.test_inputs,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            collate_fn=self.collator,
            shuffle=False,
            pin_memory=True,
        )
