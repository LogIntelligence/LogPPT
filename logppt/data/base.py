from abc import ABC, abstractmethod
from datasets import load_dataset
import os
import pandas as pd
from logppt.data.utils import log_to_dataframe, generate_logformat_regex

DOWNLOAD_URL = "https://zenodo.org/records/8275861/files/{}.zip"

def load_loghub_dataset(dataset_name="Apache", cache_dir=None, format="csv", log_format=None):
    """
    Load from cache if available, otherwise download, unzip and cache the dataset
    """
    dataset_url = DOWNLOAD_URL.format(dataset_name)
    # Check if the dataset is already downloaded
    if cache_dir is None:
        cache_dir = os.path.join(os.path.expanduser("~"), ".cache", "logppt")
    dataset_dir = os.path.join(cache_dir, dataset_name)
    if not os.path.exists(dataset_dir):
        os.makedirs(dataset_dir)
        # Download the dataset
        dataset_zip = os.path.join(cache_dir, f"{dataset_name}.zip")
        os.system(f"wget {dataset_url} -O {dataset_zip}")
        # Unzip the dataset
        os.system(f"unzip {dataset_zip} -d {os.path.dirname(dataset_dir)}")
        # Remove the zip file
        os.remove(dataset_zip)
    # Load the dataset
    if format == "csv":
        log_df = pd.read_csv(f"{dataset_dir}/{dataset_name}_full.log_structured.csv")
    elif format == "text":
        headers, regex = generate_logformat_regex(log_format)
        log_df = log_to_dataframe(f"{dataset_dir}/{dataset_name}_full.log", regex, headers)
    return log_df


class BaseDataLoader(ABC):
    def __init__(self, config) -> None:
        """
        Base class for data loaders
        Parameters:
            config: model configuration
            parameter_types: list of parameter types. if None, traditional log parsing. Otherwise, variable-aware log parsing

        """
        self.config = config
        self.vtoken = "<*>"

    def size(self):
        return len(self.raw_datasets['train'])

    def get_train_dataloader(self):
        """
        Returns the training dataloader
        """
        return self.train_loader

    def get_val_dataloader(self):
        """
        Returns the validation dataloader
        """
        return self.val_loader

    def get_test_dataloader(self):
        """
        Returns the test dataloader
        """
        return self.val_loader

    def load_data(self):
        """
        Load data from file
        """
        # Get the datasets: the data file are JSON files
        data_files = {}
        if self.config.train_file is not None:
            data_files["train"] = [self.config.train_file]
        if self.config.validation_file is not None:
            data_files["validation"] = self.config.validation_file
        if self.config.dev_file is not None:
            data_files["dev"] = self.config.dev_file

        self.raw_datasets = load_dataset("json", data_files=data_files)

        if self.raw_datasets["train"] is not None:
            column_names = self.raw_datasets["train"].column_names
        else:
            column_names = self.raw_datasets["validation"].column_names

        if self.config.text_column_name is not None:
            text_column_name = self.config.text_column_name
        else:
            text_column_name = column_names[0]

        if self.config.label_column_name is not None:
            label_column_name = self.config.label_column_name
        else:
            label_column_name = column_names[1]
        self.text_column_name = text_column_name
        self.label_column_name = label_column_name

    @abstractmethod
    def initialize(self, tokenizer):
        pass

    @abstractmethod
    def tokenize(self):
        pass

    @abstractmethod
    def build_dataloaders(self):
        pass

