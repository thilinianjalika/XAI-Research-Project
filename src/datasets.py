import os, shutil
import yaml
import joblib
import pandas as pd
import numpy as np
import wget
from typing import List, Dict
from sklearn.feature_extraction.text import TfidfVectorizer
from torch.utils.data import Dataset
from transformers import T5Tokenizer
from torch import Tensor
from typing import Tuple

try:
    from .processors import Preprocessor, LUTLabelEncoder
except ImportError:
    from processors import Preprocessor, LUTLabelEncoder


class BaseDataset(Dataset):
    def __init__(
        self, config_path: str, root: str = None, download: bool = False
    ) -> None:
        with open(config_path) as handler:
            config = yaml.load(handler, Loader=yaml.FullLoader)
        self.name = config["name"]
        if root is None:
            self.root = os.path.abspath(os.path.split(config_path)[0])
        else:
            self.root = os.path.abspath(root)
        self.source_url = config["source_url"]
        self.config = config

        self._parse_paths()
        if not self._validate_dataset():
            if download:
                self._download_n_extract()
            else:
                raise FileNotFoundError(
                    f"Dataset files not found in {self.root}. Use 'download=True' to download from source"
                )

    def _parse_paths(self) -> None:
        for k, v in self.config["paths"].items():
            self.config["paths"][k] = os.path.join(self.root, v)

    def _validate_dataset(self) -> bool:
        paths = self.config["paths"].values()
        paths_exist = [os.path.exists(path) for path in paths]
        valid = all(paths_exist)
        return valid

    def _download_n_extract(self) -> List[str]:
        os.makedirs(self.root, exist_ok=True)
        download_path = os.path.join(self.root, os.path.split(self.source_url)[1])
        print(f"Downloading from source ({self.source_url}) to {download_path}")
        download_path = wget.download(self.source_url, download_path)
        shutil.unpack_archive(download_path, self.root)


class IMDBDataset(BaseDataset):
    valid_splits = ["train", "val", "test"]

    def _encode_n_split(self):
        # encode
        x = self.processed_data["review"]
        y = self.processed_data["sentiment"]
        print("Encoding")
        feature_names = self.input_encoder.get_feature_names_out()
        x = self.input_encoder.transform(x)
        y = self.label_encoder.transform(y)

        # split
        ds_size = len(y)
        start_idx = 0
        split_ids = {"train": start_idx}
        end_idx = int(ds_size * self.config["split"]["train"])
        x_train = x[start_idx:end_idx]
        y_train = y[start_idx:end_idx]
        start_idx = int(ds_size * self.config["split"]["train"])
        split_ids["val"] = start_idx
        end_idx = start_idx + int(ds_size * self.config["split"]["val"])
        x_val = x[start_idx:end_idx]
        y_val = y[start_idx:end_idx]
        start_idx = int(
            ds_size * (self.config["split"]["train"] + self.config["split"]["val"])
        )
        split_ids["test"] = start_idx
        end_idx = -1
        x_test = x[start_idx:end_idx]
        y_test = y[start_idx:end_idx]

        self.x_train = x_train
        self.y_train = y_train
        self.x_val = x_val
        self.y_val = y_val
        self.x_test = x_test
        self.y_test = y_test
        self.feature_names = feature_names
        self.split_ids = split_ids

    def _fit_input_encoder(self):
        print("Fitting encoder")
        x = self.processed_data.review
        self.input_encoder.fit(x)

    def get_reviews(self, split: str, ids: np.ndarray) -> List[Dict[str, str]]:
        ids = ids + self.split_ids[split]
        originals = []
        for id in ids:
            originals.append(
                {
                    "original": self.original_data.review[id],
                    "preprocessed": self.processed_data.review[id],
                }
            )
        return originals

    def __init__(
        self,
        config_path: str,
        root: str = None,
        download: bool = False,
        vectorizer_fitted: bool = True,
    ):
        print("Creating dataset")
        super().__init__(config_path, root, download)
        self.config["extras"]["input_encoder_path"] = os.path.join(
            self.root, self.config["extras"]["input_encoder_path"]
        )

        # initialize
        print("Initializing objects")
        self.preprocessor = Preprocessor()
        self.label_encoder = LUTLabelEncoder(self.config["labels"])
        if vectorizer_fitted:
            self.input_encoder = joblib.load(
                self.config["extras"]["input_encoder_path"]
            )
        else:
            self.input_encoder = TfidfVectorizer(min_df=self.config["extras"]["min_df"])

        csv_path = self.config["paths"]["data"]
        self.original_data = pd.read_csv(csv_path)
        preproc_csv_path = csv_path.replace(".csv", "-preproc.csv")

        # check for preprocessed data.
        if os.path.exists(preproc_csv_path):
            processed_data = pd.read_csv(preproc_csv_path)
        else:
            processed_data = pd.read_csv(csv_path)
            print("Preprocessing")
            processed_data.review = processed_data.review.apply(self.preprocessor)
            processed_data.to_csv(preproc_csv_path, index=False)

        self.processed_data = processed_data
        if not vectorizer_fitted:
            self._fit_input_encoder()

        self._encode_n_split()
        print("Dataset created")

    def set_split(self, split: str) -> None:
        if split not in self.valid_splits:
            raise ValueError("Unsupported split definition")

        if split == "train":
            self.active_x = self.x_train.toarray()
            self.active_y = self.y_train
        elif split == "val":
            self.active_x = self.x_val.toarray()
            self.active_y = self.y_val
        else:
            self.active_x = self.x_test.toarray()
            self.active_y = self.y_test

    def __len__(self):
        try:
            y = self.active_y
            return len(y)
        except:
            raise RuntimeError("Please set the split to use this method")

    def __getitem__(self, idx: int) -> Tuple:
        try:
            x = self.active_x[idx]
            y = self.active_y[idx]
            return x, y
        except:
            raise RuntimeError("Please set the split to use this method")


class CFGenerativeDataset(BaseDataset):
    supported_splits = ["train", "val", "test"]

    def __init__(
        self,
        config_path: str,
        root: str = None,
        download: bool = False,
        split: str = "train",
    ) -> None:
        super().__init__(config_path, root, download)
        if split not in self.supported_splits:
            raise ValueError("Unsupported split")

        tokenizer = T5Tokenizer.from_pretrained(self.config["model_name"])
        text_data = pd.read_csv(self.config["paths"][split])
        sent1_inp_ids = tokenizer(
            list("contradict: " + text_data.sentence1),
            return_tensors="pt",
            padding="max_length",
            truncation=True,
            max_length=self.config["max_token_len"],
        ).input_ids

        sent2_inp_ids = tokenizer(
            list(text_data.sentence2),
            return_tensors="pt",
            padding="max_length",
            truncation=True,
            max_length=self.config["max_token_len"],
        ).input_ids
        data = list(zip(sent1_inp_ids, sent2_inp_ids))

        self.tokenizer = tokenizer
        self.text_data = text_data
        self.data = data
        self.len = len(data)

    def __len__(self) -> int:
        return self.len

    def __getitem__(self, idx) -> List[Tensor]:
        return self.data[idx]


# class ParaphraseNMTDataset(BaseDataset):
#     def __init__(
#         self,
#         config_path: str,
#         root: str = None,
#         download: bool = False,
#         config_overrides: Dict = {},
#     ):
#         super().__init__(config_path, root, download)
#         dataframe = pd.read_csv(self.config["paths"]["data"])
#         tgt_col_nm = self.config["columns"]["tgt"]
#         src_col_nm = self.config["columns"]["src"]
#         tokenizer_name = (
#             config_overrides["tokenizer"]
#             if "tokenizer" in config_overrides
#             else self.config["default_params"]["tokenizer"]
#         )
#         source_len = (
#             config_overrides["source_len"]
#             if "source_len" in config_overrides
#             else self.config["default_params"]["source_len"]
#         )
#         target_len = (
#             config_overrides["target_len"]
#             if "target_len" in config_overrides
#             else self.config["default_params"]["target_len"]
#         )
#         tokenizer = T5Tokenizer.from_pretrained(tokenizer_name)
#         self.tokenizer = tokenizer
#         self.data = dataframe
#         self.source_len = source_len
#         self.summ_len = target_len
#         self.target_text = self.data[tgt_col_nm]
#         self.source_text = self.data[src_col_nm]

#     def __len__(self):
#         return len(self.target_text)

#     def __getitem__(self, index):
#         source_text = str(self.source_text[index])
#         target_text = str(self.target_text[index])

#         # cleaning data so as to ensure data is in string type
#         source_text = " ".join(source_text.split())
#         target_text = " ".join(target_text.split())

#         source = self.tokenizer.batch_encode_plus(
#             [source_text],
#             max_length=self.source_len,
#             pad_to_max_length=True,
#             truncation=True,
#             padding="max_length",
#             return_tensors="pt",
#         )
#         target = self.tokenizer.batch_encode_plus(
#             [target_text],
#             max_length=self.summ_len,
#             pad_to_max_length=True,
#             truncation=True,
#             padding="max_length",
#             return_tensors="pt",
#         )

#         source_ids = source["input_ids"].squeeze()
#         source_mask = source["attention_mask"].squeeze()
#         target_ids = target["input_ids"].squeeze()
#         target_mask = target["attention_mask"].squeeze()

#         return {
#             "source_ids": source_ids.to(dtype=long),
#             "source_mask": source_mask.to(dtype=long),
#             "target_ids": target_ids.to(dtype=long),
#             "target_ids_y": target_ids.to(dtype=long),
#         }
