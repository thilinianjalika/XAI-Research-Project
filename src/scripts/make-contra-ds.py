import argparse
import yaml
import boto3
import pandas as pd
import os
import wget
import shutil
from typing import Dict


def download_n_extract(dir: str) -> pd.DataFrame:
    url = "https://nlp.stanford.edu/projects/snli/snli_1.0.zip"
    down_path = wget.download(url, f"{dir}/snli_1.0.zip")
    shutil.unpack_archive(down_path, dir)
    return f"{dir}/snli_1.0"


def get_sentence_pairs(dir: str) -> pd.DataFrame:
    dfs = [
        {"path": os.path.join(dir, fname), "name": fname.split(".")[0]}
        for fname in os.listdir(dir)
        if fname.endswith(".txt") and fname != "README.txt"
    ]
    dfs = [{**df, "df": pd.read_csv(df["path"], delimiter="\t")} for df in dfs]
    sentence_pairs = pd.DataFrame({"sentence1": [], "sentence2": []})
    for df_obj in dfs:
        df = df_obj["df"]
        df = df[df.gold_label == "contradiction"]
        df = df[["sentence1", "sentence2"]]
        sentence_pairs = pd.concat([sentence_pairs, df], ignore_index=True)
    sentence_pairs.dropna(inplace=True)

    return sentence_pairs


def split(
    df: pd.DataFrame, train_f: float = 0.8, val_f: float = 0.1, test_f: float = 0.1
) -> Dict[str, pd.DataFrame]:
    assert train_f + val_f + test_f == 1.0

    dfs = {}
    sz = len(df)
    start_idx = 0
    end_idx = int(sz * train_f)
    dfs["train"] = df[start_idx:end_idx].reset_index(drop=True)
    start_idx = end_idx
    end_idx = start_idx + int(sz * val_f)
    dfs["val"] = df[start_idx:end_idx].reset_index(drop=True)
    start_idx = end_idx
    dfs["test"] = df[start_idx:].reset_index(drop=True)

    return dfs


def save(splitted_dfs: Dict[str, pd.DataFrame], dir: str) -> None:
    splits = ["train", "val", "test"]
    for split in splits:
        df = splitted_dfs[split]
        path = f"{dir}/snli_1.0_contra_{split}.csv"
        df.to_csv(path, index=False)


def make_ds_config(dir: str) -> None:
    datasets_key = "datasets"
    ds_name = os.path.split(dir)[1]
    paths = os.listdir(dir)
    paths = {path.split("_")[-1].split(".")[0]: path for path in paths}
    source_url = (
        f"https://sliit-xai.s3.ap-south-1.amazonaws.com/{datasets_key}/{ds_name}.zip"
    )
    model_name = "t5-small"
    max_token_len = 64
    config = {
        "name": ds_name,
        "source_url": source_url,
        "paths": paths,
        "model_name": model_name,
        "max_token_len": max_token_len,
    }
    with open(f"{dir}/{ds_name}.yaml", "w") as handler:
        yaml.dump(config, handler)


def upload_to_s3(dir: str) -> None:
    s3_bucket = "sliit-xai"
    datasets_key = "datasets"
    ds_name = os.path.split(dir)[1]
    s3 = boto3.client("s3")

    # Upload files
    for root, _, filenames in os.walk(dir):
        for filename in filenames:
            src_path = os.path.join(root, filename)
            dst_key = src_path.replace(dir, f"{datasets_key}/{ds_name}", 1)
            s3.upload_file(src_path, s3_bucket, dst_key)

    # Upload archive
    arch_path = shutil.make_archive(ds_name, "zip", dir)
    s3.upload_file(arch_path, s3_bucket, f"{datasets_key}/{ds_name}.zip")
    os.remove(arch_path)


def main(dir: str) -> None:
    if os.path.exists(dir):
        shutil.rmtree(dir)
    os.makedirs(dir)
    print("Downloading....")
    extract_dir = download_n_extract(dir)
    print("Processing....")
    sentence_pairs = get_sentence_pairs(extract_dir)
    splitted_dfs = split(sentence_pairs)
    shutil.rmtree(dir)
    os.makedirs(dir)
    print("Exporting....")
    save(splitted_dfs, dir)
    make_ds_config(dir)
    print("Uploading....")
    upload_to_s3(dir)
    print("Done!")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-d", "--directory", required=True)
    args = parser.parse_args()

    dir = args.directory
    main(dir)
