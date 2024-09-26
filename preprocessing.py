import os
import glob
import tarfile
import datasets
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from io import BytesIO
from urllib.request import urlopen
from zipfile import ZipFile
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor
import pyarrow.csv as pv
import pyarrow as pa
from typing import Tuple, Optional
import csv


def get_dataset(name: str) -> pd.DataFrame:
    #return pd.read_csv(f"data/processed/{name}/data.csv", engine='pyarrow').dropna()
    file_path = f"data/processed/{name}/data.csv"

    try:
        df = pd.read_csv(file_path, engine='python')
    except pd.errors.ParserError:
        # If that fails, try to read the file with a more lenient approach
        with open(file_path, 'r', encoding='utf-8') as f:
            lines = f.readlines()

        cleaned_lines = [line for line in lines if len(line.split(',')) == 2]  # Assuming we expect 2 columns

        cleaned_file_path = f"data/processed/{name}/cleaned_data.csv"
        with open(cleaned_file_path, 'w', encoding='utf-8', newline='') as f:
            f.writelines(cleaned_lines)

        df = pd.read_csv(cleaned_file_path, engine='python')

    if 'text' not in df.columns or 'label' not in df.columns:
        raise ValueError(f"The dataset {name} does not have the expected 'text' and 'label' columns")

    return df[['text', 'label']].dropna()


def download_and_extract(url: str, extract_path: str) -> None:
    """ a zip or tar file"""
    with urlopen(url) as file:
        if url.endswith('.zip'):
            with ZipFile(BytesIO(file.read())) as zfile:
                zfile.extractall(extract_path)
        elif url.endswith('.tar.gz') or url.endswith('.tar.bz2'):
            with tarfile.open(fileobj=BytesIO(file.read())) as tfile:
                tfile.extractall(extract_path)

def process_file(file_path: str, label: int) -> Tuple[str, int]:
    """Process a single file and return text and label"""
    with open(file_path, 'r', encoding='ISO-8859-1') as file:
        text = file.read()
    return text, label


def preprocess_dataset(name: str, url: str, process_func) -> None:
    """Generic function to preprocess a dataset"""
    raw_path = f"data/raw/{name}"
    processed_path = f"data/processed/{name}"
    Path(raw_path).mkdir(parents=True, exist_ok=True)
    Path(processed_path).mkdir(parents=True, exist_ok=True)

    download_and_extract(url, raw_path)

    data = process_func(raw_path)

    df = pd.DataFrame(data, columns=["text", "label"])
    df = df.dropna().drop_duplicates()

    df.to_csv(f"{processed_path}/data.csv", index=False)


def process_enron(raw_path: str) -> list:
    df = pv.read_csv(f"{raw_path}/enron_spam_data.csv").to_pandas()
    df['text'] = df['Subject'] + df['Message']
    df['label'] = df['Spam/Ham'].map({"ham": 0, "spam": 1})
    return df[['text', 'label']].values.tolist()


def process_ling(raw_path: str) -> list:
    path = f"{raw_path}/lingspam_public/bare/*/*"
    with ProcessPoolExecutor() as executor:
        data = list(executor.map(lambda f: process_file(f, 1 if "spmsg" in f else 0), glob.glob(path)))
    return data


def process_sms(raw_path: str) -> list:
    df = pv.read_csv(f"{raw_path}/SMSSpamCollection", delimiter='\t', header=None).to_pandas()
    df = df.rename(columns={0: "label", 1: "text"})
    df["label"] = df["label"].map({"ham": 0, "spam": 1})
    return df[['text', 'label']].values.tolist()


def process_spamassassin(raw_path: str) -> list:
    path = f"{raw_path}/*/*"
    with ProcessPoolExecutor() as executor:
        data = list(executor.map(lambda f: process_file(f, 0 if "ham" in f else 1), glob.glob(path)))
    return data


def preprocess_enron() -> None:
    preprocess_dataset("enron", "https://github.com/MWiechmann/enron_spam_data/raw/master/enron_spam_data.zip",
                       process_enron)
    # Download and extract
    #url = "https://github.com/MWiechmann/enron_spam_data/raw/master/enron_spam_data.zip"
    #with urlopen(url) as zurl:
    #    with ZipFile(BytesIO(zurl.read())) as zfile:
    #        zfile.extractall("data/raw/enron")
    # Load dataset
    #df = pd.read_csv("data/raw/enron/enron_spam_data.csv", encoding="ISO-8859-1")

    # Preprocess
    #df = df.fillna("")
    #df["text"] = df["Subject"] + df["Message"]
    #df["label"] = df["Spam/Ham"].map({"ham": 0, "spam": 1})
    #df = df[["text", "label"]]
    #df = df.dropna()
    #df = df.drop_duplicates()

    # Save
    #df.to_csv("data/processed/enron/data.csv", index=False)


def preprocess_ling() -> None:
    preprocess_dataset("ling", "https://github.com/oreilly-japan/ml-security-jp/raw/master/ch02/lingspam_public.tar.gz",
                       process_ling)

    """ Clean and rename the dataset and save it in data/processed
        Path("data/raw/ling").mkdir(parents=True, exist_ok=True)
        Path("data/processed/ling").mkdir(parents=True, exist_ok=True)

        # Download and extract
        url = "https://github.com/oreilly-japan/ml-security-jp/raw/master/ch02/lingspam_public.tar.gz"
        r = urlopen(url)
        t = tarfile.open(name=None, fileobj=BytesIO(r.read()))
        t.extractall("data/raw/ling")
        t.close()

        path = r"data/raw/ling/lingspam_public/bare/*/*"
        data = []

        for fn in glob.glob(path):
            label = 1 if "spmsg" in fn else 0

            with open(fn, "r", encoding="ISO-8859-1") as file:
                text = file.read()
                data.append((text, label))

        df = pd.DataFrame(data, columns=["text", "label"])
        df = df.dropna()
        df = df.drop_duplicates()

        # Save
        df.to_csv("data/processed/ling/data.csv", index=False)"""


def preprocess_sms() -> None:
    preprocess_dataset("sms", "https://archive.ics.uci.edu/ml/machine-learning-databases/00228/smsspamcollection.zip",
                       process_sms)

    """Clean and rename the dataset and save it in data/processed
    Path("data/raw/sms").mkdir(parents=True, exist_ok=True)
    Path("data/processed/sms").mkdir(parents=True, exist_ok=True)

    # Download and extract
    url = "https://archive.ics.uci.edu/ml/machine-learning-databases/00228/smsspamcollection.zip"
    with urlopen(url) as zurl:
        with ZipFile(BytesIO(zurl.read())) as zfile:
            zfile.extractall("data/raw/sms")

    # Load dataset
    df = pd.read_csv("data/raw/sms/SMSSpamCollection", sep="\t", header=None)

    # Clean dataset
    df = df.drop_duplicates(keep="first")

    # Rename
    df = df.rename(columns={0: "label", 1: "text"})
    df["label"] = df["label"].map({"ham": 0, "spam": 1})

    # Preprocessing
    df = df.dropna()
    df = df.drop_duplicates()

    # Save
    df.to_csv("data/processed/sms/data.csv", index=False)"""




def preprocess_spamassassin() -> None:
    urls = [
        "https://spamassassin.apache.org/old/publiccorpus/20030228_easy_ham.tar.bz2",
        "https://spamassassin.apache.org/old/publiccorpus/20030228_easy_ham_2.tar.bz2",
        "https://spamassassin.apache.org/old/publiccorpus/20030228_hard_ham.tar.bz2",
        "https://spamassassin.apache.org/old/publiccorpus/20030228_spam.tar.bz2",
        "https://spamassassin.apache.org/old/publiccorpus/20050311_spam_2.tar.bz2",
    ]
    for url in urls:
        download_and_extract(url, "data/raw/spamassassin")
    preprocess_dataset("spamassassin", "", process_spamassassin)

"""    for url in urls:
        r = urlopen(url)
        t = tarfile.open(name=None, fileobj=BytesIO(r.read()))
        t.extractall("data/raw/spamassassin")
        t.close()

    path = r"data/raw/spamassassin/*/*"
    data = []

    for fn in glob.glob(path):
        label = 0 if "ham" in fn else 1

        with open(fn, "r", encoding="ISO-8859-1") as file:
            text = file.read()
            data.append((text, label))

    df = pd.DataFrame(data, columns=["text", "label"])
    df = df.dropna()
    df = df.drop_duplicates()

    # Save
    df.to_csv("data/processed/spamassassin/data.csv", index=False)"""

def init_datasets() -> None:
    with ProcessPoolExecutor() as executor:
        executor.map(lambda f: f(), [preprocess_enron, preprocess_ling, preprocess_sms, preprocess_spamassassin])

    """preprocess_enron()
    preprocess_ling()
    preprocess_sms()
    preprocess_spamassassin()"""

"""def train_val_test_split(df, train_size=0.8, has_val=True):
    #Return a tuple (DataFrame, DatasetDict) with a custom train/val/split
    # Convert int train_size into float
    if isinstance(train_size, int):
        train_size = train_size / len(df)

    # Shuffled train/val/test split
    df = df.sample(frac=1, random_state=0)
    df_train, df_test = train_test_split(
        df, test_size=1 - train_size, stratify=df["label"]
    )

    if has_val:
        df_test, df_val = train_test_split(
            df_test, test_size=0.5, stratify=df_test["label"]
        )
        return (
            (df_train, df_val, df_test),
            datasets.DatasetDict(
                {
                    "train": datasets.Dataset.from_pandas(df_train),
                    "val": datasets.Dataset.from_pandas(df_val),
                    "test": datasets.Dataset.from_pandas(df_test),
                }
            ),
        )

    else:
        return (
            (df_train, df_test),
            datasets.DatasetDict(
                {
                    "train": datasets.Dataset.from_pandas(df_train),
                    "test": datasets.Dataset.from_pandas(df_test),
                }
            ),
        )
"""
def train_val_test_split(df: pd.DataFrame, train_size: float = 0.8, has_val: bool = True) -> Tuple[
    Tuple, datasets.DatasetDict]:
    """Return a tuple (DataFrame, DatasetDict) with a custom train/val/split"""
    train_size = train_size if isinstance(train_size, float) else train_size / len(df)

    # Check if we have enough samples for a meaningful split
    if len(df) < 3:
        print(f"Warning: Dataset too small for splitting. Using entire dataset for train, val, and test.")
        return (
            (df, df, df),
            datasets.DatasetDict({
                "train": datasets.Dataset.from_pandas(df),
                "val": datasets.Dataset.from_pandas(df),
                "test": datasets.Dataset.from_pandas(df),
            })
        )

    df_train, df_test = train_test_split(df, test_size=1 - train_size, stratify=df["label"], random_state=42)

    if has_val:
        if len(df_test) < 2:
            print(f"Warning: Test set too small to split into val and test. Using entire test set for both.")
            df_val = df_test
        else:
            df_test, df_val = train_test_split(df_test, test_size=0.5, stratify=df_test["label"], random_state=42)

        return (
            (df_train, df_val, df_test),
            datasets.DatasetDict({
                "train": datasets.Dataset.from_pandas(df_train),
                "val": datasets.Dataset.from_pandas(df_val),
                "test": datasets.Dataset.from_pandas(df_test),
            })
        )
    else:
        return (
            (df_train, df_test),
            datasets.DatasetDict({
                "train": datasets.Dataset.from_pandas(df_train),
                "test": datasets.Dataset.from_pandas(df_test),
            })
        )


if __name__ == "__main__":
    init_datasets()