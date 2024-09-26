import nltk
from nltk.tokenize import RegexpTokenizer
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from typing import Tuple, Optional

# Initialize tokenizer, stopwords, and stemmer
tokenizer = RegexpTokenizer(r'\w+')
stop_words = set(stopwords.words("english"))
ps = PorterStemmer()

def init_nltk():
    nltk.download("punkt")
    nltk.download('stopwords')

def tokenize_words(text: str) -> list:
    """Tokenize words in text and remove punctuation"""
    return tokenizer.tokenize(str(text).lower())

def remove_stopwords(tokens: list) -> list:
    """Remove stopwords from the tokens"""
    return [token for token in tokens if token not in stop_words]

def stem(tokens: list) -> list:
    """Stem the tokens"""
    return [ps.stem(token) for token in tokens]

def transform(text: str) -> str:
    """Tokenize, remove stopwords, stem the text"""
    tokens = tokenize_words(text)
    tokens = remove_stopwords(tokens)
    tokens = stem(tokens)
    return " ".join(tokens)

def transform_df(df):
    """Apply the transform function to the dataframe"""
    df["transformed_text"] = df["text"].apply(transform)
    return df

def encode_df(df, encoder: Optional[TfidfVectorizer] = None) -> Tuple[np.ndarray, np.ndarray, TfidfVectorizer]:
    """Encode the features for training set"""
    if encoder is None or not hasattr(encoder, 'vocabulary_'):
        encoder = TfidfVectorizer(max_features=5000)
        X = encoder.fit_transform(df["transformed_text"]).toarray()
    else:
        X = encoder.transform(df["transformed_text"]).toarray()
    y = df["label"].values
    return X, y, encoder

def tokenize(dataset, tokenizer):
    """Tokenize dataset"""
    def tokenization(examples):
        return tokenizer(examples["text"], padding="max_length", truncation=True)

    def tokenization_t5(examples, padding="max_length"):
        text = ["classify as ham or spam: " + item for item in examples["text"]]
        inputs = tokenizer(text, max_length=tokenizer.model_max_length, padding=padding, truncation=True)
        labels = tokenizer(text_target=examples["label"], max_length=max_label_length, padding=True, truncation=True)
        inputs["labels"] = [[x if x != tokenizer.pad_token_id else -100 for x in label] for label in labels["input_ids"]]
        return inputs

    if tokenizer is None:
        return dataset

    elif "T5" in type(tokenizer).__name__:
        dataset = dataset.map(lambda x: {"label": "ham" if x["label"] == 0 else "spam"})
        tokenized_label = dataset["train"].map(lambda x: tokenizer(x["label"], truncation=True), batched=True)
        max_label_length = max([len(x) for x in tokenized_label["input_ids"]])
        return dataset.map(tokenization_t5, batched=True, remove_columns=["label"])

    else:
        return dataset.map(tokenization, batched=True)