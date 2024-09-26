import numpy as np
from sklearn.metrics import classification_report
from setfit import SetFitModel, SetFitTrainer

from transformers import (
    AutoModelForSequenceClassification,
    Seq2SeqTrainingArguments,
    DataCollatorForSeq2Seq,
    AutoModelForSeq2SeqLM,
    TrainingArguments,
    TrainerCallback,
    Seq2SeqTrainer,
    AutoTokenizer,
    Trainer,
    EarlyStoppingCallback,
    TrainerControl,
)
from sklearn.naive_bayes import GaussianNB, MultinomialNB, BernoulliNB

from sentence_transformers.losses import CosineSimilarityLoss
from setfit import SetFitModel, SetFitTrainer, sample_dataset
import evaluate
import numpy as np
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from xgboost import XGBClassifier
from catboost import CatBoostClassifier
from lightgbm import LGBMClassifier
from sklearn.svm import SVC
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import classification_report

#import copy
#import pickle

import time
import pandas as pd
import torch
from typing import Dict, Any, Tuple

from src.spamdetection.preprocessing import get_dataset, train_val_test_split
from src.spamdetection.utils import SCORING, set_seed, plot_loss, plot_scores, save_scores
from src.spamdetection.transforms import transform_df, encode_df, tokenize, init_nltk

"""from src.spamdetection.utils import (
    SCORING,
    set_seed,
    plot_loss,
    plot_scores,
    save_scores,
)"""

# Added Job No.
MODELS = {
    "NB": (MultinomialNB(), 1000),
    "LR": (LogisticRegression(n_jobs=-1), 500),
    "KNN": (KNeighborsClassifier(n_neighbors=1, n_jobs=-1), 150),
    "SVM": (SVC(kernel="sigmoid", gamma=1.0), 3000),
    "XGBoost": (XGBClassifier(learning_rate=0.01, n_estimators=150, n_jobs=-1), 2000),
    "LightGBM": (LGBMClassifier(learning_rate=0.1, num_leaves=20, n_jobs=-1), 3000),
}

# Forced RoBERTa
LLMS = {
    "RoBERTa": (
        AutoModelForSequenceClassification.from_pretrained("roberta-base", num_labels=2),
        AutoTokenizer.from_pretrained("roberta-base"),
    ),
    "SetFit-mpnet": (
        SetFitModel.from_pretrained("sentence-transformers/all-mpnet-base-v2"),
        None,
    ),
    "FLAN-T5-base": (
        AutoModelForSeq2SeqLM.from_pretrained("google/flan-t5-base"),
        AutoTokenizer.from_pretrained("google/flan-t5-base"),
    ),
}
"""Custom callback to evaluate on the training set during training."""
class EvalOnTrainCallback(TrainerCallback):
    def __init__(self, trainer):
        super().__init__()
        self._trainer = trainer

    def on_epoch_end(self, args, state, control, **kwargs):
        if control.should_evaluate:
            # Create a new TrainerControl object instead of cloning
            control_train = TrainerControl()
            control_train.should_evaluate = True
            self._trainer.evaluate(eval_dataset=self._trainer.train_dataset, metric_key_prefix="train")
            return control_train

"""Return a trainer object for transformer models."""
def get_trainer(model, dataset, tokenizer=None):
    def compute_metrics(y_pred):
        logits, labels = y_pred
        predictions = np.argmax(logits, axis=-1)
        return evaluate.load("f1").compute(predictions=predictions, references=labels, average="macro")

    if isinstance(model, SetFitModel):
        trainer = SetFitTrainer(
            model=model,
            train_dataset=dataset["train"],
            eval_dataset=dataset["val"],
            loss_class=CosineSimilarityLoss,
            metric="f1",
            batch_size=16,
            num_iterations=20,
            num_epochs=3,
        )
        return trainer

    elif "T5" in model.__class__.__name__ or "FLAN" in model.__class__.__name__:
        def compute_metrics_t5(y_pred):
            predictions, labels = y_pred
            predictions = tokenizer.batch_decode(predictions, skip_special_tokens=True)
            labels = np.where(labels != -100, labels, tokenizer.pad_token_id)
            labels = tokenizer.batch_decode(labels, skip_special_tokens=True)
            predictions = [1 if "spam" in pred else 0 for pred in predictions]
            labels = [1 if "spam" in label else 0 for label in labels]
            return evaluate.load("f1").compute(predictions=predictions, references=labels, average="macro")

        data_collator = DataCollatorForSeq2Seq(tokenizer, model=model, label_pad_token_id=-100, pad_to_multiple_of=8)

        training_args = Seq2SeqTrainingArguments(
            output_dir="experiments",
            per_device_train_batch_size=8,
            per_device_eval_batch_size=8,
            learning_rate=5e-5,
            num_train_epochs=5,
            predict_with_generate=True,
            fp16=True,
            evaluation_strategy="epoch",
            save_strategy="epoch",
            load_best_model_at_end=True,
            save_total_limit=5,
        )

        trainer = Seq2SeqTrainer(
            model=model,
            args=training_args,
            train_dataset=dataset["train"],
            eval_dataset=dataset["val"],
            data_collator=data_collator,
            compute_metrics=compute_metrics_t5,
            callbacks=[EarlyStoppingCallback(early_stopping_patience=3)]
        )
        trainer.add_callback(EvalOnTrainCallback(trainer))
        return trainer

    else:
        training_args = TrainingArguments(
            output_dir="experiments",
            per_device_train_batch_size=16,
            per_device_eval_batch_size=8,
            learning_rate=5e-5,
            num_train_epochs=10,
            evaluation_strategy="epoch",
            save_strategy="epoch",
            load_best_model_at_end=True,
            save_total_limit=10,
            fp16=True,
        )

        trainer = Trainer(
            model=model,
            args=training_args,
            train_dataset=dataset["train"],
            eval_dataset=dataset["val"],
            compute_metrics=compute_metrics,
            callbacks=[EarlyStoppingCallback(early_stopping_patience=3)]
        )
        trainer.add_callback(EvalOnTrainCallback(trainer))
        return trainer

@torch.no_grad()
def predict(trainer, model, dataset, tokenizer=None):
    if isinstance(model, SetFitModel):
        return model(dataset["text"])

    elif "T5" in model.__class__.__name__:
        predictions = trainer.predict(dataset)
        predictions = tokenizer.batch_decode(predictions.predictions, skip_special_tokens=True)
        return [1 if "spam" in pred else 0 for pred in predictions]

    else:
        return trainer.predict(dataset).predictions.argmax(axis=-1)

"""Train all the large language models."""
def train_llms(seeds, datasets, train_sizes, test_set="test"):
    for seed in seeds:
        set_seed(seed)

        for dataset_name in datasets:
            for train_size in train_sizes:
                scores = pd.DataFrame(
                    index=list(LLMS.keys()),
                    columns=list(SCORING.keys()) + ["training_time", "inference_time"],
                )

                df = get_dataset(dataset_name)

                # Check if the dataset is too small for the current train_size
                if len(df) < train_size:
                    print(
                        f"Warning: Dataset '{dataset_name}' is smaller than the requested train_size of {train_size}. Skipping this configuration.")
                    continue

                _, dataset = train_val_test_split(df, train_size=train_size, has_val=True)

                experiment = f"llm_{dataset_name}_{test_set}_{train_size}_train_seed_{seed}"

                for model_name, (model, tokenizer) in LLMS.items():
                    tokenized_dataset = tokenize(dataset, tokenizer)

                    try:
                        if isinstance(model, SetFitModel):
                            # Check if all examples belong to the same class
                            if len(np.unique(dataset['train']['label'])) == 1:
                                print(
                                    f"Warning: All examples in the training set belong to the same class. Skipping SetFit training for {model_name}.")
                                continue

                        trainer = get_trainer(model, tokenized_dataset, tokenizer)

                        start = time.time()
                        train_result = trainer.train()
                        end = time.time()
                        scores.loc[model_name, "training_time"] = end - start

                        if "SetFit" not in model_name:
                            log = pd.DataFrame(trainer.state.log_history)
                            log.to_csv(f"outputs/csv/loss_{model_name}_{experiment}.csv")
                            plot_loss(experiment, dataset_name, model_name)

                        start = time.time()
                        predictions = predict(trainer, model, tokenized_dataset[test_set], tokenizer)
                        end = time.time()

                        report = classification_report(dataset[test_set]["label"], predictions, output_dict=True,
                                                       zero_division=0)

                        # Handle binary classification case
                        if 'f1-score' in report:
                            scores.loc[model_name, "f1"] = report['f1-score']
                        elif '1' in report:  # Assuming '1' is the positive class
                            scores.loc[model_name, "f1"] = report['1']['f1-score']
                        else:
                            scores.loc[model_name, "f1"] = np.nan

                        for score_name in SCORING.keys():
                            if score_name == 'f1':
                                continue  # We've already handled this above
                            if score_name in report:
                                scores.loc[model_name, score_name] = report[score_name]
                            elif '1' in report:  # Assuming '1' is the positive class
                                scores.loc[model_name, score_name] = report['1'][score_name]
                            else:
                                scores.loc[model_name, score_name] = np.nan

                        scores.loc[model_name, "inference_time"] = end - start
                        save_scores(experiment, model_name, scores.loc[model_name].to_dict())

                    except Exception as e:
                        print(f"An error occurred while training {model_name}: {str(e)}")
                        scores.loc[model_name] = np.nan

                plot_scores(experiment, dataset_name)
                print(scores)

"""Train all the baseline models."""
def train_baselines(seeds, datasets, train_sizes, test_set="test"):
    init_nltk()

    for seed in seeds:
        set_seed(seed)

        for dataset_name in datasets:
            for train_size in train_sizes:
                scores = pd.DataFrame(
                    index=list(MODELS.keys()),
                    columns=list(SCORING.keys()) + ["training_time", "inference_time"],
                )

                df = get_dataset(dataset_name)
                df = transform_df(df)
                (df_train, df_val, df_test), _ = train_val_test_split(df, train_size=train_size, has_val=True)

                experiment = f"ml_{dataset_name}_{test_set}_{train_size}_train_seed_{seed}"

                for model_name, (model, max_iter) in MODELS.items():
                    encoder = TfidfVectorizer(max_features=max_iter)
                    X_train, y_train, encoder = encode_df(df_train, encoder)
                    X_test, y_test, encoder = encode_df(df_test, encoder)

                    if test_set == "val":
                        X_val, y_val, _ = encode_df(df_val, encoder)
                        start = time.time()
                        model.fit(X_train, y_train)
                        end = time.time()
                        scores.loc[model_name]["training_time"] = end - start

                        start = time.time()
                        y_pred = model.predict(X_val)
                        end = time.time()

                        report = classification_report(y_val, y_pred, output_dict=True)
                    else:
                        start = time.time()
                        model.fit(X_train, y_train)
                        end = time.time()
                        scores.loc[model_name]["training_time"] = end - start

                        start = time.time()
                        y_pred = model.predict(X_test)
                        end = time.time()

                        report = classification_report(y_test, y_pred, output_dict=True)

                    scores.loc[model_name]["inference_time"] = end - start
                    for score_name in SCORING.keys():
                        scores.loc[model_name][score_name] = report[score_name]

                    save_scores(experiment, model_name, scores.loc[model_name].to_dict())

                plot_scores(experiment, dataset_name)
                print(scores)

if __name__ == "__main__":
    seeds = [42, 123, 456]
    datasets = ["enron", "ling", "sms", "spamassassin", "spam1","spam2","spam3","spam4","spam5", "spam6", "spam7","spam8", "spam9"]
    train_sizes = [500, 1000, 5000]

    train_llms(seeds, datasets, train_sizes)
    train_baselines(seeds, datasets, train_sizes)