"""
This module contains helper functions specifically written for
the DS 8008 project.

Author: Robert Helmeczi
Date: April 17 2022
"""
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.linear_model import LogisticRegression
import pet
import os
import numpy as np
import tqdm
import datasets
import pandas as pd

def get_yelp_polarity_data(show_progress_bars=False):
    """
    Download and load the Yelp polarity dataset.

    Returns (Labeled Training Data, Unlabeled Training Data, Test Data)

    Notes:
        The unlabeled training data has labels! We will use these labels
        to investigate the performance of the MLM, but we will NEVER use
        them for training purposes. In a real weak supervision task, we
        would not have access to these labels at all.
    """
    if not show_progress_bars: datasets.logging.disable_progress_bar()
    yelp_polarity_dataset = datasets.load_dataset('yelp_polarity')
    df_train = pd.DataFrame(yelp_polarity_dataset['train'][:2040])
    df_test = pd.DataFrame(yelp_polarity_dataset['test'])
    df_train['label'] = (df_train['label'] + 1).astype(str)
    df_test['label'] = (df_test['label'] + 1).astype(str)
    return df_train.loc[:40], df_train.loc[40:], df_test

def prepare_data(X, y=None):
    """
    Prepare text data for training or evaluation as input into PET functions.

    Args:
        X (Sequence[str]): Text data.
        y (Sequence[Any], optional): Labels.
    """
    if y is None:
        return [pet.InputExample(i, text) for i, text in enumerate(X)]
    else:
        return [pet.InputExample(i, text, label=label)
                for i, (text, label) in enumerate(zip(X, y))]

def train_models(pattern_ids, X_train, y_train, task_name, label_list,
                 output_dir='../trained_models'):
    """
    Fine tune a model for each supplied pattern id.

    Args:
        pattern_ids (Sequence[int]): The ids of the patterns to train.
            Patterns can be observed in pvp.py by looking at the
            PVP for `task_name`.
        X_train (Sequence[str]): The reviews.
        y_train (Sequence[Any]): The labels for the data.
        task_name (str): The task to train on. For example, 'yelp-polarity'.
        output_dir (str): The path to save the models at. Defaults to
            `../trained_models`.

    Returns:
        List[str]: The paths to the saved models.

    Notes:
        For the Yelp polarity task, the labels are "1" and "2". The
        verbalizer is common to all patterns, where "1" maps to "bad"
        and "2" maps to "good".

        The patterns used in the implementation are as follows, where
        {review} is substituted by the actual review.

        | id | pattern                            |
        |----|------------------------------------|
        | 0  | It was ___. {review}               |
        | 1  | {review}. All in all, it was ___.  |
        | 2  | Just ___! || {review}              |
    """
    wrapper_type = 'mlm' # tells PET we are using a masked language model
    model_type = 'roberta'
    model_name = 'roberta-large'
    train_data = prepare_data(X_train, y_train)
    # models will be saved locally. This stores the paths they are saved to
    model_paths = []
    for id in tqdm.tqdm(pattern_ids, desc='Training patterns', position=1, leave=True):
        """
        The arguments below are chosen following two criteria:
            1. To match the parameters used in the paper as much as possible.
            2. To ensure GPU memory was never exceeded. `per_gpu_train_batch_size`
                and `per_gpu_eval_batch_size` are big factors in memory
                problems.
        We train each model separately to ensure we do not run out of memory.
        """
        wrapper_cfg = pet.wrapper.WrapperConfig(model_type=model_type,
                                                model_name_or_path=model_name,
                                                pattern_id=id,
                                                task_name=task_name,
                                                wrapper_type=wrapper_type,
                                                max_seq_length=256,
                                                label_list=label_list)
        # decreasing per_gpu_train_batch_size helps with out of memory problems
        train_cfg = pet.TrainConfig('cuda', max_steps=10,
                                    gradient_accumulation_steps=16,
                                    per_gpu_train_batch_size=1)
        eval_cfg = pet.EvalConfig(device='cuda', n_gpu=1, per_gpu_eval_batch_size=1,
                                  metrics=['acc'], decoding_strategy='default',
                                  priming=False)
        pet.modeling.train_pet_ensemble(
            wrapper_cfg, train_cfg, eval_cfg, [id],
            output_dir=output_dir, train_data=train_data, do_eval=False,
            save_unlabeled_logits=False, ipet_data_dir=None, repetitions=1
        )
        model_paths.append(os.path.join(output_dir, f'p{id}-i0'))
    return model_paths

def score_models(model_paths, X_train, y_train, label_list):
    """
    Return the accuracies of a set of trained models.

    The accuracies are used to weight the predictions of trained models
    when combining predictions.

    Args:
        model_paths (Sequence[str]): The paths to the models.
        X_train (Sequence[str]): The training data to predict.
        y_train (Sequence[Any]): The predictions.
        labels_list (Sequence[Any]): The labels for the data.
    """
    scores = []
    label_list = np.array(label_list)
    train_data = prepare_data(X_train)
    for path in model_paths:
        # do predictions and map them to their corresponding label
        predictions = label_list[predict(train_data, path)['predictions']]
        scores.append(np.sum(y_train == predictions) / len(y_train))
    return scores


def ensemble_predict(model_paths, scores, X_unlabeled):
    """
    Return the probabilities of each class label for a set of unlabeled
    data.

    The combined probabilities are simply weighted by the accuracy on the
    training set.

    Args:
        model_paths (Sequence[str]): The paths to the models.
        scores (Sequence[float]): The accuracies of each model.
        X_unlabeled (Sequence[str]): The unlabeled data to predict.
    """
    ensemble_probs = None
    unlabeled_data = prepare_data(X_unlabeled)
    for score, path in zip(scores, model_paths):
        probs = predict(unlabeled_data, path)['logits']
        # the logits are returned as a matrix, so let's apply softmax row-wise
        probs -= np.max(probs, axis=1, keepdims=True) # prevents overflow
        probs = np.exp(probs)
        probs /= probs.sum(axis=1, keepdims=True) # normalize
        probs *= score # weight the probability by its accuracy
        if ensemble_probs is None: # for first run through of the for loop
            ensemble_probs = probs
        else:
            ensemble_probs += probs
    # normalize probabilities and return
    return ensemble_probs / ensemble_probs.sum(axis=1, keepdims=True)

def predict(examples, model_path):
    """
    Return prediction results from a set of examples.

    Args:
        examples (Sequence[pet.InputExample]): The data to predict.
        model_path (str): The path to the pretrained transformer model.
    """
    eval_config = pet.EvalConfig(device='cuda', n_gpu=1,
                                 per_gpu_eval_batch_size=8,
                                 metrics=['acc'],
                                 decoding_strategy='default', priming=False)
    wrapper = pet.wrapper.TransformerModelWrapper.from_pretrained(model_path)
    return pet.evaluate(wrapper, examples, eval_config)

def score_predictions(y_true, pred_probs, label_list):
    """
    Return the prediction accuracy given class probabilities.

    Args:
        y_true (np.ndarray): The ground truth labels.
        pred_probs (np.ndarray): The probabilities for each label. Rows
            are samples, columns are labels.
        label_list (np.ndarray): The labels for the columns in `pred_probs`.
    """
    return np.sum(y_true == label_list[pred_probs.argmax(axis=1)]) / len(y_true)

def train_and_score_logistic_regression_model(X_train, y_train, X_test, y_test,
                                              unlabeled_X_train=None,
                                              unlabeled_probabilities=None,
                                              label_list=None):
    """
    Train and get the accuracy of a logistic regression model.

    If unlabeled data are provided with predictions, they are incorporated
    into the training data.

    Args:
        X_train (Sequence[str]): The labeled training data.
        y_train (Sequence[Any]): The labels for the training data.
        X_test (Sequence[str]): The labeled test data.
        y_test (Sequence[Any]): The labels for the test data.
        unlabeled_X_train (Sequence[str], optional): The training data
            for which we have generated weak labels.
        unlabeled_probabilities (Sequence[Sequence[float]], optional):
            The probabilities of each label for the unlabeled training
            data.
        label_list (Sequence[Any]): The list of labels. These correspond
            to the columns of `unlabeled_probabilities`.

    Notes:
        The idea to use logstic regression came from a Snorkel tutorial:

            https://www.snorkel.org/use-cases/01-spam-tutorial

        As mentioned in the writeup, PET typically fine tunes a sequence
        classification head as the final model. We choose logistic
        regression here because it performs sufficiently well and returns
        a result much quicker than a PLM does. Feel free to see if you
        can improve the accuracy of the final model just by training a
        different classifier.
    """
    vectorizer = CountVectorizer(ngram_range=(1, 5))
    if unlabeled_X_train is not None:
        X_train = list(X_train) + list(unlabeled_X_train)
        if unlabeled_probabilities is None: raise ValueError('unlabeled data require predictions')
        y_train = list(y_train) + label_list[np.argmax(unlabeled_probabilities, axis=1)].tolist()
    X_train_transformed = vectorizer.fit_transform(X_train)
    X_test_transformed = vectorizer.transform(X_test)
    model = LogisticRegression()
    model.fit(X=X_train_transformed, y=y_train)
    return model.score(X=X_test_transformed, y=y_test)
