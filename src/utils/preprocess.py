import os
import pandas as pd
import glob
from typing import Optional, Tuple
from isbnlib import to_isbn13
from langcodes import Language
import numpy as np
from pathlib import Path


def load_user_ratings() -> pd.DataFrame:
    """
    Load user ratings from CSV files located in the 'goodreads_2m/raw' directory.
    Returns a concatenated DataFrame of all user ratings.
    """
    # Define the path to the 'goodreads_2m/raw' directory
    data_dir = Path(__file__).resolve().parent.parent / "data" / "goodreads_2m" / "raw"

    # Use a list comprehension to find all CSV files matching the pattern
    csv_files = [file for file in data_dir.glob("user_rating_*.csv")]

    if not csv_files:
        raise FileNotFoundError(f"No user rating CSV files found in {data_dir}")

    # Load and concatenate all found CSV files into a single DataFrame
    user_ratings_df = pd.concat(
        [pd.read_csv(file) for file in csv_files], ignore_index=True
    )

    return user_ratings_df


def convert_to_isbn13(value: str) -> Optional[str]:
    """
    Convert a value to ISBN-13. If it is impossible
    to convert the value to ISBN-13, None will be returned.

    :param value: Value to convert.
    :return: Converted value or None.
    """
    try:
        new_value = to_isbn13(value)
        return new_value if new_value else None
    except Exception as e:  # Catch broader exceptions for unexpected issues
        print(f"Error converting ISBN: {e}")  # Logging error for debugging
        return None


def get_weighted_rating(
    df: pd.DataFrame, min_rate_count: float, mean_rate: float
) -> float:
    """
    Calculate weighted rating using information about
    average rating and the number of votes it has accumulated.

    :param df: DataFrame with ratings of an item.
    :param min_rate_count: Minimum rating count
    required to be listed in the chart.
    :param mean_rate: Mean rating across the whole ratings.
    :return: Weighted rating as a float.
    """
    rates = df["rating"].dropna()
    ratings_count = rates.count()
    ratings_mean = rates.mean()

    if ratings_count == 0:  # Check for division by zero
        return 0.0

    weighted_rating = (
        ratings_count * ratings_mean / (ratings_count + min_rate_count)
    ) + (min_rate_count * mean_rate / (ratings_count + min_rate_count))

    return weighted_rating


def normalize_language_code(language_code: str) -> Optional[str]:
    """
    Normalize language code to its corresponding language name, if possible.

    :param language_code: Language code to normalize.
    :return: Normalized language name or None if not found.
    """
    try:
        language = Language.get(language_code)
        return language.display_name()
    except Exception as e:  # Broader exception handling
        print(f"Error normalizing language code: {e}")  # Logging for debugging
        return None


def stratified_split(
    df: pd.DataFrame, by: str, train_size: float = 0.70
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Stratified split of a DataFrame by a specific column
    with a given train size percentage.

    :param df: DataFrame to split.
    :param by: Column name to stratify by.
    :param train_size: Proportion of the dataset
    to include in the train split.
    :return: Tuple of (train, test) DataFrames.
    """
    if not 0 < train_size < 1:
        raise ValueError("train_size must be between 0 and 1")

    splits = []
    for _, group in df.groupby(by):
        group = group.sample(frac=1, random_state=123)
        split_idx = round(train_size * len(group))
        group_splits = [group.iloc[:split_idx], group.iloc[split_idx:]]

        # Label the train and test sets.
        for i, split in enumerate(group_splits):
            split["split_index"] = i
            splits.append(split)

    # Concatenate splits for all the groups together.
    splits_all = pd.concat(splits, ignore_index=True)

    # Separate train and test splits using split_index.
    train = splits_all[splits_all["split_index"] == 0].drop("split_index", axis=1)
    test = splits_all[splits_all["split_index"] == 1].drop("split_index", axis=1)

    return train, test


def numpy_stratified_split(
    X: np.ndarray, ratio: float = 0.75, seed: int = 42
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Stratified split of a numpy array into train and test sets,
    preserving the ratio of non-zero elements in each row.

    :param X: Numpy array to split.
    :param ratio: Ratio of the train set size to the total dataset size.
    :param seed: Random seed for reproducibility.
    :return: Tuple of (train, test) numpy arrays.
    """
    if not 0 < ratio < 1:
        raise ValueError("ratio must be between 0 and 1")

    np.random.seed(seed)  # set the random seed
    test_cut = int((1 - ratio) * 100)  # percentage of items to go in the test set

    Xtr = X.copy()
    Xtst = np.zeros_like(X)

    for u in range(X.shape[0]):
        idx = np.where(X[u] != 0)[0]  # indexes of non-zero ratings
        tst_size = np.around(len(idx) * test_cut / 100).astype(
            int
        )  # number of items to move to test

        if tst_size > 0:
            idx_tst = np.random.choice(
                idx, tst_size, replace=False
            )  # select random indices for test set
            Xtr[u, idx_tst] = 0  # set selected indices to 0 in train set
            Xtst[u, idx_tst] = X[u, idx_tst]  # copy selected indices to test set

    return Xtr, Xtst
