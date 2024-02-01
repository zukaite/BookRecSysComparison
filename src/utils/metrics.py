import numpy as np
import pandas as pd
from typing import Any, Tuple


def relevant_df(
    recommended: pd.DataFrame,
    test: pd.DataFrame,
    user_col: str,
    item_col: str,
    rank_col: str,
) -> pd.DataFrame:
    """
    Calculate the dataframe of relevant items by merging recommended items with test items based on user and item columns.

    Parameters:
    - recommended: DataFrame containing recommended items.
    - test: DataFrame containing test items.
    - user_col: Column name representing user IDs.
    - item_col: Column name representing item IDs.
    - rank_col: Column name representing ranking.

    Returns:
    DataFrame containing relevant items and their counts.
    """
    try:
        df_relevant = pd.merge(recommended, test, on=[user_col, item_col])[
            [user_col, item_col, rank_col]
        ]
        df_relevant_count = df_relevant.groupby(user_col, as_index=False)[user_col].agg(
            {"relevant": "count"}
        )
        test_count = test.groupby(user_col, as_index=False)[user_col].agg(
            {"actual": "count"}
        )
        relevant_ratio = pd.merge(df_relevant_count, test_count, on=user_col)
        return relevant_ratio
    except Exception as e:
        raise ValueError(f"Error calculating relevant dataframe: {e}")


def precision_at_k(
    recommended: pd.DataFrame,
    test: pd.DataFrame,
    user_col: str,
    item_col: str,
    rank_col: str,
) -> float:
    """
    Calculate precision at k metric for recommender system evaluation.

    Parameters:
    - recommended: DataFrame containing recommended items.
    - test: DataFrame containing test items.
    - user_col: Column name representing user IDs.
    - item_col: Column name representing item IDs.
    - rank_col: Column name representing ranking.

    Returns:
    Float representing the precision at k metric.
    """
    relevant_ratio = relevant_df(recommended, test, user_col, item_col, rank_col)
    precision_at_k = (
        (relevant_ratio["relevant"] / 10) / len(test[user_col].unique())
    ).sum()
    return precision_at_k


def recall_at_k(
    recommended: pd.DataFrame,
    test: pd.DataFrame,
    user_col: str,
    item_col: str,
    rank_col: str,
) -> float:
    """
    Calculate recall at k metric for recommender system evaluation.

    Parameters:
    - recommended: DataFrame containing recommended items.
    - test: DataFrame containing test items.
    - user_col: Column name representing user IDs.
    - item_col: Column name representing item IDs.
    - rank_col: Column name representing ranking.

    Returns:
    Float representing the recall at k metric.
    """
    relevant_ratio = relevant_df(recommended, test, user_col, item_col, rank_col)
    recall_at_k = (
        (relevant_ratio["relevant"] / relevant_ratio["actual"])
        / len(test[user_col].unique())
    ).sum()
    return recall_at_k


def mean_average_precision(
    recommended: pd.DataFrame,
    test: pd.DataFrame,
    user_col: str,
    item_col: str,
    rank_col: str,
) -> float:
    """
    Calculate mean average precision (MAP) for recommender system evaluation.

    Parameters:
    - recommended: DataFrame containing recommended items.
    - test: DataFrame containing test items.
    - user_col: Column name representing user IDs.
    - item_col: Column name representing item IDs.
    - rank_col: Column name representing ranking.

    Returns:
    Float representing the mean average precision.
    """
    df_relevant = pd.merge(recommended, test, on=[user_col, item_col])[
        [user_col, item_col, rank_col]
    ]
    relevant_ratio = relevant_df(recommended, test, user_col, item_col, rank_col)
    df_relevant["precision@k"] = (
        df_relevant.groupby(user_col).cumcount() + 1
    ) / df_relevant[rank_col]
    relevant_ordered = (
        df_relevant.groupby(user_col).agg({"precision@k": "sum"}).reset_index()
    )
    merged = pd.merge(relevant_ordered, relevant_ratio, on=user_col)
    merged["avg_precision"] = merged["precision@k"] / merged["actual"]
    score = merged["avg_precision"].sum() / len(test[user_col].unique())
    return score


def ndcg(
    recommended: pd.DataFrame,
    test: pd.DataFrame,
    user_col: str,
    item_col: str,
    rank_col: str,
) -> float:
    """
    Calculate normalized discounted cumulative gain (NDCG) for recommender system evaluation.

    Parameters:
    - recommended: DataFrame containing recommended items.
    - test: DataFrame containing test items.
    - user_col: Column name representing user IDs.
    - item_col: Column name representing item IDs.
    - rank_col: Column name representing ranking.

    Returns:
    Float representing the NDCG score.
    """
    df_relevant = pd.merge(recommended, test, on=[user_col, item_col])[
        [user_col, item_col, rank_col]
    ]
    relevant_ratio = relevant_df(recommended, test, user_col, item_col, rank_col)
    df_relevant["dcg"] = 1 / np.log1p(df_relevant[rank_col])
    dcg = df_relevant.groupby(user_col, as_index=False, sort=False).agg({"dcg": "sum"})
    ndcg = pd.merge(dcg, relevant_ratio, on=user_col)
    ndcg["idcg"] = ndcg["actual"].apply(
        lambda x: sum(1 / np.log1p(range(1, min(x, 10) + 1)))
    )
    score = (ndcg["dcg"] / ndcg["idcg"]).sum() / len(test[user_col].unique())
    return score
