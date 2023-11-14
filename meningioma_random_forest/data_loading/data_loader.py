from pathlib import Path
from typing import Tuple, List

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

from meningioma_random_forest.config import RANDOM_SEED
from meningioma_random_forest.training.remove_correlated_features import (
    remove_correlated_features,
)


def _load_data(path: Path):
    return pd.read_csv(path, index_col=0)


def _make_train_test_split(
    full_dataframe: pd.DataFrame, test_size: float, random_seed: int
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    return train_test_split(
        full_dataframe,
        test_size=test_size,
        random_state=random_seed,
        stratify=full_dataframe["label"],
    )


def do_scaling(x_dataframe: pd.DataFrame, scaler=StandardScaler()):
    x_dataframe_scaled = scaler.fit_transform(x_dataframe)
    return (
        pd.DataFrame(
            x_dataframe_scaled, index=x_dataframe.index, columns=x_dataframe.columns
        ),
        scaler,
    )


def get_metadata_columns(dataframe: pd.DataFrame) -> List[str]:
    metadata_columns = ["image", "mask"]
    for column in dataframe.columns:
        if column.startswith("diagnostics_"):
            metadata_columns.append(column)
    return metadata_columns


def remove_irrelevant_features(
    x_dataframe: pd.DataFrame,
    only_simple_radiomics: bool,
    drop_correlated_features: bool,
) -> pd.DataFrame:
    if drop_correlated_features:
        # Identify and remove strongly correlated features in the training dataset
        x_dataframe, dropped_features = remove_correlated_features(
            x_dataframe, corr_threshold=0.8
        )
        feature_names = x_dataframe.columns.to_numpy()
        print("Number of remaining features:", len(feature_names))
    if only_simple_radiomics:
        x_dataframe = x_dataframe.drop(
            [column for column in x_dataframe if not column.startswith("original")],
            axis=1,
        )
    return x_dataframe


def load_train_test_splits(
    base_dir: Path, radiomics_file_name: str, redo_split: bool, test_size: float
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    radiomics_train_file_name = f"train_{radiomics_file_name}"
    radiomics_test_file_name = f"test_{radiomics_file_name}"
    full_dataframe = _load_data(Path(base_dir, radiomics_file_name))
    if redo_split:
        train_dataframe, test_dataframe = _make_train_test_split(
            full_dataframe, test_size, RANDOM_SEED
        )
        train_dataframe.to_csv(Path(base_dir, radiomics_train_file_name))
        test_dataframe.to_csv(Path(base_dir, radiomics_test_file_name))
    else:
        train_dataframe = _load_data(Path(base_dir, radiomics_train_file_name))
        test_dataframe = _load_data(Path(base_dir, radiomics_test_file_name))
    return test_dataframe, train_dataframe
