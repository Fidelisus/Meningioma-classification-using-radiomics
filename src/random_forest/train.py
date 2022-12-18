from pathlib import Path
from typing import Tuple, List

import pandas as pd

from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import roc_auc_score, accuracy_score, precision_score, recall_score
from sklearn.model_selection import train_test_split

from src.random_forest.training.remove_correlated_features import remove_correlated_features
from src.random_forest.analysis.visualize_random_forest import make_plots

TEST_SIZE = 0.2

RANDOM_SEED = 123


def load_data(path: Path):
    return pd.read_csv(path, index_col=0)


def _create_train_test_split(full_dataframe: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame]:
    return train_test_split(full_dataframe, test_size=TEST_SIZE, random_state=RANDOM_SEED,
                            stratify=full_dataframe["label"])


def _get_metadata_columns(dataframe: pd.DataFrame) -> List[str]:
    metadata_columns = ["image", "mask"]
    for column in dataframe.columns:
        if column.startswith("diagnostics_"):
            metadata_columns.append(column)
    return metadata_columns


def _internal_validation(rf: RandomForestClassifier, y_dataframe: pd.DataFrame, predictions_dataframe: pd.DataFrame):
    print(f"Accuracy: {accuracy_score(y_dataframe, predictions_dataframe)}")
    print(f"Precision: {precision_score(y_dataframe, predictions_dataframe, average='weighted')}")
    print(f"Recall: {recall_score(y_dataframe, predictions_dataframe, average='weighted')}")


def train(train_dataframe: pd.DataFrame, corr_threshold: float = 0.8) -> Tuple[RandomForestClassifier, pd.DataFrame]:
    train_dataframe_no_index = train_dataframe.reset_index(drop=True)
    x_dataframe = train_dataframe_no_index.drop(_get_metadata_columns(train_dataframe), axis=1).drop("label", axis=1)
    y_dataframe = train_dataframe_no_index["label"]

    # Identify and remove strongly correlated features in the training dataset
    x_dataframe_uncorrelated, dropped_features = remove_correlated_features(x_dataframe,
                                                                            corr_threshold=corr_threshold)

    feature_names = x_dataframe_uncorrelated.columns.to_numpy()
    print("Number of remaining features:", len(feature_names))

    rf = RandomForestClassifier(n_estimators=1000, oob_score=True, random_state=RANDOM_SEED)
    rf.fit(x_dataframe_uncorrelated, y_dataframe)
    predictions_dataframe = rf.predict(x_dataframe_uncorrelated)

    print("Train metrics:")
    _internal_validation(rf, y_dataframe, predictions_dataframe)

    return rf, x_dataframe_uncorrelated


def evaluate(rf: RandomForestClassifier, test_dataframe: pd.DataFrame, x_dataframe_uncorrelated: pd.DataFrame):
    x_test_dataframe_selected_features = test_dataframe[x_dataframe_uncorrelated.columns]
    y_test_dataframe = test_dataframe["label"]

    predictions_dataframe = rf.predict(x_test_dataframe_selected_features)

    print("Test metrics:")
    _internal_validation(rf, y_test_dataframe, predictions_dataframe)

    feature_importance = pd.DataFrame(
        {"feature_name": rf.feature_names_in_, "importance": rf.feature_importances_}).sort_values("importance",
                                                                                                   ascending=False)
    print(feature_importance.iloc[:10])


def main(do_split: bool = False):
    base_dir = Path("..", "..", "data")
    radiomics_file_name = "radiomics_features_2022_12_10_11_20_08.csv"
    radiomics_train_file_name = f"train_{radiomics_file_name}"
    radiomics_test_file_name = f"test_{radiomics_file_name}"

    full_dataframe = load_data(Path(base_dir, radiomics_file_name))

    # make_plots(full_dataframe.drop(_get_metadata_columns(full_dataframe), axis=1))

    if do_split:
        train_dataframe, test_dataframe = _create_train_test_split(full_dataframe)
        train_dataframe.to_csv(Path(base_dir, radiomics_train_file_name))
        test_dataframe.to_csv(Path(base_dir, radiomics_test_file_name))
    else:
        train_dataframe = load_data(Path(base_dir, radiomics_train_file_name))
        test_dataframe = load_data(Path(base_dir, radiomics_test_file_name))

    rf, x_dataframe_uncorrelated = train(train_dataframe)
    evaluate(rf, test_dataframe, x_dataframe_uncorrelated)


if __name__ == "__main__":
    main(do_split=True)
