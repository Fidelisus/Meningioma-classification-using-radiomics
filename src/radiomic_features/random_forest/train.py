from pathlib import Path
from typing import Tuple, List, Optional
import joblib

import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
import seaborn as sns

from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import roc_auc_score, accuracy_score, precision_score, recall_score, RocCurveDisplay
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import StandardScaler

from random_forest.training.remove_correlated_features import remove_correlated_features
from random_forest.analysis.visualize_random_forest import make_plots, _plot_feature_importance

TEST_SIZE = 0.2

RANDOM_SEED = 123


def load_data(path: Path):
    return pd.read_csv(path, index_col=0)


def _create_train_test_split(full_dataframe: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame]:
    return train_test_split(full_dataframe, test_size=TEST_SIZE, random_state=RANDOM_SEED,
                            stratify=full_dataframe["label"])


def get_metadata_columns(dataframe: pd.DataFrame) -> List[str]:
    metadata_columns = ["image", "mask"]
    for column in dataframe.columns:
        if column.startswith("diagnostics_"):
            metadata_columns.append(column)
    return metadata_columns


def _internal_validation(rf: RandomForestClassifier, y_dataframe: pd.DataFrame, predictions_dataframe: pd.DataFrame):
    print(f"Accuracy: {accuracy_score(y_dataframe, predictions_dataframe)}")
    print(f"Precision: {precision_score(y_dataframe, predictions_dataframe, average='weighted')}")
    print(f"Recall: {recall_score(y_dataframe, predictions_dataframe, average='weighted')}")


def train(train_dataframe: pd.DataFrame, corr_threshold: float = 0.8, delete_correlated_features: bool = True) -> Tuple[
    RandomForestClassifier, pd.DataFrame, Optional[StandardScaler]]:
    train_dataframe_no_index = train_dataframe.reset_index(drop=True)
    x_dataframe = train_dataframe_no_index.drop(get_metadata_columns(train_dataframe), axis=1).drop("label", axis=1)
    y_dataframe = train_dataframe_no_index["label"]

    x_dataframe_cleaned, standard_scaler = do_preprocessing(corr_threshold, delete_correlated_features, x_dataframe)

    grid_search_estimator = _create_grid_search()
    # grid_search_estimator.fit(x_dataframe_cleaned, y_dataframe)
    # joblib.dump(grid_search_estimator, "grid_search.sav")

    grid_search_estimator = joblib.load("grid_search.sav")

    print(f"Best parameters: {grid_search_estimator.best_params_}")

    _get_cross_validation_results(grid_search_estimator)

    # rf = RandomForestClassifier(n_estimators=1000, oob_score=True, random_state=RANDOM_SEED)
    # rf.fit(x_dataframe_cleaned, y_dataframe)
    # predictions_dataframe = rf.predict(x_dataframe_cleaned)

    predictions_dataframe = grid_search_estimator.predict(x_dataframe_cleaned)
    print("Train metrics:")
    _internal_validation(grid_search_estimator.best_estimator_, y_dataframe, predictions_dataframe)

    return grid_search_estimator.best_estimator_, x_dataframe_cleaned, standard_scaler


def _create_grid_search():
    n_estimators = [int(x) for x in np.linspace(start=100, stop=10000, num=3)]
    max_features = ['log2', 'sqrt']
    max_depth = [int(x) for x in np.linspace(start=1, stop=50, num=3)]
    min_samples_split = [int(x) for x in np.linspace(start=2, stop=50, num=3)]
    min_samples_leaf = [int(x) for x in np.linspace(start=2, stop=50, num=3)]
    bootstrap = [True, False]
    param_dist = {'n_estimators': n_estimators,
                  # 'max_features': max_features,
                  'max_depth': max_depth,
                  # 'min_samples_split': min_samples_split,
                  # 'min_samples_leaf': min_samples_leaf,
                  # 'bootstrap': bootstrap,
                  'random_state': [RANDOM_SEED]}
    rf = RandomForestClassifier()
    rs = GridSearchCV(rf, param_dist, cv=3, verbose=2, n_jobs=-1, scoring='f1_micro')

    return rs


def _get_cross_validation_results(rs: GridSearchCV):
    rs_df = pd.DataFrame(rs.cv_results_).sort_values('rank_test_score').reset_index(drop=True)
    rs_df = rs_df.drop([
        'mean_fit_time',
        'std_fit_time',
        'mean_score_time',
        'std_score_time',
        'params',
        'split0_test_score',
        'split1_test_score',
        'split2_test_score',
        'std_test_score'],
        axis=1)
    print(rs_df.head(10))


def do_preprocessing(corr_threshold: float, delete_correlated_features: bool, x_dataframe: pd.DataFrame):
    if delete_correlated_features:
        # Identify and remove strongly correlated features in the training dataset
        x_dataframe, dropped_features = remove_correlated_features(x_dataframe, corr_threshold=corr_threshold)
        feature_names = x_dataframe.columns.to_numpy()
        print("Number of remaining features:", len(feature_names))
    x_dataframe_scaled, standard_scaler = do_scaling(x_dataframe)
    return x_dataframe_scaled, standard_scaler


def do_scaling(x_dataframe: pd.DataFrame, standard_scaler: Optional[StandardScaler] = None):
    if standard_scaler is None:
        standard_scaler = StandardScaler().fit(x_dataframe)
    x_dataframe_scaled = standard_scaler.transform(x_dataframe)
    return pd.DataFrame(x_dataframe_scaled, index=x_dataframe.index, columns=x_dataframe.columns), standard_scaler


def _plot_roc_curve(rf: RandomForestClassifier, y_test_dataframe: pd.DataFrame, x_dataframe: pd.DataFrame):
    from sklearn.preprocessing import LabelBinarizer

    y_score = rf.predict_proba(x_dataframe)

    label_binarizer = LabelBinarizer().fit(y_test_dataframe)
    y_onehot_test = label_binarizer.transform(y_test_dataframe)

    RocCurveDisplay.from_predictions(
        y_onehot_test.ravel(),
        y_score.ravel(),
        name="micro-average OvR",
        color="darkorange",
    )
    plt.axis("square")
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title("Micro-averaged One-vs-Rest")
    plt.legend()
    plt.show()


def evaluate(rf: RandomForestClassifier, test_dataframe: pd.DataFrame, x_dataframe: pd.DataFrame,
             standard_scaler: StandardScaler):
    x_test_dataframe_selected_features = test_dataframe[x_dataframe.columns]
    x_test_dataframe_scaled, standard_scaler = do_scaling(x_test_dataframe_selected_features, standard_scaler)
    y_test_dataframe = test_dataframe["label"]

    predictions_dataframe = rf.predict(x_test_dataframe_scaled)

    print("Test metrics:")
    _internal_validation(rf, y_test_dataframe, predictions_dataframe)
    _plot_feature_importance(rf, x_test_dataframe_scaled)
    _plot_roc_curve(rf, y_test_dataframe, x_test_dataframe_scaled)

    # feature_importance = pd.DataFrame(
    #     {"feature_name": rf.feature_names_in_, "importance": rf.feature_importances_}).sort_values("importance",
    #                                                                                                ascending=False)
    # print(feature_importance.iloc[:10])


def main(do_split: bool = False):
    base_dir = Path("../..", "..", "data")
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

    rf, x_dataframe_uncorrelated, standard_scaler = train(train_dataframe)
    evaluate(rf, test_dataframe, x_dataframe_uncorrelated, standard_scaler)


if __name__ == "__main__":
    main(do_split=True)
