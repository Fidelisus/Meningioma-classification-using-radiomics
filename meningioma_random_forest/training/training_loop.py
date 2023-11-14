from pathlib import Path
from typing import Tuple

import joblib
import pandas as pd
from sklearn.model_selection import GridSearchCV

from meningioma_random_forest.data_analysis.visualize_random_forest import (
    plot_confusion_matrix,
)
from meningioma_random_forest.data_loading.data_loader import (
    get_metadata_columns,
    remove_irrelevant_features,
)
from meningioma_random_forest.evaluation.evaluation import internal_validation
from meningioma_random_forest.training.cross_validation_helper import (
    create_grid_search_estimator,
    get_predictions_on_validation_set,
)


def training_loop(
    train_dataframe: pd.DataFrame,
    random_seed: int,
    only_unfiltered_radiomics: bool = True,
    drop_correlated_features: bool = False,
    path_to_save_model: Path = Path("..", "data", "grid_search.sav"),
) -> Tuple[GridSearchCV, pd.DataFrame]:
    train_dataframe_no_index = train_dataframe.reset_index(drop=True)
    x_dataframe = train_dataframe_no_index.drop(
        get_metadata_columns(train_dataframe), axis=1
    ).drop("label", axis=1)
    y_dataframe = train_dataframe_no_index["label"]

    x_dataframe_cleaned = remove_irrelevant_features(
        x_dataframe, only_unfiltered_radiomics, drop_correlated_features
    )

    grid_search_estimator = create_grid_search_estimator(random_seed)
    grid_search_estimator.fit(x_dataframe_cleaned, y_dataframe)
    joblib.dump(grid_search_estimator, str(path_to_save_model))

    grid_search_estimator = joblib.load(str(path_to_save_model))

    predictions_dataframe = grid_search_estimator.predict(x_dataframe_cleaned)

    print(f"Best parameters: {grid_search_estimator.best_params_}")

    print("Train metrics:")
    internal_validation(y_dataframe, predictions_dataframe)

    (
        y_validation_dataframe,
        validation_predictions_dataframe,
    ) = get_predictions_on_validation_set(
        grid_search_estimator, y_dataframe, x_dataframe, random_seed
    )
    plot_confusion_matrix(
        y_validation_dataframe, validation_predictions_dataframe, "train"
    )
    print("Validation metrics:")
    internal_validation(y_validation_dataframe, validation_predictions_dataframe)

    return grid_search_estimator, x_dataframe_cleaned
