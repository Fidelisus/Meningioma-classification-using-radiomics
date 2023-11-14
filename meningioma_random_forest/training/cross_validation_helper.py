import copy

import pandas as pd
from sklearn import decomposition
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV, StratifiedKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler


def get_cross_validation_results_dataframe(
    grid_search_estimator: GridSearchCV,
) -> pd.DataFrame:
    grid_search_estimator_df = (
        pd.DataFrame(grid_search_estimator.cv_results_)
        .sort_values("rank_test_score")
        .reset_index(drop=True)
    )
    grid_search_estimator_df = grid_search_estimator_df.drop(
        [
            "mean_fit_time",
            "std_fit_time",
            "mean_score_time",
            "std_score_time",
            "params",
            "split0_test_score",
            "split1_test_score",
            "split2_test_score",
            "std_test_score",
        ],
        axis=1,
    )
    return grid_search_estimator_df


def get_predictions_on_validation_set(
    grid_search_estimator: GridSearchCV,
    y_dataframe: pd.DataFrame,
    x_dataframe: pd.DataFrame,
    random_seed: int,
):
    model = copy.deepcopy(grid_search_estimator.best_estimator_)
    predictions = []
    true_values = []
    for train_index, test_index in StratifiedKFold(
        5, random_state=random_seed, shuffle=True
    ).split(x_dataframe, y_dataframe):
        model.fit(x_dataframe.iloc[train_index], y_dataframe.iloc[train_index])
        predictions.append(
            pd.DataFrame(data=model.predict(x_dataframe.iloc[test_index]))
        )
        true_values.append(y_dataframe.iloc[test_index])
    return pd.concat(true_values), pd.concat(predictions)


def create_grid_search_estimator(random_seed: int) -> GridSearchCV:
    n_estimators = [10000]  # [1000, 10000]
    n_components = [20, 30, 50]
    pca = decomposition.PCA()
    scaler = StandardScaler()

    param_dist = [
        # {
        #     "classifier__n_estimators": n_estimators,
        #     "preprocessor": [pca],
        #     "preprocessor__n_components": n_components,
        #     "classifier__random_state": [random_seed],
        # },
        {
            "classifier__n_estimators": n_estimators,
            "preprocessor": [None],
            "classifier__random_state": [random_seed],
        },
    ]
    rf = RandomForestClassifier()
    pipe = Pipeline(
        steps=[("scaler", scaler), ("preprocessor", pca), ("classifier", rf)]
    )
    grid_search_estimator = GridSearchCV(
        pipe,
        param_dist,
        cv=StratifiedKFold(5, random_state=random_seed, shuffle=True),
        verbose=4,
        n_jobs=-1,
        scoring="f1_micro",
    )

    return grid_search_estimator
