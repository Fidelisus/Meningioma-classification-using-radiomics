import pandas as pd
from matplotlib import pyplot as plt
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import RocCurveDisplay, f1_score, precision_score, recall_score
from sklearn.model_selection import GridSearchCV

from meningioma_random_forest.data_analysis.visualize_random_forest import (
    plot_feature_importance,
    plot_confusion_matrix,
)
from meningioma_random_forest.training.cross_validation_helper import (
    get_cross_validation_results_dataframe,
)


def _plot_roc_curve(
    rf: RandomForestClassifier,
    y_test_dataframe: pd.DataFrame,
    x_dataframe: pd.DataFrame,
):
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


def evaluate_best_model(
    grid_search_estimator: GridSearchCV,
    test_dataframe: pd.DataFrame,
    x_dataframe: pd.DataFrame,
):
    x_test_dataframe_selected_features = test_dataframe[x_dataframe.columns]
    y_test_dataframe = test_dataframe["label"]

    predictions_dataframe = grid_search_estimator.predict(
        x_test_dataframe_selected_features
    )

    cv_results = get_cross_validation_results_dataframe(grid_search_estimator)
    print(cv_results)

    print("Test metrics:")
    plot_confusion_matrix(y_test_dataframe, predictions_dataframe)
    internal_validation(y_test_dataframe, predictions_dataframe)
    plot_feature_importance(
        grid_search_estimator.best_estimator_, x_test_dataframe_selected_features
    )

    # _plot_roc_curve(
    #     grid_search_estimator.best_estimator_, y_test_dataframe, x_test_dataframe_selected_features
    # )

    # predictions_proba = grid_search_estimator.predict_proba(x_test_dataframe_selected_features)
    # predict_proba_df = (
    #     pd.DataFrame(data=predictions_proba, columns=("1", "2", "3"))
    #     .join(y_test_dataframe.reset_index(drop=True) / 3.0)
    #     .join(pd.DataFrame(data=predictions_dataframe, columns=["prediction"]))
    #     .sort_values("label")
    # ).reset_index(drop=True)
    # predict_proba_df["prediction"] = predict_proba_df["prediction"] / 3.0
    # import plotly.express as px
    #
    # fig = px.imshow(
    #     predict_proba_df,
    #     # color_continuous_scale="RdBu_r",
    #     # y=label.to_list()
    # )
    # fig.write_html("predict_proba_new_radiomics.html")


def internal_validation(
    y_dataframe: pd.DataFrame,
    predictions_dataframe: pd.DataFrame,
):
    print(
        f"F1 score: {f1_score(y_dataframe, predictions_dataframe, average='weighted')}"
    )
    print(
        f"Precision: {precision_score(y_dataframe, predictions_dataframe, average='weighted')}"
    )
    print(
        f"Recall: {recall_score(y_dataframe, predictions_dataframe, average='weighted')}"
    )
