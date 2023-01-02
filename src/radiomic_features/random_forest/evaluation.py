import pandas as pd
from matplotlib import pyplot as plt
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import RocCurveDisplay
from sklearn.model_selection import GridSearchCV
from sklearn.preprocessing import StandardScaler

from random_forest.analysis.visualize_random_forest import _plot_feature_importance
from random_forest.utils import do_scaling, _internal_validation


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
