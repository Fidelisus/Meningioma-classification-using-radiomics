import pandas as pd
import plotly.express as px
import seaborn as sns
from matplotlib import pyplot as plt
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import MinMaxScaler
from sklearn import metrics
import plotly.figure_factory as ff

from meningioma_random_forest.config import PLOTS_PATH
from meningioma_random_forest.data_loading.data_loader import remove_irrelevant_features


def plot_features_heatmap(full_dataframe: pd.DataFrame):
    label = full_dataframe["label"]
    x = remove_irrelevant_features(full_dataframe, True, False).reset_index(drop=True)
    scaler = MinMaxScaler()
    x_scaled = pd.DataFrame(
        data=scaler.fit_transform(x), columns=x.columns, index=x.index
    )
    x_sorted_by_label = (
        pd.concat([label, x_scaled], axis=1).sort_values("label").reset_index(drop=True)
    )
    x_sorted_by_label["label"] /= 3.0

    fig = px.imshow(
        x_sorted_by_label,
        color_continuous_scale="RdBu_r",
    )
    fig.write_html(PLOTS_PATH.joinpath("features_heatmap.html"))

    b = pd.concat([label, x_scaled], axis=1).groupby("label").mean()
    fig = px.imshow(
        b,
        color_continuous_scale="RdBu_r",
    )
    fig.write_html(PLOTS_PATH.joinpath("mean_of_features_for_each_label_heatmap.html"))


def plot_feature_importance(rf: Pipeline, test_dataframe: pd.DataFrame):
    features = {}
    for feature, importance in zip(
        test_dataframe.columns, rf.named_steps.get("classifier").feature_importances_
    ):
        features[feature] = importance

    importances = pd.DataFrame.from_dict(features, orient="index").rename(
        columns={0: "Gini-Importance"}
    )
    importances = importances.sort_values(by="Gini-Importance", ascending=False)
    importances = importances.reset_index()
    importances = importances.rename(columns={"index": "Features"})

    sns.set(font_scale=5)
    sns.set(style="whitegrid", color_codes=True, font_scale=1.7)
    fig, ax = plt.subplots()
    fig.set_size_inches(30, 15)

    sns.barplot(
        x=importances["Gini-Importance"],
        y=importances["Features"],
        data=importances,
        color="skyblue",
    )
    plt.xlabel("Importance", fontsize=25, weight="bold")
    plt.ylabel("Features", fontsize=25, weight="bold")
    plt.title("Feature Importance", fontsize=25, weight="bold")

    plt.savefig(PLOTS_PATH.joinpath(f"feature_importance.pdf"))


def plot_confusion_matrix(
    y_test_dataframe: pd.DataFrame,
    predictions_dataframe: pd.DataFrame,
    name: str = "test",
):
    confusion_matrix = metrics.confusion_matrix(y_test_dataframe, predictions_dataframe)
    confusion_matrix = confusion_matrix.astype(int)
    fig = ff.create_annotated_heatmap(
        confusion_matrix,
        x=["predicted 1", "predicted 2", "predicted 3"],
        y=["real 1", "real 2", "real 3"],
        colorscale="Viridis",
    )

    fig.update_layout(title_text="<i><b>Confusion matrix</b></i>")

    fig["data"][0]["showscale"] = True
    fig.write_html(PLOTS_PATH.joinpath(f"confusion_matrix_{name}.html"))
