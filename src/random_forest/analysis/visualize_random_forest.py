import pandas as pd
import seaborn as sns
from matplotlib import pyplot as plt

from src.random_forest.training.remove_correlated_features import remove_correlated_features


def make_plots(full_dataframe):
    full_dataframe = full_dataframe.reset_index(drop=True).sort_values("label")
    uncorrelated_dataframe, _ = remove_correlated_features(full_dataframe)

    pp = sns.clustermap(uncorrelated_dataframe, figsize=(40, 40))  # z_score=1
    _ = plt.setp(pp.ax_heatmap.get_yticklabels(), rotation=0)

    plt.show()
