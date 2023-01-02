import pandas as pd
import seaborn as sns
from matplotlib import pyplot as plt
from sklearn.ensemble import RandomForestClassifier

from random_forest.training.remove_correlated_features import remove_correlated_features


def make_plots(full_dataframe):
    # TODO refactor
    full_dataframe = full_dataframe.reset_index(drop=True).sort_values("label")
    uncorrelated_dataframe, _ = remove_correlated_features(full_dataframe)

    pp = sns.clustermap(uncorrelated_dataframe, figsize=(40, 40))  # z_score=1
    _ = plt.setp(pp.ax_heatmap.get_yticklabels(), rotation=0)

    plt.show()


def _plot_feature_importance(rf: RandomForestClassifier, test_dataframe: pd.DataFrame):
    # TODO clean - I copied it from web
    feats = {}
    for feature, importance in zip(test_dataframe.columns, rf.feature_importances_):
        feats[feature] = importance
    importances = pd.DataFrame.from_dict(feats, orient='index').rename(columns={0: 'Gini-Importance'})
    importances = importances.sort_values(by='Gini-Importance', ascending=False)
    importances = importances.reset_index()
    importances = importances.rename(columns={'index': 'Features'})
    sns.set(font_scale=5)
    sns.set(style="whitegrid", color_codes=True, font_scale=1.7)
    fig, ax = plt.subplots()
    fig.set_size_inches(30, 15)
    sns.barplot(x=importances['Gini-Importance'], y=importances['Features'], data=importances, color='skyblue')
    plt.xlabel('Importance', fontsize=25, weight='bold')
    plt.ylabel('Features', fontsize=25, weight='bold')
    plt.title('Feature Importance', fontsize=25, weight='bold')
    plt.show()
    print(importances)
