from typing import Optional, List

import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score
from sklearn.preprocessing import StandardScaler


def do_scaling(x_dataframe: pd.DataFrame, standard_scaler: Optional[StandardScaler] = None):
    if standard_scaler is None:
        standard_scaler = StandardScaler().fit(x_dataframe)
    x_dataframe_scaled = standard_scaler.transform(x_dataframe)
    return pd.DataFrame(x_dataframe_scaled, index=x_dataframe.index, columns=x_dataframe.columns), standard_scaler


def _internal_validation(rf: RandomForestClassifier, y_dataframe: pd.DataFrame, predictions_dataframe: pd.DataFrame):
    print(f"Accuracy: {accuracy_score(y_dataframe, predictions_dataframe)}")
    print(f"Precision: {precision_score(y_dataframe, predictions_dataframe, average='weighted')}")
    print(f"Recall: {recall_score(y_dataframe, predictions_dataframe, average='weighted')}")


def get_metadata_columns(dataframe: pd.DataFrame) -> List[str]:
    metadata_columns = ["image", "mask"]
    for column in dataframe.columns:
        if column.startswith("diagnostics_"):
            metadata_columns.append(column)
    return metadata_columns
