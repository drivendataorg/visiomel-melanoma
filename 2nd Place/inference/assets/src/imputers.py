# Third-party libraries
import pandas as pd


__all__ = ['ulceration_breslow_from_relapse_imputer']

def ulceration_breslow_from_relapse_imputer(dataframe):
    """Impute ulceration and breslow from relapse.
    
    Args:
        dataframe (pandas.DataFrame): Dataframe to impute.
    
    Returns:
        pandas.DataFrame: Imputed dataframe.
    """

    for row in dataframe.itertuples():
        if pd.isna(row.breslow):
            if row.relapse == 1:
                dataframe.at[row.Index, "breslow"] = ">=4"
            else:
                dataframe.at[row.Index, "breslow"] = "<0.8"

        if pd.isna(row.ulceration):
            if row.relapse == 1:
                dataframe.at[row.Index, "ulceration"] = "YES"
            else:
                dataframe.at[row.Index, "ulceration"] = "NO"

    return dataframe