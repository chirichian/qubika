import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer

def preprocessor(numeric_features:list)->ColumnTransformer:
    """Return preprocessor pipeline.

    Parameters
    ----------
    numeric_features : list
       list of features to process

    Returns
    -------
    ColumnTransformer
        pipeline to preprocess the data
    """
    numeric_transformer = Pipeline(
        steps=[("imputer", SimpleImputer(fill_value=0))]
    )

    preprocessor = ColumnTransformer(
        transformers=[
            ("num", numeric_transformer, numeric_features),
        ]
    )
    return preprocessor