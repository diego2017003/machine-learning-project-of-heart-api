import logging
from multiprocessing import Value
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
from sklearn_pandas import DataFrameMapper
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline, FeatureUnion
from sklearn.preprocessing import FunctionTransformer
from sklearn.base import BaseEstimator, TransformerMixin
import numpy as np

from preprocessing import *


class Target_encoder(BaseEstimator, TransformerMixin):
    def __init__(self) -> None:
        return None

    def fit(self, y):
        self.encoder = LabelEncoder()
        self.encoder.fit(y)

    def transform(self, y):
        return self.encoder.fit_transform(y)


class Model_pipeline(BaseEstimator, TransformerMixin):
    def __init__(self, model="decisionTreeClassifier"):
        self.model = model

    def fit(self, X, y):
        self.model_pipeline = Pipeline(
            steps=[
                ("preprocess", pipeline_preprocessing()),
                (
                    "train",
                    DecisionTreeClassifier(
                        criterion="gini", splitter="best", max_depth=4
                    ),
                ),
            ]
        )
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)
        self.model_pipeline.fit(X_train, y_train)
        return self

    def predict(self, X):
        return self.model_pipeline.predict(X)

    def transform(self, X, y):
        self.model_pipeline.transform(X, y)
        return self.model_pipeline
