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


def get_numerical_categorical_features(data: pd.DataFrame):
    numerical_features = []
    categorical_features = []
    for i in data.columns:
        if data[i].dtypes == "float64":
            numerical_features.append(i)
        else:
            categorical_features.append(i)
    return numerical_features, categorical_features


def split_categorical_binary_features(data: pd.DataFrame):
    binary_features = []
    categorical_features = []
    for i in data.columns:
        if len(data[i].unique()) == 2:
            binary_features.append(i)
        else:
            categorical_features.append(i)
    return binary_features, categorical_features


def treat_numerical_data(data: pd.DataFrame, scaler: int):
    if scaler == 0:
        scaler_model = StandardScaler()
    elif scaler == 1:
        scaler_model = MinMaxScaler(feature_range=(0, 1))
    else:
        scaler_model = MinMaxScaler(feature_range=(-1, 1))

    mapper = DataFrameMapper([(data.columns, scaler_model)])
    scaled_features = mapper

    return scaled_features


def treat_categorical_data(data: pd.DataFrame, encoder):
    if encoder == 0:
        # labeled_features_df = pd.get_dummies(data=data, prefix=data.columns)
        treat_model = OneHotEncoder(sparse=False, drop="first")
    else:
        treat_model = LabelEncoder()
        # data2 = data.copy()
        # for i in data.columns:
        #     data2[i] = le.fit(data2[i])
        # labeled_features_df = data2

    return treat_model


def resample_data_stratified(data: pd.DataFrame, column: str):
    min_sample = min(data[column].value_counts())
    resample_data = (
        data.groupby(column, group_keys=False)
        .apply(lambda x: x.sample(min_sample))
        .reset_index(drop=True)
    )
    return resample_data


def inputation_categorical_data(data: pd.DataFrame, inputation_type: int):
    categorical_columns = data.columns.to_list()
    if inputation_type == 0:
        imp = SimpleImputer(strategy="most_frequent")
    elif inputation_type == 1:
        imp = SimpleImputer(strategy="constant", fill_value="Nao se aplica")
    return imp


def inputation_numerical_data(data: pd.DataFrame, inputation_type: int):
    if inputation_type == 0:
        imp = SimpleImputer(strategy="mean")
    elif inputation_type == 1:
        imp = SimpleImputer(strategy="median")
    elif inputation_type == 2:
        imp = SimpleImputer(strategy="most_frequent")
    if inputation_type == 3:
        imp = SimpleImputer(strategy="constant", fill_value=0)

    return imp


def remove_outliers(numerical_data: pd.DataFrame, column: str):

    q25, q75 = np.percentile(numerical_data[column], 25), np.percentile(
        numerical_data[column], 75
    )
    iqr = q75 - q25
    lim_inferior = q25 - 1.5 * iqr
    lim_superior = q75 + 1.5 * iqr
    numerical_data = numerical_data.loc[
        (numerical_data[column] > lim_inferior)
        and (numerical_data[column] > lim_superior),
        :,
    ]
    return numerical_data


class Feature_selector(BaseEstimator, TransformerMixin):
    def __init__(self, data_type="numerical"):
        self.data_type = data_type

    def fit(self, X: pd.DataFrame):
        (
            self.numerical_features,
            self.categorical_features,
        ) = get_numerical_categorical_features(X)
        return self

    def transform(self, X: pd.DataFrame):

        if self.data_type == "numerical":
            return X.loc[:, self.numerical_features]
        else:
            return X.loc[:, self.categorical_features]


class Preprocessing_initial_data(BaseEstimator, TransformerMixin):
    def __init__(
        self,
        remove_outlier=False,
        columns_with_outliers=[],
        resample=True,
        target_column=None,
    ):
        self.remove_outlier = remove_outlier
        self.columns_with_outliers = columns_with_outliers
        self.resample = resample
        self.target_column = target_column

    def fit(self, X):
        return self

    def transform(self, X):
        if self.resample:
            X = resample_data_stratified(X, self.target_column)
        if self.remove_outlier:
            for columns in self.columns_with_outliers:
                X = remove_outliers(X, columns)
        return X


class Categorical_tranformer(BaseEstimator, TransformerMixin):
    def __init__(
        self,
        inputation_categorical=True,
        inputation_type=0,
        treat_categorical=False,
        treat_type=0,
        colnames=[],
    ):
        self.inputation_categorical = inputation_categorical
        self.inputation_type = inputation_type
        self.treat_categorical = treat_categorical
        self.treat_type = treat_type
        self.colnames = colnames

    def get_feature_names_out(self):
        return self.colnames

    def fit(self, X, y=None):
        # X = Feature_selector(data_type="categorical").fit_transform(X)
        if self.inputation_categorical:
            self.categorical_inputation = inputation_categorical_data(
                X, self.inputation_type
            )
            for i in X.columns:
                X[i] = self.categorical_inputation.fit(X[i].values.reshape(-1, 1))

        if self.treat_categorical:
            self.categorical_treat = treat_categorical_data(X, self.treat_type)
            for i in X.columns:
                X[i] = self.categorical_treat.fit(X[i])

        self.colnames = X.columns
        return self

    def transform(self, X, y=None):
        # X = Feature_selector(data_type="categorical").fit_transform(X)
        if self.inputation_categorical:
            for i in X.columns:
                X[i] = self.categorical_inputation.fit_transform(
                    X[i].values.reshape(-1, 1)
                )
        if self.treat_categorical:
            for i in X.columns:
                X[i] = self.categorical_treat.fit_transform(X[i])

        return X


class Numerical_tranformer(BaseEstimator, TransformerMixin):
    def __init__(
        self,
        inputation_numerical=False,
        inputation_type=0,
        treat_numerical=True,
        treat_type=0,
        colnames=[],
    ):
        self.inputation_numerical = inputation_numerical
        self.inputation_type = inputation_type
        self.treat_numerical = treat_numerical
        self.treat_type = treat_type
        self.colnames = colnames

    def get_feature_names_out(self):
        return self.colnames

    def fit(self, X, y=None):
        # X = Feature_selector(data_type="numerical").fit_transform(X)
        if self.inputation_numerical:
            self.input_numerical = inputation_numerical_data(X, self.inputation_type)

        if self.treat_numerical:
            self.treat_numerical = treat_numerical_data(X, self.treat_type)
            self.treat_numerical.fit(X.copy())

        return self

    def transform(self, X, y=None):
        X = Feature_selector(data_type="numerical").fit_transform(X)
        if self.treat_numerical:
            scaled_features = self.treat_numerical.transform(X.copy())
            X = pd.DataFrame(scaled_features, index=X.index, columns=X.columns)

        if self.inputation_numerical:
            for i in X.columns:
                X[i] = self.inputation_numerical.transform(X[i])

        return X


class pipeline_preprocessing(BaseEstimator, TransformerMixin):
    def __init__(
        self,
        inputation_numerical=False,
        inputation_type=0,
        treat_numerical=True,
        treat_type=1,
        inputation_categorical=True,
        inputation_cat_type=0,
        treat_categorical=True,
        treat_cat_type=1,
    ):
        self.inputation_categorical = inputation_categorical
        self.inputation_type = inputation_type
        self.treat_categorical = treat_categorical
        self.treat_type = treat_type
        self.inputation_numerical = inputation_numerical
        self.inputation_cat_type = inputation_cat_type
        self.treat_numerical = treat_numerical
        self.treat_cat_type = treat_cat_type

    def fit(self, X: pd.DataFrame, y=None):
        numerical_features, categorical_features = get_numerical_categorical_features(X)

        transform_numerical = Numerical_tranformer(
            inputation_numerical=self.inputation_numerical,
            inputation_type=self.inputation_type,
            treat_numerical=self.treat_numerical,
            treat_type=self.treat_type,
            colnames=numerical_features,
        )
        self.numerical_pipeline = Pipeline(
            steps=[
                ("select numerical_data", Feature_selector(data_type="numerical")),
                ("transform numerical data", transform_numerical),
            ]
        )
        self.numerical_pipeline.fit(X)

        transform_categorical = Categorical_tranformer(
            inputation_categorical=self.inputation_categorical,
            inputation_type=self.inputation_cat_type,
            treat_categorical=self.treat_categorical,
            treat_type=self.treat_cat_type,
            colnames=categorical_features,
        )
        self.categorical_pipeline = Pipeline(
            steps=[
                (
                    "select categorical data",
                    Feature_selector(data_type="categorical"),
                ),
                ("transform categorical data", transform_categorical),
            ]
        )
        self.categorical_pipeline.fit(X)
        # if len(y) > 0:
        #     self.answer = LabelEncoder().fit(y.reshape(-1, 1))
        return self

    def transform(self, X: pd.DataFrame, y=None):
        pipeline_preprocessing = FeatureUnion(
            transformer_list=[
                ("numerical_pipeline", self.numerical_pipeline),
                ("categorical_pipeline", self.categorical_pipeline),
            ]
        )
        categorical_cols = list(
            pipeline_preprocessing.get_params()["categorical_pipeline"][
                1
            ].get_feature_names_out()
        )

        numerical_cols = list(
            pipeline_preprocessing.get_params()["numerical_pipeline"][
                1
            ].get_feature_names_out()
        )
        colnames = categorical_cols + numerical_cols
        X = pd.DataFrame(pipeline_preprocessing.transform(X), columns=colnames)
        # if len(y) > 0:
        #     y = pd.DataFrame(
        #         self.answer().transform(y.reshape(-1, 1)),
        #         columns=["HeartDisease"],
        #     )
        return X
