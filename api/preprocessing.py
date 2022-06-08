import logging
from multiprocessing import Value
import pandas as pd
from sklearn import preprocessing
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
    """iterate into the data columns and analyse the column data type to
    select the numeric and the string objects in different lists

    Args:
        data (pd.DataFrame): the data to analyse the columns types

    Returns:
        tuple(list,list): two lists with the numerical and categorical columns
        respectively
    """
    numerical_features = []
    categorical_features = []
    for i in data.columns:
        if data[i].dtypes == "float64":
            numerical_features.append(i)
        else:
            categorical_features.append(i)
    return numerical_features, categorical_features


def split_categorical_binary_features(data: pd.DataFrame):
    """analyse the categorical data to select the size of unique values and split the columns
    into multilabel and binary columns

    Args:
        data (pd.DataFrame): categorical data to split the categories

    Returns:
        tuple(list,list): two lists with the binary and multilabel categorical columns
        respectively
    """
    binary_features = []
    categorical_features = []
    for i in data.columns:
        if len(data[i].unique()) == 2:
            binary_features.append(i)
        else:
            categorical_features.append(i)
    return binary_features, categorical_features


def treat_numerical_data(data: pd.DataFrame, scaler: int):
    """takes the data and one number to choose what is the best scaler for that data according
    to the user, and return with a mapper to the columns names cognization

    Args:
        data (pd.DataFrame): numerical data
        scaler (int): choosing the best scaler

    Returns:
        sklearn.preprocessing object : encoder
    """
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
    """takes the data and one number to choose what is the best scaler for that data according
    to the user, and return the categorical encoder with.

    Args:
        data (pd.DataFrame): categorical data
        scaler (int): choosing the best scaler

    Returns:
        sklearn.preprocessing object : encoder
    """
    if encoder == 0:
        treat_model = OneHotEncoder(sparse=False, drop="first")
    else:
        treat_model = LabelEncoder()

    return treat_model


def resample_data_stratified(data: pd.DataFrame, column: str):
    """get the dataset and applies a downsample operation to equals the number of values
    by categorie according to the column passed as argument

    Args:
        data (pd.DataFrame): full dataset
        column (str): column that guides the downsample

    Returns:
        pd.DataFrame: the data reduced by resample
    """
    min_sample = min(data[column].value_counts())
    resample_data = (
        data.groupby(column, group_keys=False)
        .apply(lambda x: x.sample(min_sample))
        .reset_index(drop=True)
    )
    return resample_data


def inputation_categorical_data(data: pd.DataFrame, inputation_type: int):
    """choice of the best inputation for the categorical data according to the user

    Args:
        data (pd.DataFrame): full dataset
        inputation_type (int): number of the inputation in the if and else

    Returns:
        pbject : inputer object
    """
    categorical_columns = data.columns.to_list()
    if inputation_type == 0:
        imp = SimpleImputer(strategy="most_frequent")
    elif inputation_type == 1:
        imp = SimpleImputer(strategy="constant", fill_value="Nao se aplica")
    return imp


def inputation_numerical_data(data: pd.DataFrame, inputation_type: int):
    """choice of the best inputation for the numerical data according to the user

    Args:
        data (pd.DataFrame): full dataset
        inputation_type (int): number of the inputation in the if and else

    Returns:
        pbject : inputer object
    """
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
    """Find the first and third quartil and remove the outliers of the numerical data according
    to the range of values produced by : lim_inferior = quartil1 - 1.5 * rangeinterquartil
    lim_superior = quartil3 - 1.5 * rangeinterquartil

    Args:
        numerical_data (pd.DataFrame): numerical dataset
        column (str): column to remove the outliers

    Returns:
        pd.DataFrame: dataset without outliers in the column passed as argument
    """
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
    """class to select and split the dataset between categorical and numerical

    Args:
        BaseEstimator (_type_): extends BaseEstimator
        TransformerMixin (_type_): extends TransformerMixin
    """

    def __init__(self, data_type="numerical"):
        """Feature_selector class builder"""
        self.data_type = data_type

    def fit(self, X: pd.DataFrame):
        """select the numerical and categorical columns"""
        (
            self.numerical_features,
            self.categorical_features,
        ) = get_numerical_categorical_features(X)
        return self

    def transform(self, X: pd.DataFrame):
        """split the dataset into numerical and categorical data

        Args:
            X (pd.DataFrame): dataset

        Returns:
            pd.DataFrame: dataset splitted to be numerical or categorical
        """

        if self.data_type == "numerical":
            return X.loc[:, self.numerical_features]
        else:
            return X.loc[:, self.categorical_features]


class Preprocessing_initial_data(BaseEstimator, TransformerMixin):
    """class to resample and remove the outliers, separated function to change
    the indexs of the full dataset
    """

    def __init__(
        self,
        remove_outlier=False,
        columns_with_outliers=[],
        resample=True,
        target_column=None,
    ):
        """Preprocessing_initial_data class builder, the user select the parameters to initial preprocessing

        Args:
            remove_outlier (bool, optional): if remove outliers or not. Defaults to False.
            columns_with_outliers (list, optional): columsn to remove the outliers. Defaults to [].
            resample (bool, optional): if you want to resample the dataset. Defaults to True.
            target_column (_type_, optional): column to base downsample. Defaults to None.
        """
        self.remove_outlier = remove_outlier
        self.columns_with_outliers = columns_with_outliers
        self.resample = resample
        self.target_column = target_column

    def fit(self, X):
        """nothing yet"""
        return self

    def transform(self, X):
        """resample and remove the data outliers depending to the user choices

        Args:
            X (pd.DataFrame): dataset

        Returns:
            _type_: dataset with dowsample, remove outliers or resample
        """
        if self.resample:
            X = resample_data_stratified(X, self.target_column)
        if self.remove_outlier:
            for columns in self.columns_with_outliers:
                X = remove_outliers(X, columns)
        return X


class Categorical_tranformer(BaseEstimator, TransformerMixin):
    """class to tranform the categorical data, applying the inputation and the encoders
    according to the user choice

    Args:
        BaseEstimator (_type_): extends BaseEstimator
        TransformerMixin (_type_): extends TransformerMixin
    """

    def __init__(
        self,
        inputation_categorical=True,
        inputation_type=0,
        treat_categorical=False,
        treat_type=0,
        colnames=[],
    ):
        """Categorical_tranformer class builder

        Args:
            inputation_categorical (bool, optional): if gonna applies an inputation. Defaults to True.
            inputation_type (int, optional): what is the inputation?. Defaults to 0.
            treat_categorical (bool, optional): if gonna applies an encoder. Defaults to False.
            treat_type (int, optional): what is the encoder?. Defaults to 0.
            colnames (list, optional): categorical columns to reference. Defaults to [].
        """
        self.inputation_categorical = inputation_categorical
        self.inputation_type = inputation_type
        self.treat_categorical = treat_categorical
        self.treat_type = treat_type
        self.colnames = colnames

    def get_feature_names_out(self):
        """returns the categorical columns passed to the class"""
        return self.colnames

    def fit(self, X, y=None):
        """fit the inputation and/or the encoder according to the user choice

        Args:
            X (pd.DataFrame): raw data
            y (pd.DataFrame, optional): target column in the dataset. Defaults to None.

        Returns:
            Categorical_tranformer: object with preprocessing model trained
        """
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
        """applies the transformation in the categorical data

        Args:
            X (pd.DataFrame): raw dataset
            y (pd.DataFrame, optional): target column. Defaults to None.

        Returns:
            pd.DataFrame: transformed data
        """
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
    """class to transform the numerical data

    Args:
        BaseEstimator (_type_): extends BaseEstimator
        TransformerMixin (_type_): extends TransformerMixin
    """

    def __init__(
        self,
        inputation_numerical=False,
        inputation_type=0,
        treat_numerical=True,
        treat_type=0,
        colnames=[],
    ):
        """Numerical_tranformer class builder

        Args:
            inputation_numerical (bool, optional): if gonna applies an inputation. Defaults to True.
            inputation_type (int, optional): what is the inputation?. Defaults to 0.
            treat_numerical (bool, optional): if gonna applies an encoder. Defaults to False.
            treat_type (int, optional): what is the encoder?. Defaults to 0.
            colnames (list, optional): categorical columns to reference. Defaults to [].
        """
        self.inputation_numerical = inputation_numerical
        self.inputation_type = inputation_type
        self.treat_numerical = treat_numerical
        self.treat_type = treat_type
        self.colnames = colnames

    def get_feature_names_out(self):
        """return the numerical dataset's columns"""
        return self.colnames

    def fit(self, X, y=None):
        """fit the inputation and/or the encoder according to the user choice

        Args:
            X (pd.DataFrame): raw data
            y (pd.DataFrame, optional): target column in the dataset. Defaults to None.

        Returns:
            Numerical_tranformer: object with preprocessing model trained
        """
        if self.inputation_numerical:
            self.input_numerical = inputation_numerical_data(X, self.inputation_type)

        if self.treat_numerical:
            self.treat_numerical = treat_numerical_data(X, self.treat_type)
            self.treat_numerical.fit(X.copy())

        return self

    def transform(self, X, y=None):
        """applies the transformation in the numerical data

        Args:
            X (pd.DataFrame): raw dataset
            y (pd.DataFrame, optional): target column. Defaults to None.

        Returns:
            pd.DataFrame: transformed data
        """
        X = Feature_selector(data_type="numerical").fit_transform(X)
        if self.treat_numerical:
            scaled_features = self.treat_numerical.transform(X.copy())
            X = pd.DataFrame(scaled_features, index=X.index, columns=X.columns)

        if self.inputation_numerical:
            for i in X.columns:
                X[i] = self.inputation_numerical.transform(X[i])

        return X


class pipeline_preprocessing(BaseEstimator, TransformerMixin):
    """union of the numerical transformer and the categorical transformer to create a pipeline object
    to process the data

    Args:
        BaseEstimator (_type_): extends BaseEstimator
        TransformerMixin (_type_): extends TransformerMixin
    """

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
        """pipeline_preprocessing class builder

        Args:
            inputation_numerical (bool, optional): if it gonna applies the inputation into numerical dataset
            .Defaults to False.
            inputation_type (int, optional): type of the inputation to numeric dataset. Defaults to 0.
            treat_numerical (bool, optional): if gonna encode numerical dataset. Defaults to True.
            treat_type (int, optional): whats is the numerical encoder. Defaults to 1.
            inputation_categorical (bool, optional): if it gonna applies the inputation into categorical
            dataset. Defaults to True.
            inputation_cat_type (int, optional):type of the inputation to categorical dataset. Defaults to 0.
            treat_categorical (bool, optional): if gonna encode categorical dataset. Defaults to True.
            treat_cat_type (int, optional): whats is the categorical encoder. Defaults to 1.
        """
        self.inputation_categorical = inputation_categorical
        self.inputation_type = inputation_type
        self.treat_categorical = treat_categorical
        self.treat_type = treat_type
        self.inputation_numerical = inputation_numerical
        self.inputation_cat_type = inputation_cat_type
        self.treat_numerical = treat_numerical
        self.treat_cat_type = treat_cat_type

    def fit(self, X: pd.DataFrame, y=None):
        """build categorical and numerical pipelines and applies an union between the
        2 pipelines

        Args:
            X (pd.DataFrame): raw dataset
            y (_type_, optional): target column. Defaults to None.

        Returns:
             preprocessing_pipeline : pipeline model trained
        """
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

        return self

    def transform(self, X: pd.DataFrame, y=None):
        """transform the raw data

        Args:
            X (pd.DataFrame): raw dataset
            y (_type_, optional): target column. Defaults to None.

        Returns:
           pd.DataFrame: trsnaformed data
        """
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

        return X
