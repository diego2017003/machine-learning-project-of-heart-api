from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder

from source.mlops.modules.preprocessing import *


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
