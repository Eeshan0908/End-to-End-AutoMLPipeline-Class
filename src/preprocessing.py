import pandas as pd

from sklearn.impute import SimpleImputer
from sklearn.preprocessing import (
    LabelEncoder,
    StandardScaler
)


class DataPreprocessor:

    def __init__(self, df, target_column):
        self.df = df.copy()
        self.target_column = target_column
        self.encoders = {}
        self.scaler = StandardScaler()

    def separate_columns(self):

        numerical_cols = self.df.select_dtypes(
            include=["int64", "float64"]
        ).columns.tolist()

        categorical_cols = self.df.select_dtypes(
            include=["object", "category"]
        ).columns.tolist()

        if self.target_column in numerical_cols:
            numerical_cols.remove(self.target_column)

        if self.target_column in categorical_cols:
            categorical_cols.remove(self.target_column)

        return numerical_cols, categorical_cols

    def fill_missing_values(self):

        numerical_cols, categorical_cols = self.separate_columns()

        if numerical_cols:

            numerical_imputer = SimpleImputer(
                strategy="median"
            )

            self.df[numerical_cols] = numerical_imputer.fit_transform(
                self.df[numerical_cols]
            )

        if categorical_cols:

            categorical_imputer = SimpleImputer(
                strategy="most_frequent"
            )

            self.df[categorical_cols] = categorical_imputer.fit_transform(
                self.df[categorical_cols]
            )

        return self.df

    def encode_categorical_columns(self):

        _, categorical_cols = self.separate_columns()

        for column in categorical_cols:

            encoder = LabelEncoder()

            self.df[column] = encoder.fit_transform(
                self.df[column].astype(str)
            )

            self.encoders[column] = encoder

        return self.df

    def encode_target(self):

        if self.df[self.target_column].dtype == "object":

            encoder = LabelEncoder()

            self.df[self.target_column] = encoder.fit_transform(
                self.df[self.target_column].astype(str)
            )

            self.encoders[self.target_column] = encoder

        return self.df

    def scale_features(self, selected_features):

        self.df[selected_features] = self.scaler.fit_transform(
            self.df[selected_features]
        )

        return self.df

    def get_scaler(self):
        return self.scaler

    def preprocess(self):

        self.fill_missing_values()

        self.encode_categorical_columns()

        self.encode_target()

        return self.df