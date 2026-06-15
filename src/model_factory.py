from sklearn.linear_model import (
    LinearRegression,
    LogisticRegression
)

from sklearn.tree import (
    DecisionTreeRegressor,
    DecisionTreeClassifier
)

from sklearn.ensemble import (
    RandomForestRegressor,
    RandomForestClassifier
)

from sklearn.neighbors import (
    KNeighborsRegressor,
    KNeighborsClassifier
)

from sklearn.naive_bayes import GaussianNB

from sklearn.svm import SVC


class ModelFactory:

    @staticmethod
    def get_regression_models():

        models = {

            "LinearRegression": {

                "model": LinearRegression(),

                "params": {}
            },

            "DecisionTreeRegressor": {

                "model": DecisionTreeRegressor(random_state=42),

                "params": {

                    "max_depth": [3, 5, 10, 15, 20, None],

                    "min_samples_split": [2, 5, 10, 20],

                    "min_samples_leaf": [1, 2, 4, 8]
                }
            },

            "RandomForestRegressor": {

                "model": RandomForestRegressor(random_state=42),

                "params": {

                    "n_estimators": [100, 200, 300, 500],

                    "max_depth": [5, 10, 20, None],

                    "min_samples_split": [2, 5, 10],

                    "min_samples_leaf": [1, 2, 4]
                }
            },

            "KNNRegressor": {

                "model": KNeighborsRegressor(),

                "params": {

                    "n_neighbors": [3, 5, 7, 9, 11, 15],

                    "weights": ["uniform","distance"],

                    "metric": ["euclidean","manhattan","minkowski"]
                }
            }
        }

        return models

    @staticmethod
    def get_classification_models():

        models = {

            "LogisticRegression": {

                "model": LogisticRegression(max_iter=5000),

                "params": {

                    "C": [
                        0.001,
                        0.01,
                        0.1,
                        1,
                        10,
                        100
                    ],

                    "solver": ["liblinear","lbfgs"]
                }
            },

            "DecisionTreeClassifier": {

                "model": DecisionTreeClassifier(random_state=42),

                "params": {

                    "max_depth": [3,5,10,15,20,None],

                    "min_samples_split": [2,5,10,20],

                    "min_samples_leaf": [1,2,4,8],

                    "criterion": ["gini","entropy"]
                }
            },

            "RandomForestClassifier": {

                "model": RandomForestClassifier(random_state=42),

                "params": {

                    "n_estimators": [100,200,300,500],

                    "max_depth": [5,10,20,None],

                    "min_samples_split": [2,5,10],

                    "min_samples_leaf": [1,2,4],

                    "criterion": ["gini","entropy"]
                }
            },

            "GaussianNB": {

                "model": GaussianNB(),

                "params": {

                    "var_smoothing": [

                        1e-12,
                        1e-11,
                        1e-10,
                        1e-9,
                        1e-8

                    ]
                }
            },

            "SVC": {

                "model": SVC(),

                "params": {

                    "C": [
                        0.001,
                        0.01,
                        0.1,
                        1,
                        10,
                        100
                    ],

                    "kernel": ["linear","rbf","poly"],

                    "gamma": ["scale","auto"]
                }
            },

            "KNNClassifier": {

                "model": KNeighborsClassifier(),

                "params": {

                    "n_neighbors": [3,5,7,9,11,15],

                    "weights": ["uniform","distance"],

                    "metric": ["euclidean","manhattan","minkowski"]
                }
            }
        }

        return models