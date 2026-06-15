from sklearn.model_selection import (
    RandomizedSearchCV
)

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    r2_score,
    mean_absolute_error,
    mean_squared_error
)


class ModelTrainer:

    def __init__(
        self,
        models,
        problem_type
    ):

        self.models = models

        self.problem_type = problem_type

        self.results = []

        self.best_model = None

        self.best_model_name = None

        self.best_score = float("-inf")

    def train(
        self,
        x_train,
        x_test,
        y_train,
        y_test
    ):

        for model_name, model_info in self.models.items():

            model = model_info["model"]

            params = model_info["params"]

            print(
                f"\nTraining: {model_name}"
            )

            if params:

                scoring = (

                    "accuracy"

                    if self.problem_type
                    == "classification"

                    else

                    "r2"

                )

                search = RandomizedSearchCV(

                    estimator=model,

                    param_distributions=params,

                    n_iter=30,

                    cv=5,

                    scoring=scoring,

                    n_jobs=-1,

                    random_state=42

                )

                search.fit(
                    x_train,
                    y_train
                )

                trained_model = (
                    search.best_estimator_
                )

                best_params = (
                    search.best_params_
                )

            else:

                model.fit(
                    x_train,
                    y_train
                )

                trained_model = model

                best_params = {}

            predictions = (
                trained_model.predict(
                    x_test
                )
            )

            if (
                self.problem_type
                == "classification"
            ):

                score = accuracy_score(
                    y_test,
                    predictions
                )

                result = {

                    "model_name":
                    model_name,

                    "accuracy":
                    round(score, 4),

                    "precision":
                    round(
                        precision_score(
                            y_test,
                            predictions,
                            average="weighted",
                            zero_division=0
                        ),
                        4
                    ),

                    "recall":
                    round(
                        recall_score(
                            y_test,
                            predictions,
                            average="weighted",
                            zero_division=0
                        ),
                        4
                    ),

                    "f1_score":
                    round(
                        f1_score(
                            y_test,
                            predictions,
                            average="weighted",
                            zero_division=0
                        ),
                        4
                    ),

                    "best_params":
                    best_params
                }

            else:

                score = r2_score(
                    y_test,
                    predictions
                )

                result = {

                    "model_name":
                    model_name,

                    "r2_score":
                    round(score, 4),

                    "mae":
                    round(
                        mean_absolute_error(
                            y_test,
                            predictions
                        ),
                        4
                    ),

                    "rmse":
                    round(
                        mean_squared_error(
                            y_test,
                            predictions,
                            squared=False
                        ),
                        4
                    ),

                    "best_params":
                    best_params
                }

            self.results.append(
                result
            )

            if score > self.best_score:

                self.best_score = score

                self.best_model = (
                    trained_model
                )

                self.best_model_name = (
                    model_name
                )

        return (

            self.best_model,

            self.best_model_name,

            self.results

        )