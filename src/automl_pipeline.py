from sklearn.model_selection import train_test_split

from src.preprocessing import DataPreprocessor
from src.feature_selection import FeatureSelector
from src.model_factory import ModelFactory
from src.trainer import ModelTrainer
from src.exporter import Exporter


class AutoMLPipeline:

    def __init__(self, df, target_column):

        self.df = df.copy()

        self.target_column = target_column

    def detect_problem_type(self):

        y = self.df[self.target_column]

        if (
            str(y.dtype) == "object"
            or str(y.dtype) == "category"
            or str(y.dtype) == "bool"
        ):

            return "classification"

        if y.nunique() <= 10:

            return "classification"

        return "regression"

    def run(self):

        print(
            "\nStarting AutoML Pipeline..."
        )

        # -------------------------
        # DETECT PROBLEM TYPE
        # -------------------------

        problem_type = (
            self.detect_problem_type()
        )

        print(
            f"\nProblem Type: {problem_type}"
        )

        # -------------------------
        # PREPROCESSING
        # -------------------------

        preprocessor = DataPreprocessor(
            self.df,
            self.target_column
        )

        df = preprocessor.preprocess()

        print(
            "\nColumns after preprocessing:"
        )

        print(
            df.columns.tolist()
        )

        # -------------------------
        # FEATURE SELECTION
        # -------------------------

        selector = FeatureSelector(
            df=df,
            target_column=self.target_column
        )

        selected_features = (
            selector.select_features()
        )

        if len(selected_features) == 0:

            raise ValueError(
                "No features selected. "
                "Try lowering the correlation threshold."
            )

        print(
            "\nSelected Features:"
        )

        print(
            selected_features
        )

        # -------------------------
        # SCALING
        # -------------------------

        df = preprocessor.scale_features(
            selected_features
        )

        x = df[selected_features]

        y = df[self.target_column]

        # -------------------------
        # TRAIN TEST SPLIT
        # -------------------------

        x_train, x_test, y_train, y_test = (
            train_test_split(
                x,
                y,
                test_size=0.20,
                random_state=42
            )
        )

        # -------------------------
        # LOAD MODELS
        # -------------------------

        if problem_type == "classification":

            models = (
                ModelFactory
                .get_classification_models()
            )

        else:

            models = (
                ModelFactory
                .get_regression_models()
            )

        # -------------------------
        # TRAIN MODELS
        # -------------------------

        trainer = ModelTrainer(
            models=models,
            problem_type=problem_type
        )

        (
            best_model,
            best_model_name,
            results
        ) = trainer.train(
            x_train,
            x_test,
            y_train,
            y_test
        )

        # -------------------------
        # RETRAIN BEST MODEL
        # -------------------------

        print(
            f"\nBest Model: "
            f"{best_model_name}"
        )

        best_model.fit(
            x,
            y
        )

        # -------------------------
        # EXPORT
        # -------------------------

        Exporter.save_model(
            best_model
        )

        Exporter.save_results(
            results
        )

        print(
            "\nPipeline Completed Successfully."
        )

        return {

            "best_model":
            best_model_name,

            "selected_features":
            selected_features,

            "results":
            results
        }