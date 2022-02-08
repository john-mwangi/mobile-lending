import dill
import pandas as pd
from typing import Tuple


class Loader:
    def __init__(self) -> None:
        pass

    @staticmethod
    def load_models() -> tuple:
        """Load fitted models."""
        with open(file="./outputs/api_models.pkl", mode="rb") as f:
            model_objs = dill.load(file=f)

            dt_cv = model_objs.get("dt_cv")
            rf_cv = model_objs.get("rf_cv")
            gb_cv = model_objs.get("gb_cv")
            svm_cv = model_objs.get("svm_cv")
            knn_cv = model_objs.get("knn_cv")
            nn_cv = model_objs.get("nn_cv")

        return dt_cv, rf_cv, gb_cv, svm_cv, knn_cv, nn_cv

    @staticmethod
    def load_test_data() -> tuple:
        """Load test data and optimised threshold"""
        with open(file="./outputs/api_data.pkl", mode="rb") as f:
            data_objs = dill.load(file=f)

            train_X = data_objs.get("train_X")
            op_thr = data_objs.get("op_thr")

        return train_X, op_thr

    def obtain_preds(
        self, data: pd.DataFrame, op_thr: float
    ) -> Tuple[list, list]:
        """
        Retunrs predictions given input data. \n
        Args
        ----
        data: Data on which you want predictions.
        op_thr: Optimal probability of default.

        Returns
        ----
        default predictions: a list,
        adjusted predictions: a list
        """
        dt_cv, rf_cv, gb_cv, svm_cv, knn_cv, nn_cv = self.load_models()

        dt_probs = dt_cv.predict_proba(data)[:, 0]
        rf_probs = rf_cv.predict_proba(data)[:, 0]
        gb_probs = gb_cv.predict_proba(data)[:, 0]
        svm_probs = svm_cv.predict_proba(data)[:, 0]
        knn_probs = knn_cv.predict_proba(data)[:, 0]

        base_probs = [dt_probs, rf_probs, gb_probs, svm_probs, knn_probs]

        base_df = pd.DataFrame()

        for p in base_probs:
            temp_df = pd.DataFrame(p)
            base_df = pd.concat(objs=[base_df, temp_df], axis=1)

        base_df.columns = [
            "dt_probs",
            "rf_probs",
            "gb_probs",
            "svm_probs",
            "knn_probs",
        ]

        default_probs = nn_cv.predict_proba(base_df)[:, 0]
        default_preds = nn_cv.predict(base_df)
        default_preds = default_preds.tolist()

        adjusted_preds = [
            0 if default > (op_thr) else 1 for default in default_probs
        ]

        return default_preds, adjusted_preds
