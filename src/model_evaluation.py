import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os
import json
import pickle
import logging

from typing import Optional, Any

from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
)

from classification_evaluator import ClassificationEvaluator


# Logging configuration
logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)

if not logger.handlers:

    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging.DEBUG)

    file_handler = logging.FileHandler("error.log")
    file_handler.setLevel(logging.ERROR)

    formatter = logging.Formatter(
        "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    )

    console_handler.setFormatter(formatter)
    file_handler.setFormatter(formatter)

    logger.addHandler(console_handler)
    logger.addHandler(file_handler)


# Load model and scaler
def load_model_and_scaler(
    model_path: str,
    scaler_path: str
) -> tuple[
    Optional[Any],
    Optional[StandardScaler]
]:

    try:
        logger.info("Loading model and scaler...")

        with open(model_path, "rb") as f:
            model = pickle.load(f)

        with open(scaler_path, "rb") as f:
            scaler = pickle.load(f)

        logger.info(
            "Model and scaler loaded successfully."
        )

        return model, scaler

    except FileNotFoundError:
        logger.exception(
            "Model or scaler file not found."
        )

    except pickle.UnpicklingError:
        logger.exception(
            "Error loading pickle file."
        )

    except Exception:
        logger.exception(
            "Unexpected error while loading model/scaler."
        )

    return None, None


# Load validation data
def load_validation_data(
    val_path: str
) -> Optional[pd.DataFrame]:

    try:
        logger.info("Loading validation data...")

        val_df = pd.read_csv(val_path)

        logger.info(
            "Validation data loaded successfully."
        )

        return val_df

    except FileNotFoundError:
        logger.exception(
            "Validation file not found."
        )

    except pd.errors.EmptyDataError:
        logger.exception(
            "Validation CSV is empty."
        )

    except pd.errors.ParserError:
        logger.exception(
            "Error parsing validation CSV."
        )

    except Exception:
        logger.exception(
            "Unexpected error while loading validation data."
        )

    return None


# Prepare validation data
def prepare_validation_data(
    val_df: pd.DataFrame,
    scaler: StandardScaler
) -> tuple[
    Optional[np.ndarray],
    Optional[pd.Series]
]:

    try:
        logger.info(
            "Preparing validation data..."
        )

        if 'driver_acceptance' not in val_df.columns:
            raise KeyError(
                "'driver_acceptance' column missing."
            )

        X_val = val_df.drop(
            columns=['driver_acceptance']
        )

        y_val = val_df['driver_acceptance']

        X_val = scaler.transform(X_val)

        logger.info(
            "Validation data preparation complete."
        )

        return X_val, y_val

    except KeyError:
        logger.exception(
            "Required target column missing."
        )

    except ValueError:
        logger.exception(
            "Error during feature scaling."
        )

    except Exception:
        logger.exception(
            "Unexpected error during validation preparation."
        )

    return None, None


# Save evaluation report
def save_evaluation_report(
    evaluator: ClassificationEvaluator,
    y_val: pd.Series
) -> None:

    try:
        logger.info(
            "Saving evaluation report..."
        )

        report = {
            "accuracy": accuracy_score(
                y_val,
                evaluator.y_pred
            ),
            "precision": precision_score(
                y_val,
                evaluator.y_pred
            ),
            "recall": recall_score(
                y_val,
                evaluator.y_pred
            ),
            "f1_score": f1_score(
                y_val,
                evaluator.y_pred
            ),
            "roc_auc": roc_auc_score(
                y_val,
                evaluator.y_prob
            )
        }

        os.makedirs("reports", exist_ok=True)

        output_path = os.path.join(
            "reports",
            "evaluation_report.json"
        )

        with open(output_path, "w") as f:
            json.dump(report, f, indent=4)

        logger.info(
            "Evaluation report saved successfully."
        )

    except PermissionError:
        logger.exception(
            "Permission denied while saving report."
        )

    except TypeError:
        logger.exception(
            "Invalid data type in evaluation report."
        )

    except Exception:
        logger.exception(
            "Unexpected error while saving evaluation report."
        )


def main() -> None:

    try:

        model, scaler = load_model_and_scaler(
            "models/lightgbm_model.pkl",
            "models/scaler.pkl"
        )

        if model is None or scaler is None:
            logger.error(
                "Model/scaler loading failed."
            )
            return

        val_df = load_validation_data(
            "data/feature/test.csv"
        )

        if val_df is None:
            logger.error(
                "Validation data loading failed."
            )
            return

        X_val, y_val = prepare_validation_data(
            val_df,
            scaler
        )

        if X_val is None or y_val is None:
            logger.error(
                "Validation data preparation failed."
            )
            return

        logger.info(
            "Initializing evaluator..."
        )

        evaluator = ClassificationEvaluator(
            model,
            X_val,
            y_val
        )

        logger.info(
            "Generating evaluation metrics..."
        )

        evaluator.basic_metrics()
        evaluator.classification_report()
        evaluator.confusion_matrix_plot()
        evaluator.threshold_analysis()

        save_evaluation_report(
            evaluator,
            y_val
        )

        logger.info(
            "Model evaluation pipeline completed successfully."
        )

    except Exception:
        logger.exception(
            "Unexpected error in evaluation pipeline."
        )


if __name__ == "__main__":
    main()