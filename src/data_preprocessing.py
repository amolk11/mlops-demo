import numpy as np
import pandas as pd
import os
import logging

from typing import Optional
from missing_value_imputer import BookingValueImputer

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


# Load the data
def load_data(
    train_path: str,
    test_path: str
) -> tuple[
    Optional[pd.DataFrame],
    Optional[pd.DataFrame]
]:

    try:
        logger.debug("Loading data...")

        train_df = pd.read_csv(train_path)
        test_df = pd.read_csv(test_path)

        logger.debug("Data loaded successfully.")

        return train_df, test_df

    except FileNotFoundError:
        logger.exception("File not found.")

    except pd.errors.EmptyDataError:
        logger.exception("CSV file is empty.")

    except pd.errors.ParserError:
        logger.exception("Error parsing CSV file.")

    except Exception:
        logger.exception("Unexpected error while loading data.")

    return None, None


# Converting Datetime to datetime format
def convert_datetime(
    train_df: pd.DataFrame,
    test_df: pd.DataFrame
) -> tuple[
    Optional[pd.DataFrame],
    Optional[pd.DataFrame]
]:

    try:
        logger.debug("Converting Datetime to datetime format...")

        train_df['Datetime'] = pd.to_datetime(
            train_df['Datetime'],
            errors='coerce'
        )

        test_df['Datetime'] = pd.to_datetime(
            test_df['Datetime'],
            errors='coerce'
        )

        cols = ['Datetime'] + [
            col for col in train_df.columns
            if col != 'Datetime'
        ]

        train_df = train_df[cols]
        test_df = test_df[cols]

        logger.debug("Datetime conversion complete.")

        return train_df, test_df

    except KeyError:
        logger.exception("Datetime column not found.")

    except Exception:
        logger.exception("Unexpected error during datetime conversion.")

    return None, None


# Target creation and feature selection
def create_target_and_select_features(
    train_df: pd.DataFrame,
    test_df: pd.DataFrame
) -> tuple[
    Optional[pd.DataFrame],
    Optional[pd.DataFrame]
]:

    try:
        logger.info("Selecting features...")

        features = [
            'Datetime',
            'Vehicle Type',
            'Pickup Location',
            'Drop Location',
            'Booking Value',
        ]

        required_cols = features + ['Booking Status']

        missing_train_cols = set(required_cols) - set(train_df.columns)
        missing_test_cols = set(required_cols) - set(test_df.columns)

        if missing_train_cols:
            raise KeyError(
                f"Missing columns in train data: {missing_train_cols}"
            )

        if missing_test_cols:
            raise KeyError(
                f"Missing columns in test data: {missing_test_cols}"
            )

        train_df = train_df[required_cols].copy()
        test_df = test_df[required_cols].copy()

        logger.debug("Feature selection complete.")

        logger.info(
            "Converting Booking Status to driver acceptance..."
        )

        invalid_statuses = [
            'Cancelled by Customer',
            'Incomplete'
        ]

        train_df = train_df[
            ~train_df['Booking Status'].isin(invalid_statuses)
        ].copy()

        test_df = test_df[
            ~test_df['Booking Status'].isin(invalid_statuses)
        ].copy()

        train_df['driver_acceptance'] = (
            train_df['Booking Status'] != 'No Driver Found'
        ).astype(int)

        test_df['driver_acceptance'] = (
            test_df['Booking Status'] != 'No Driver Found'
        ).astype(int)

        train_df.drop(
            columns=['Booking Status'],
            inplace=True
        )

        test_df.drop(
            columns=['Booking Status'],
            inplace=True
        )

        logger.info("Booking Status conversion complete.")

        return train_df, test_df

    except KeyError:
        logger.exception("Required columns missing.")

    except Exception:
        logger.exception(
            "Unexpected error during feature engineering."
        )

    return None, None


# Handling missing values
def handle_missing_values(
    train_df: pd.DataFrame,
    test_df: pd.DataFrame
) -> tuple[
    Optional[pd.DataFrame],
    Optional[pd.DataFrame]
]:

    try:
        logger.info("Handling missing values...")

        imputer = BookingValueImputer()

        # Fit ONLY on train
        train_df = imputer.fit_transform(train_df)

        # Apply SAME learned stats on test
        test_df = imputer.transform(test_df)

        logger.info("Missing value imputation complete.")

        return train_df, test_df

    except ValueError:
        logger.exception("Invalid values found during imputation.")

    except Exception:
        logger.exception(
            "Unexpected error during missing value handling."
        )

    return None, None


# Save the preprocessed data into processed folder
def save_preprocessed_data(
    train_df: pd.DataFrame,
    test_df: pd.DataFrame
) -> None:

    try:
        logger.info(
            "Saving preprocessed data into processed folder..."
        )

        data_path = os.path.join("data", "processed")

        os.makedirs(data_path, exist_ok=True)

        train_df.to_csv(
            os.path.join(data_path, "train.csv"),
            index=False
        )

        test_df.to_csv(
            os.path.join(data_path, "test.csv"),
            index=False
        )

        logger.info(
            "Preprocessed data saved successfully."
        )

    except PermissionError:
        logger.exception("Permission denied while saving files.")

    except Exception:
        logger.exception(
            "Unexpected error while saving processed data."
        )


def main() -> None:

    try:
        train_df, test_df = load_data(
            "data/raw/train.csv",
            "data/raw/test.csv"
        )

        if train_df is None or test_df is None:
            logger.error("Data loading failed.")
            return

        train_df, test_df = convert_datetime(
            train_df,
            test_df
        )

        if train_df is None or test_df is None:
            logger.error("Datetime conversion failed.")
            return

        train_df, test_df = create_target_and_select_features(
            train_df,
            test_df
        )

        if train_df is None or test_df is None:
            logger.error("Feature engineering failed.")
            return

        train_df, test_df = handle_missing_values(
            train_df,
            test_df
        )

        if train_df is None or test_df is None:
            logger.error("Missing value handling failed.")
            return

        save_preprocessed_data(train_df, test_df)

    except Exception:
        logger.exception("Unexpected error in main pipeline.")


if __name__ == "__main__":
    main()