import pandas as pd
import os
import logging
from typing import Optional

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


# Extract data from Google Drive
def load_data(
    train_url: str,
    test_url: str
) -> tuple[
    Optional[pd.DataFrame],
    Optional[pd.DataFrame]
]:

    try:
        logger.debug("Extracting data from Google Drive...")

        train_df = pd.read_csv(train_url)
        test_df = pd.read_csv(test_url)

        logger.debug("Data extraction complete.")

        return train_df, test_df

    except FileNotFoundError:
        logger.exception("File not found.")

    except pd.errors.EmptyDataError:
        logger.exception("CSV file is empty.")

    except pd.errors.ParserError:
        logger.exception("Unable to parse CSV file.")

    except Exception:
        logger.exception("Unexpected error while loading data.")

    return None, None


# Save data into raw folder
def save_data(
    train_df: pd.DataFrame,
    test_df: pd.DataFrame
) -> None:

    try:
        logger.debug("Saving data into raw folder...")

        data_path = os.path.join("data", "raw")

        os.makedirs(data_path, exist_ok=True)

        train_df.to_csv(
            os.path.join(data_path, "train.csv"),
            index=False
        )

        test_df.to_csv(
            os.path.join(data_path, "test.csv"),
            index=False
        )

        logger.debug("Data saved successfully.")

    except PermissionError:
        logger.exception("Permission denied while saving files.")

    except Exception:
        logger.exception("Unexpected error while saving data.")


def main() -> None:

    train_url = (
        "https://drive.google.com/uc?id=1ZKPqkUkP93coteNakxVKeEBpJsdIgxio"
    )

    test_url = (
        "https://drive.google.com/uc?id=1y6JBpCTDpOwgO1UkIvx3iCI-EPVK9tvr"
    )

    train_df, test_df = load_data(train_url, test_url)

    if train_df is not None and test_df is not None:
        save_data(train_df, test_df)
    else:
        logger.error("Data loading failed. Saving skipped.")


if __name__ == "__main__":
    main()