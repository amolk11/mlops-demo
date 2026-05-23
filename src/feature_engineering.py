import numpy as np
import pandas as pd
import os
import logging

from typing import Optional
from sklearn.preprocessing import OrdinalEncoder

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


# Load data
def load_data(
    train_path: str,
    test_path: str
) -> tuple[
    Optional[pd.DataFrame],
    Optional[pd.DataFrame]
]:

    try:
        logger.info("Loading data...")

        train_df = pd.read_csv(train_path)
        test_df = pd.read_csv(test_path)

        logger.info("Data loaded successfully.")

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


# Convert Datetime
def convert_datetime(
    train_df: pd.DataFrame,
    test_df: pd.DataFrame
) -> tuple[
    Optional[pd.DataFrame],
    Optional[pd.DataFrame]
]:

    try:
        logger.info("Converting Datetime column...")

        train_df['Datetime'] = pd.to_datetime(
            train_df['Datetime'],
            errors='coerce'
        )

        test_df['Datetime'] = pd.to_datetime(
            test_df['Datetime'],
            errors='coerce'
        )

        logger.info("Datetime conversion complete.")

        return train_df, test_df

    except KeyError:
        logger.exception("Datetime column missing.")

    except Exception:
        logger.exception("Unexpected error during datetime conversion.")

    return None, None


# Time features
def create_time_features(
    train_df: pd.DataFrame,
    test_df: pd.DataFrame
) -> tuple[
    Optional[pd.DataFrame],
    Optional[pd.DataFrame]
]:

    try:
        logger.info("Creating time features...")

        train_df['day_of_week'] = train_df['Datetime'].dt.dayofweek
        test_df['day_of_week'] = test_df['Datetime'].dt.dayofweek

        train_df['month'] = train_df['Datetime'].dt.month
        test_df['month'] = test_df['Datetime'].dt.month

        train_df['is_weekend'] = (
            train_df['day_of_week'] >= 5
        ).astype(int)

        test_df['is_weekend'] = (
            test_df['day_of_week'] >= 5
        ).astype(int)

        train_df["is_peak_hour"] = (
            train_df['Datetime'].dt.hour.isin([7, 8, 9, 17, 18, 19])
        ).astype(int)

        test_df["is_peak_hour"] = (
            test_df['Datetime'].dt.hour.isin([7, 8, 9, 17, 18, 19])
        ).astype(int)

        train_df.drop(columns=['Datetime'], inplace=True)
        test_df.drop(columns=['Datetime'], inplace=True)

        logger.info("Time features created successfully.")

        return train_df, test_df

    except KeyError:
        logger.exception("Datetime column missing.")

    except Exception:
        logger.exception("Unexpected error during time feature creation.")

    return None, None


# Encode categorical features
def encode_categorical_features(
    train_df: pd.DataFrame,
    test_df: pd.DataFrame
) -> tuple[
    Optional[pd.DataFrame],
    Optional[pd.DataFrame]
]:

    try:
        logger.info("Encoding categorical features...")

        vehicle_price_order = [[
            'eBike',
            'Bike',
            'Auto',
            'Go Mini',
            'Go Sedan',
            'Premier Sedan',
            'Uber XL'
        ]]

        encoder = OrdinalEncoder(
            categories=vehicle_price_order
        )

        train_df[['Vehicle Type']] = encoder.fit_transform(
            train_df[['Vehicle Type']]
        )

        test_df[['Vehicle Type']] = encoder.transform(
            test_df[['Vehicle Type']]
        )

        logger.info("Categorical encoding complete.")

        return train_df, test_df

    except KeyError:
        logger.exception("Vehicle Type column missing.")

    except ValueError:
        logger.exception("Unexpected category found.")

    except Exception:
        logger.exception("Unexpected error during encoding.")

    return None, None


# Create location features
def create_location_features(
    train_df: pd.DataFrame,
    test_df: pd.DataFrame
) -> tuple[
    Optional[pd.DataFrame],
    Optional[pd.DataFrame]
]:

    try:
        logger.info("Creating location features...")

        pickup_freq = train_df['Pickup Location'].value_counts()

        train_df['pickup_freq'] = (
            train_df['Pickup Location'].map(pickup_freq)
        )

        test_df['pickup_freq'] = (
            test_df['Pickup Location'].map(pickup_freq).fillna(0)
        )

        drop_freq = train_df['Drop Location'].value_counts()

        train_df['drop_freq'] = (
            train_df['Drop Location'].map(drop_freq)
        )

        test_df['drop_freq'] = (
            test_df['Drop Location'].map(drop_freq).fillna(0)
        )

        logger.info("Location features created successfully.")

        return train_df, test_df

    except KeyError:
        logger.exception("Pickup/Drop Location columns missing.")

    except Exception:
        logger.exception("Unexpected error during location feature creation.")

    return None, None


# Route feature
def create_route_feature(
    train_df: pd.DataFrame,
    test_df: pd.DataFrame
) -> tuple[
    Optional[pd.DataFrame],
    Optional[pd.DataFrame]
]:

    try:
        logger.info("Creating route features...")

        train_df['route'] = (
            train_df['Pickup Location']
            + "_"
            + train_df['Drop Location']
        )

        test_df['route'] = (
            test_df['Pickup Location']
            + "_"
            + test_df['Drop Location']
        )

        route_avg = (
            train_df.groupby('route')['Booking Value'].mean()
        )

        train_df['route_avg_price'] = (
            train_df['route'].map(route_avg)
        )

        test_df['route_avg_price'] = (
            test_df['route'].map(route_avg).fillna(
                train_df['Booking Value'].mean()
            )
        )

        train_df['price_vs_route'] = (
            train_df['Booking Value']
            / train_df['route_avg_price']
        )

        test_df['price_vs_route'] = (
            test_df['Booking Value']
            / test_df['route_avg_price']
        )

        train_df.drop(
            columns=['route', 'Pickup Location', 'Drop Location'],
            inplace=True
        )

        test_df.drop(
            columns=['route', 'Pickup Location', 'Drop Location'],
            inplace=True
        )

        logger.info("Route features created successfully.")

        return train_df, test_df

    except KeyError:
        logger.exception("Required columns missing for route features.")

    except ZeroDivisionError:
        logger.exception("Division by zero encountered.")

    except Exception:
        logger.exception("Unexpected error during route feature creation.")

    return None, None


# Feature transformation
def feature_transformation(
    train_df: pd.DataFrame,
    test_df: pd.DataFrame
) -> tuple[
    Optional[pd.DataFrame],
    Optional[pd.DataFrame]
]:

    try:
        logger.info("Applying feature transformation...")

        train_df['log_price'] = np.log1p(train_df['Booking Value'])
        test_df['log_price'] = np.log1p(test_df['Booking Value'])

        train_df['log_route_avg'] = np.log1p(
            train_df['route_avg_price']
        )

        test_df['log_route_avg'] = np.log1p(
            test_df['route_avg_price']
        )

        train_df['is_high_value'] = (
            train_df['price_vs_route'] > 1
        ).astype(int)

        test_df['is_high_value'] = (
            test_df['price_vs_route'] > 1
        ).astype(int)

        logger.info("Feature transformation complete.")

        return train_df, test_df

    except Exception:
        logger.exception("Unexpected error during feature transformation.")

    return None, None


# Location popularity features
def create_location_popularity_features(
    train_df: pd.DataFrame,
    test_df: pd.DataFrame
) -> tuple[
    Optional[pd.DataFrame],
    Optional[pd.DataFrame]
]:

    try:
        logger.info("Creating location popularity features...")

        train_df['location_demand'] = (
            train_df['pickup_freq']
            + train_df['drop_freq']
        )

        train_df['location_interaction'] = (
            train_df['pickup_freq']
            * train_df['drop_freq']
        )

        test_df['location_demand'] = (
            test_df['pickup_freq']
            + test_df['drop_freq']
        )

        test_df['location_interaction'] = (
            test_df['pickup_freq']
            * test_df['drop_freq']
        )

        train_df.drop(
            columns=['pickup_freq', 'drop_freq'],
            inplace=True
        )

        test_df.drop(
            columns=['pickup_freq', 'drop_freq'],
            inplace=True
        )

        logger.info("Location popularity features created.")

        return train_df, test_df

    except Exception:
        logger.exception(
            "Unexpected error during location popularity feature creation."
        )

    return None, None


# Feature selection
def select_features(
    train_df: pd.DataFrame,
    test_df: pd.DataFrame
) -> tuple[
    Optional[pd.DataFrame],
    Optional[pd.DataFrame]
]:

    try:
        logger.info("Selecting features...")

        drop_cols = [
            'Booking Value',
            'route_avg_price',
            'location_interaction',
            'month'
        ]

        train_df = train_df.drop(columns=drop_cols)
        test_df = test_df.drop(columns=drop_cols)

        logger.info("Feature selection complete.")

        return train_df, test_df

    except KeyError:
        logger.exception("Columns missing during feature selection.")

    except Exception:
        logger.exception("Unexpected error during feature selection.")

    return None, None


# Save engineered data
def save_engineered_data(
    train_df: pd.DataFrame,
    test_df: pd.DataFrame
) -> None:

    try:
        logger.info("Saving engineered data...")

        data_path = os.path.join("data", "feature")

        os.makedirs(data_path, exist_ok=True)

        train_df.to_csv(
            os.path.join(data_path, "train.csv"),
            index=False
        )

        test_df.to_csv(
            os.path.join(data_path, "test.csv"),
            index=False
        )

        logger.info("Engineered data saved successfully.")

    except PermissionError:
        logger.exception("Permission denied while saving data.")

    except Exception:
        logger.exception("Unexpected error while saving data.")


def main() -> None:

    try:

        train_df, test_df = load_data(
            "data/processed/train.csv",
            "data/processed/test.csv"
        )

        if train_df is None or test_df is None:
            return

        train_df, test_df = convert_datetime(
            train_df,
            test_df
        )

        train_df, test_df = create_time_features(
            train_df,
            test_df
        )

        train_df, test_df = encode_categorical_features(
            train_df,
            test_df
        )

        train_df, test_df = create_location_features(
            train_df,
            test_df
        )

        train_df, test_df = create_route_feature(
            train_df,
            test_df
        )

        train_df, test_df = feature_transformation(
            train_df,
            test_df
        )

        train_df, test_df = create_location_popularity_features(
            train_df,
            test_df
        )

        train_df, test_df = select_features(
            train_df,
            test_df
        )

        save_engineered_data(
            train_df,
            test_df
        )

    except Exception:
        logger.exception("Unexpected error in feature engineering pipeline.")


if __name__ == "__main__":
    main()