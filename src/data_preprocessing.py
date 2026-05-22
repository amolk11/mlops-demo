import numpy as np
import pandas as pd
import os

# Load the data
print("Loading data...")
train_df = pd.read_csv("./data/raw/train.csv")
test_df = pd.read_csv("./data/raw/test.csv")
print("Data loaded successfully.")

# Converting Datetime to datetime format
print("Converting Datetime to datetime format...")
train_df['Datetime'] = pd.to_datetime(train_df['Datetime'])
test_df['Datetime'] = pd.to_datetime(test_df['Datetime'])
cols = ['Datetime'] + [col for col in train_df.columns if col != 'Datetime']
train_df = train_df[cols]
test_df = test_df[cols]
print("Datetime conversion complete.")

# feature selection
print("Selecting features...")
features = [
    'Datetime',
    'Vehicle Type',
    'Pickup Location',
    'Drop Location',
    'Booking Value',
]

train_df = train_df[features + ['Booking Status']]
test_df = test_df[features + ['Booking Status']]
print("Feature selection complete.")

# Converting Booking Status to driver acceptance (0 or 1)
print("Converting Booking Status to driver acceptance...") 
train_df = train_df[~train_df['Booking Status'].isin(['Cancelled by Customer','Incomplete'])]

train_df['driver_acceptance'] = train_df['Booking Status'].apply(lambda x: 0 if x == 'No Driver Found' else 1)

train_df.drop(columns=['Booking Status'], inplace=True)

test_df = test_df[~test_df['Booking Status'].isin(['Cancelled by Customer','Incomplete'])]

test_df['driver_acceptance'] = test_df['Booking Status'].apply(lambda x: 0 if x == 'No Driver Found' else 1)

test_df.drop(columns=['Booking Status'], inplace=True)
print("Booking Status conversion complete.")

# Handling missing values
print("Handling missing values...")
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin

class BookingValueImputer(BaseEstimator, TransformerMixin):
    def __init__(self):
        self.rv_median = None
        self.route_mean = None
        self.vehicle_scale = None
        self.v_median = None
        self.global_mean = None

    def fit(self, X, y=None):
        X = X.copy()

        # Create route
        X['route'] = X['Pickup Location'] + "_" + X['Drop Location']

        # Store global mean (final fallback)
        self.global_mean = X['Booking Value'].mean()

        # Route + Vehicle median
        self.rv_median = X.groupby(
            ['Vehicle Type', 'Pickup Location', 'Drop Location']
        )['Booking Value'].median()

        # Route mean
        self.route_mean = X.groupby('route')['Booking Value'].mean()

        # Vehicle median
        self.v_median = X.groupby('Vehicle Type')['Booking Value'].median()

        # Vehicle scale
        self.vehicle_scale = self.v_median / self.global_mean

        return self

    def transform(self, X):
        X = X.copy()
        print("Number of missing values before imputation:", X['Booking Value'].isnull().sum())
        print(f"Statistics of 'Booking Value' before imputation:\n{X['Booking Value'].describe()}")

        # Create route
        X['route'] = X['Pickup Location'] + "_" + X['Drop Location']

        # Merge mappings
        X = X.merge(self.rv_median.rename('rv_median'),
                    on=['Vehicle Type', 'Pickup Location', 'Drop Location'],
                    how='left')

        X = X.merge(self.route_mean.rename('route_mean'),
                    on='route',
                    how='left')

        X = X.merge(self.vehicle_scale.rename('vehicle_scale'),
                    on='Vehicle Type',
                    how='left')

        X = X.merge(self.v_median.rename('v_median'),
                    on='Vehicle Type',
                    how='left')

        # Step 1
        X['Booking Value'] = X['Booking Value'].fillna(X['rv_median'])
        print("Number of missing values after vehicle + route imputation:", X['Booking Value'].isnull().sum())
        print(f"Statistics of 'Booking Value' after vehicle + route imputation:\n{X['Booking Value'].describe()}")

        # Step 2
        X['scaled_route_price'] = X['route_mean'] * X['vehicle_scale']
        X['Booking Value'] = X['Booking Value'].fillna(X['scaled_route_price'])
        print("Number of missing values after scaled route price imputation:", X['Booking Value'].isnull().sum())
        print(f"Statistics of 'Booking Value' after scaled route price imputation:\n{X['Booking Value'].describe()}")

        # Step 3
        X['Booking Value'] = X['Booking Value'].fillna(X['v_median'])
        print("Number of missing values after vehicle median imputation:", X['Booking Value'].isnull().sum())
        print(f"Statistics of 'Booking Value' after vehicle median imputation:\n{X['Booking Value'].describe()}")

        # Final fallback 
        X['Booking Value'] = X['Booking Value'].fillna(self.global_mean)
        print("Number of missing values after global mean imputation:", X['Booking Value'].isnull().sum())
        print(f"Statistics of 'Booking Value' after global mean imputation:\n{X['Booking Value'].describe()}")

        # Cleanup
        X.drop(columns=[
            'rv_median', 'route_mean', 'vehicle_scale',
            'v_median', 'scaled_route_price', 'route'
        ], inplace=True)

        return X
imputer = BookingValueImputer()

# Fit ONLY on train
train_df = imputer.fit_transform(train_df)

# Apply SAME learned stats on test
test_df = imputer.transform(test_df)

# Save the preprocessed data into processed folder
print("Saving preprocessed data into processed folder...")

data_path = os.path.join("data", "processed")
if not os.path.exists(data_path):
    os.makedirs(data_path)
train_df.to_csv(os.path.join(data_path, "train.csv"), index=False)
test_df.to_csv(os.path.join(data_path, "test.csv"), index=False)
print("Preprocessed data saved successfully.")