import numpy as np 
import pandas as pd
import os

from sklearn.preprocessing import OrdinalEncoder

# load data
print("Loading data...")
train_df = pd.read_csv("./data/processed/train.csv")
test_df = pd.read_csv("./data/processed/test.csv")
print("Data loaded successfully.")

# conver Datetime to datetime format
print("Converting Datetime to datetime format...")
train_df['Datetime'] = pd.to_datetime(train_df['Datetime'])
test_df['Datetime'] = pd.to_datetime(test_df['Datetime'])
print("Datetime conversion complete.")

# Time features
print("Creating time features...")
train_df['day_of_week'] = train_df['Datetime'].dt.dayofweek
test_df['day_of_week'] = test_df['Datetime'].dt.dayofweek
train_df['month'] = train_df['Datetime'].dt.month
test_df['month'] = test_df['Datetime'].dt.month
train_df['is_weekend'] = train_df['day_of_week'].apply(lambda x: 1 if x >= 5 else 0)
test_df['is_weekend'] = test_df['day_of_week'].apply(lambda x: 1 if x >= 5 else 0)
train_df["is_peak_hour"] = train_df['Datetime'].dt.hour.apply(lambda x: 1 if (7 <= x <= 9) or (17 <= x <= 19) else 0)
test_df["is_peak_hour"] = test_df['Datetime'].dt.hour.apply(lambda x: 1 if (7 <= x <= 9) or (17 <= x <= 19) else 0)
train_df.drop(columns=['Datetime'], inplace=True)
test_df.drop(columns=['Datetime'], inplace=True)
print("Time features created successfully.")

# Label encoding for categorical features
print("Encoding categorical features...")
# Define order (low → high prices)
vehicle_price_order = [['eBike', 'Bike', 'Auto', 'Go Mini', 'Go Sedan', 'Premier Sedan', 'Uber XL']]

encoder = OrdinalEncoder(categories=vehicle_price_order)

train_df[['Vehicle Type']] = encoder.fit_transform(train_df[['Vehicle Type']])
test_df[['Vehicle Type']] = encoder.transform(test_df[['Vehicle Type']])
print("Categorical features encoded successfully.")

# creating features for pickup and drop locations
print("Creating features for pickup and drop locations...")
pickup_freq = train_df['Pickup Location'].value_counts()

train_df['pickup_freq'] = train_df['Pickup Location'].map(pickup_freq)
test_df['pickup_freq']  = test_df['Pickup Location'].map(pickup_freq).fillna(0)

drop_freq = train_df['Drop Location'].value_counts()

train_df['drop_freq'] = train_df['Drop Location'].map(drop_freq)
test_df['drop_freq']  = test_df['Drop Location'].map(drop_freq).fillna(0)
print("Features for pickup and drop locations created successfully.")

# Creating route feature
print("Creating route feature...")

train_df['route'] = train_df['Pickup Location'] + "_" + train_df['Drop Location']
test_df['route']  = test_df['Pickup Location'] + "_" + test_df['Drop Location']
route_avg = train_df.groupby('route')['Booking Value'].mean()
train_df['route_avg_price'] = train_df['route'].map(route_avg)
test_df['route_avg_price']  = test_df['route'].map(route_avg).fillna(train_df['Booking Value'].mean())

train_df['price_vs_route'] = train_df['Booking Value'] / train_df['route_avg_price']
test_df['price_vs_route']  = test_df['Booking Value'] / test_df['route_avg_price']

train_df.drop(columns=['route',"Pickup Location","Drop Location"], inplace=True)
test_df.drop(columns=['route',"Pickup Location","Drop Location"], inplace=True)
print("Route feature created successfully.")

# Feature transformation
print("Applying feature transformation...")

train_df['log_price'] = np.log1p(train_df['Booking Value'])
train_df['log_route_avg'] = np.log1p(train_df['route_avg_price'])
train_df['is_high_value'] = (train_df['price_vs_route'] > 1).astype(int)
test_df['log_price'] = np.log1p(test_df['Booking Value'])
test_df['log_route_avg'] = np.log1p(test_df['route_avg_price'])
test_df['is_high_value'] = (test_df['price_vs_route'] > 1).astype(int)
print("Feature transformation applied successfully.")

# Feature related to location popularity
print("Creating features related to location popularity...")
train_df['location_demand'] = train_df['pickup_freq'] + train_df['drop_freq']
train_df['location_interaction'] = train_df['pickup_freq'] * train_df['drop_freq']
train_df.drop(columns=['pickup_freq', 'drop_freq'], inplace=True)
test_df['location_demand'] = test_df['pickup_freq'] + test_df['drop_freq']
test_df['location_interaction']  = test_df['pickup_freq'] * test_df['drop_freq']
test_df.drop(columns=['pickup_freq', 'drop_freq'], inplace=True)
print("Features related to location popularity created successfully.")

# Feature selection
print("Selecting features...")
drop_cols = [
    'Booking Value',
    'route_avg_price',
    'location_interaction',
    'month'
]
train_df = train_df.drop(columns=drop_cols)
test_df  = test_df.drop(columns=drop_cols)
print("Features selected successfully.")

# Save the engineered data
print("Saving engineered data...")
data_path = os.path.join("data", "feature")
if not os.path.exists(data_path):
    os.makedirs(data_path)
train_df.to_csv(os.path.join(data_path, "train.csv"), index=False)
test_df.to_csv(os.path.join(data_path, "test.csv"), index=False)
print("Engineered data saved successfully.")

