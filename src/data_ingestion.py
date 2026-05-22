import numpy as np
import pandas as pd

import os

# Extract data from gdrive
print("Extracting data from gdrive...")
train_df = pd.read_csv("https://drive.google.com/uc?id=1ZKPqkUkP93coteNakxVKeEBpJsdIgxio")
test_df = pd.read_csv("https://drive.google.com/uc?id=1y6JBpCTDpOwgO1UkIvx3iCI-EPVK9tvr")
print("Data extraction complete.")

# Save the data into raw folder
print("Saving data into raw folder...")

data_path = os.path.join("data", "raw")
if not os.path.exists(data_path):
    os.makedirs(data_path)
train_df.to_csv("data/raw/train.csv", index=False)
test_df.to_csv("data/raw/test.csv", index=False)
print("Data saved successfully.")
