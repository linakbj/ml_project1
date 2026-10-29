import numpy as np
from helpers import load_csv_data

x_train, _, y_train, _, _ = load_csv_data("data/dataset")

missing = np.isnan(x_train)

print("Dataset shape:", x_train.shape)
print("Total missing values:", missing.sum())
print("Missing percentage:", 100 * missing.mean())

missing_per_column = missing.mean(axis=0)

print("Columns containing missing values:", np.sum(missing_per_column > 0))

top_columns = np.argsort(missing_per_column)[-10:][::-1]

for col in top_columns:
    print(f"Column {col}: " f"{missing_per_column[col] * 100:.2f}% missing")
