"""Data quality inspection script for Mall_Customers.csv."""
import json
import pandas as pd
import numpy as np

def inspect_dataset(csv_path: str = "data/raw/Mall_Customers.csv"):
    df = pd.read_csv(csv_path)
    
    report = {
        "file_path": csv_path,
        "row_count": int(len(df)),
        "column_count": int(len(df.columns)),
        "columns": list(df.columns),
        "data_types": {col: str(dtype) for col, dtype in df.dtypes.items()},
        "missing_values": {col: int(val) for col, val in df.isnull().sum().items()},
        "total_missing": int(df.isnull().sum().sum()),
        "duplicate_customer_ids": int(df["CustomerID"].duplicated().sum()),
        "duplicate_rows": int(df.duplicated().sum()),
        "unique_customers": int(df["CustomerID"].nunique()),
        "gender_counts": {str(k): int(v) for k, v in df["Gender"].value_counts().items()},
        "age_stats": {
            "min": int(df["Age"].min()),
            "max": int(df["Age"].max()),
            "mean": round(float(df["Age"].mean()), 2),
            "median": float(df["Age"].median()),
            "invalid_count": int(((df["Age"] <= 0) | (df["Age"] > 120)).sum()),
        },
        "annual_income_k_stats": {
            "min": int(df["Annual Income (k$)"].min()),
            "max": int(df["Annual Income (k$)"].max()),
            "mean": round(float(df["Annual Income (k$)"].mean()), 2),
            "median": float(df["Annual Income (k$)"].median()),
            "invalid_count": int((df["Annual Income (k$)"] < 0).sum()),
        },
        "spending_score_stats": {
            "min": int(df["Spending Score (1-100)"].min()),
            "max": int(df["Spending Score (1-100)"].max()),
            "mean": round(float(df["Spending Score (1-100)"].mean()), 2),
            "median": float(df["Spending Score (1-100)"].median()),
            "invalid_count": int(((df["Spending Score (1-100)"] < 1) | (df["Spending Score (1-100)"] > 100)).sum()),
        }
    }
    print(json.dumps(report, indent=2))
    return report

if __name__ == "__main__":
    inspect_dataset()
