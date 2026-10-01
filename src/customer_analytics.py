"""Customer Analytics and Segmentation Module.

This module processes Mall_Customers.csv, validates data quality, performs K-Means clustering,
computes analytics and metrics, and manages the database table `mall_customers`.

NOTE: This is strictly separate from the music recommendation catalog.
"""
from dataclasses import asdict, dataclass
import json
import logging
import os
import sqlite3
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.metrics import calinski_harabasz_score, davies_bouldin_score, silhouette_score
from sklearn.preprocessing import StandardScaler

logger = logging.getLogger(__name__)

DEFAULT_DB_PATH = os.environ.get("CUSTOMER_DB_PATH", "data/customers.db")
DEFAULT_RAW_CSV = "data/raw/Mall_Customers.csv"
DEFAULT_PROCESSED_CSV = "data/processed/mall_customers_processed.csv"
DEFAULT_METRICS_JSON = "data/processed/customer_clusters_metrics.json"

RAW_REQUIRED_COLUMNS = [
    "CustomerID",
    "Gender",
    "Age",
    "Annual Income (k$)",
    "Spending Score (1-100)"
]

COLUMN_MAPPING = {
    "CustomerID": "customer_id",
    "Gender": "gender",
    "Age": "age",
    "Annual Income (k$)": "annual_income_k",
    "Spending Score (1-100)": "spending_score"
}

FEATURE_COLUMNS = ["age", "annual_income_k", "spending_score"]


class DataQualityValidationError(ValueError):
    """Raised when customer raw dataset fails schema or quality validation."""
    pass


def validate_and_clean_data(raw_path: str = DEFAULT_RAW_CSV) -> Tuple[pd.DataFrame, Dict[str, Any]]:
    """Validate, inspect, and clean the raw customer dataset.
    
    Returns:
        cleaned_df: DataFrame with standardized column names and validated data types.
        report: Detailed data quality validation report.
    """
    if not os.path.exists(raw_path):
        # Also check parent directory if needed
        fallback = os.path.join(os.path.dirname(__file__), "..", raw_path)
        if os.path.exists(fallback):
            raw_path = fallback
        else:
            raise FileNotFoundError(f"Customer dataset not found at '{raw_path}'.")

    df = pd.read_csv(raw_path)
    
    # 1. Validate required columns
    missing_cols = [col for col in RAW_REQUIRED_COLUMNS if col not in df.columns]
    if missing_cols:
        raise DataQualityValidationError(f"Missing required columns in CSV: {missing_cols}")

    initial_row_count = len(df)
    
    # 2. Check for duplicate rows
    duplicate_rows = int(df.duplicated().sum())
    
    # 3. Rename columns according to data model
    clean_df = df[RAW_REQUIRED_COLUMNS].copy()
    clean_df = clean_df.rename(columns=COLUMN_MAPPING)

    # 4. Check for null / missing values
    missing_values = clean_df.isnull().sum().to_dict()
    total_missing = int(sum(missing_values.values()))
    if total_missing > 0:
        # Strict handling: drop rows with missing values (do not fabricate)
        clean_df = clean_df.dropna(subset=list(COLUMN_MAPPING.values()))

    # 5. Type and range validation
    # CustomerID unique & numeric
    try:
        clean_df["customer_id"] = pd.to_numeric(clean_df["customer_id"], errors="raise").astype(int)
    except Exception as e:
        raise DataQualityValidationError(f"Invalid non-numeric customer_id: {e}")

    duplicate_ids = int(clean_df["customer_id"].duplicated().sum())
    if duplicate_ids > 0:
        clean_df = clean_df.drop_duplicates(subset=["customer_id"], keep="first")

    # Age numeric & valid
    try:
        clean_df["age"] = pd.to_numeric(clean_df["age"], errors="raise").astype(int)
    except Exception as e:
        raise DataQualityValidationError(f"Invalid non-numeric age: {e}")

    invalid_age_mask = (clean_df["age"] <= 0) | (clean_df["age"] > 120)
    invalid_age_count = int(invalid_age_mask.sum())
    if invalid_age_count > 0:
        clean_df = clean_df[~invalid_age_mask]

    # Annual Income numeric & non-negative
    try:
        clean_df["annual_income_k"] = pd.to_numeric(clean_df["annual_income_k"], errors="raise").astype(float)
    except Exception as e:
        raise DataQualityValidationError(f"Invalid non-numeric annual_income_k: {e}")

    invalid_income_mask = clean_df["annual_income_k"] < 0
    invalid_income_count = int(invalid_income_mask.sum())
    if invalid_income_count > 0:
        clean_df = clean_df[~invalid_income_mask]

    # Spending Score numeric & between 1 and 100
    try:
        clean_df["spending_score"] = pd.to_numeric(clean_df["spending_score"], errors="raise").astype(int)
    except Exception as e:
        raise DataQualityValidationError(f"Invalid non-numeric spending_score: {e}")

    invalid_spending_mask = (clean_df["spending_score"] < 1) | (clean_df["spending_score"] > 100)
    invalid_spending_count = int(invalid_spending_mask.sum())
    if invalid_spending_count > 0:
        clean_df = clean_df[~invalid_spending_mask]

    # Gender normalized (e.g., 'Male', 'Female')
    clean_df["gender"] = clean_df["gender"].astype(str).str.strip().str.capitalize()
    valid_genders = {"Male", "Female", "Other"}
    # Verify gender values
    clean_df = clean_df[clean_df["gender"].isin(valid_genders)]

    # Final report
    report = {
        "raw_path": raw_path,
        "initial_rows": initial_row_count,
        "cleaned_rows": len(clean_df),
        "columns_validated": list(clean_df.columns),
        "missing_values_found": missing_values,
        "total_missing": total_missing,
        "duplicate_rows": duplicate_rows,
        "duplicate_customer_ids": duplicate_ids,
        "invalid_age_records": invalid_age_count,
        "invalid_income_records": invalid_income_count,
        "invalid_spending_records": invalid_spending_count,
        "is_valid": len(clean_df) > 0
    }
    
    return clean_df, report


def evaluate_clustering(
    df: pd.DataFrame,
    features: Optional[List[str]] = None,
    min_k: int = 2,
    max_k: int = 10,
    random_state: int = 42
) -> Dict[str, Any]:
    """Evaluate K-Means across multiple values of K using objective metrics."""
    if features is None:
        features = FEATURE_COLUMNS
        
    X = df[features].values
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    k_evaluations = []
    best_k = min_k
    best_sil = -1.0

    for k in range(min_k, max_k + 1):
        km = KMeans(n_clusters=k, random_state=random_state, n_init=10)
        labels = km.fit_predict(X_scaled)
        
        sil = float(silhouette_score(X_scaled, labels))
        db = float(davies_bouldin_score(X_scaled, labels))
        ch = float(calinski_harabasz_score(X_scaled, labels))
        inertia = float(km.inertia_)

        k_evaluations.append({
            "k": k,
            "silhouette_score": round(sil, 4),
            "inertia": round(inertia, 2),
            "davies_bouldin_index": round(db, 4),
            "calinski_harabasz_score": round(ch, 2)
        })

        if sil > best_sil:
            best_sil = sil
            best_k = k

    return {
        "features_used": features,
        "k_range": [min_k, max_k],
        "evaluations": k_evaluations,
        "optimal_k": best_k,
        "best_silhouette_score": round(best_sil, 4)
    }


def fit_customer_clusters(
    df: pd.DataFrame,
    n_clusters: Optional[int] = None,
    features: Optional[List[str]] = None,
    random_state: int = 42
) -> Tuple[pd.DataFrame, Dict[str, Any]]:
    """Fit K-Means clustering on customer dataset and assign neutral cluster labels.
    
    Returns:
        df_clustered: DataFrame with added 'cluster_id' and 'cluster_name' columns.
        cluster_metadata: Dictionary with cluster profiles, centroids, and evaluation metrics.
    """
    if features is None:
        features = FEATURE_COLUMNS

    eval_results = evaluate_clustering(df, features=features, random_state=random_state)
    selected_k = n_clusters if n_clusters is not None else eval_results["optimal_k"]

    X = df[features].values
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    km = KMeans(n_clusters=selected_k, random_state=random_state, n_init=10)
    cluster_labels = km.fit_predict(X_scaled)

    df_clustered = df.copy()
    df_clustered["cluster_id"] = cluster_labels
    df_clustered["cluster_name"] = [f"Cluster {cid}" for cid in cluster_labels]

    # Calculate cluster statistics and derive objective descriptions
    overall_mean_age = float(df_clustered["age"].mean())
    overall_mean_income = float(df_clustered["annual_income_k"].mean())
    overall_mean_spending = float(df_clustered["spending_score"].mean())

    cluster_profiles = []
    for cid in range(selected_k):
        c_df = df_clustered[df_clustered["cluster_id"] == cid]
        c_count = len(c_df)
        c_mean_age = round(float(c_df["age"].mean()), 2)
        c_mean_inc = round(float(c_df["annual_income_k"].mean()), 2)
        c_mean_spd = round(float(c_df["spending_score"].mean()), 2)
        c_gender_dist = c_df["gender"].value_counts().to_dict()

        # Derive objective characterization based on measured differences from dataset average
        age_desc = "Older" if c_mean_age > overall_mean_age + 3 else ("Younger" if c_mean_age < overall_mean_age - 3 else "Avg Age")
        inc_desc = "High Income" if c_mean_inc > overall_mean_income + 5 else ("Low Income" if c_mean_inc < overall_mean_income - 5 else "Moderate Income")
        spd_desc = "High Spending" if c_mean_spd > overall_mean_spending + 5 else ("Low Spending" if c_mean_spd < overall_mean_spending - 5 else "Moderate Spending")
        
        derived_description = f"{age_desc}, {inc_desc}, {spd_desc}"

        cluster_profiles.append({
            "cluster_id": cid,
            "cluster_name": f"Cluster {cid}",
            "customer_count": c_count,
            "percentage": round(c_count / len(df_clustered) * 100, 2),
            "mean_age": c_mean_age,
            "mean_annual_income_k": c_mean_inc,
            "mean_spending_score": c_mean_spd,
            "gender_distribution": c_gender_dist,
            "derived_description": derived_description
        })

    cluster_metadata = {
        "algorithm": "K-Means",
        "selected_k": selected_k,
        "features": features,
        "evaluation": eval_results,
        "silhouette_score": round(float(silhouette_score(X_scaled, cluster_labels)), 4),
        "cluster_profiles": cluster_profiles,
        "overall_averages": {
            "mean_age": round(overall_mean_age, 2),
            "mean_annual_income_k": round(overall_mean_income, 2),
            "mean_spending_score": round(overall_mean_spending, 2),
        }
    }

    return df_clustered, cluster_metadata


def compute_customer_statistics(df: pd.DataFrame) -> Dict[str, Any]:
    """Compute aggregate statistical distributions from the actual customer dataset."""
    total = len(df)
    if total == 0:
        return {}

    gender_counts = df["gender"].value_counts().to_dict()
    gender_percentages = {k: round(v / total * 100, 2) for k, v in gender_counts.items()}

    # Age distribution bins: 18-25, 26-35, 36-50, 51-70+
    age_bins = [17, 25, 35, 50, 100]
    age_labels = ["18-25", "26-35", "36-50", "51+"]
    df_age_binned = pd.cut(df["age"], bins=age_bins, labels=age_labels)
    age_dist = df_age_binned.value_counts()[age_labels].to_dict()

    # Income distribution bins: <30k, 30k-60k, 60k-90k, 90k+
    income_bins = [0, 30, 60, 90, 200]
    income_labels = ["< $30k", "$30k - $60k", "$60k - $90k", "$90k+"]
    df_inc_binned = pd.cut(df["annual_income_k"], bins=income_bins, labels=income_labels)
    income_dist = df_inc_binned.value_counts()[income_labels].to_dict()

    # Spending Score distribution bins: 1-25, 26-50, 51-75, 76-100
    spd_bins = [0, 25, 50, 75, 100]
    spd_labels = ["Low (1-25)", "Moderate (26-50)", "High (51-75)", "Very High (76-100)"]
    df_spd_binned = pd.cut(df["spending_score"], bins=spd_bins, labels=spd_labels)
    spd_dist = df_spd_binned.value_counts()[spd_labels].to_dict()

    return {
        "total_customers": total,
        "average_age": round(float(df["age"].mean()), 2),
        "median_age": float(df["age"].median()),
        "average_annual_income_k": round(float(df["annual_income_k"].mean()), 2),
        "median_annual_income_k": float(df["annual_income_k"].median()),
        "average_spending_score": round(float(df["spending_score"].mean()), 2),
        "median_spending_score": float(df["spending_score"].median()),
        "gender_counts": gender_counts,
        "gender_percentages": gender_percentages,
        "age_distribution": {k: int(v) for k, v in age_dist.items()},
        "income_distribution": {k: int(v) for k, v in income_dist.items()},
        "spending_score_distribution": {k: int(v) for k, v in spd_dist.items()},
        "correlations": {
            "age_vs_spending": round(float(df["age"].corr(df["spending_score"])), 4),
            "income_vs_spending": round(float(df["annual_income_k"].corr(df["spending_score"])), 4),
            "age_vs_income": round(float(df["age"].corr(df["annual_income_k"])), 4),
        }
    }


# ==========================================
# Database Operations
# ==========================================

def get_db_connection(db_path: str = DEFAULT_DB_PATH) -> sqlite3.Connection:
    """Create or connect to the SQLite database."""
    os.makedirs(os.path.dirname(os.path.abspath(db_path)), exist_ok=True)
    conn = sqlite3.connect(db_path)
    conn.row_factory = sqlite3.Row
    return conn


def init_customer_table(db_path: str = DEFAULT_DB_PATH):
    """Initialize the separate `mall_customers` table."""
    conn = get_db_connection(db_path)
    with conn:
        conn.execute("""
            CREATE TABLE IF NOT EXISTS mall_customers (
                customer_id INTEGER PRIMARY KEY,
                gender TEXT NOT NULL,
                age INTEGER NOT NULL,
                annual_income_k REAL NOT NULL,
                spending_score INTEGER NOT NULL,
                cluster_id INTEGER
            );
        """)
        conn.execute("CREATE INDEX IF NOT EXISTS idx_cust_cluster ON mall_customers(cluster_id);")
        conn.execute("CREATE INDEX IF NOT EXISTS idx_cust_gender ON mall_customers(gender);")
    conn.close()


def save_customers_to_db(df: pd.DataFrame, db_path: str = DEFAULT_DB_PATH) -> int:
    """Save/upsert customer records into the `mall_customers` table."""
    init_customer_table(db_path)
    conn = get_db_connection(db_path)
    
    records = []
    for _, row in df.iterrows():
        records.append((
            int(row["customer_id"]),
            str(row["gender"]),
            int(row["age"]),
            float(row["annual_income_k"]),
            int(row["spending_score"]),
            int(row["cluster_id"]) if "cluster_id" in row and pd.notnull(row["cluster_id"]) else None
        ))

    with conn:
        conn.executemany("""
            INSERT INTO mall_customers (customer_id, gender, age, annual_income_k, spending_score, cluster_id)
            VALUES (?, ?, ?, ?, ?, ?)
            ON CONFLICT(customer_id) DO UPDATE SET
                gender=excluded.gender,
                age=excluded.age,
                annual_income_k=excluded.annual_income_k,
                spending_score=excluded.spending_score,
                cluster_id=excluded.cluster_id;
        """, records)
    
    count = conn.execute("SELECT COUNT(*) FROM mall_customers;").fetchone()[0]
    conn.close()
    return count


def get_customers_from_db(
    db_path: str = DEFAULT_DB_PATH,
    skip: int = 0,
    limit: int = 50,
    gender: Optional[str] = None,
    cluster_id: Optional[int] = None
) -> Tuple[List[Dict[str, Any]], int]:
    """Retrieve paginated customer records from the database."""
    init_customer_table(db_path)
    conn = get_db_connection(db_path)
    
    query = "SELECT customer_id, gender, age, annual_income_k, spending_score, cluster_id FROM mall_customers WHERE 1=1"
    count_query = "SELECT COUNT(*) FROM mall_customers WHERE 1=1"
    params: List[Any] = []

    if gender is not None:
        query += " AND LOWER(gender) = LOWER(?)"
        count_query += " AND LOWER(gender) = LOWER(?)"
        params.append(gender)
    
    if cluster_id is not None:
        query += " AND cluster_id = ?"
        count_query += " AND cluster_id = ?"
        params.append(cluster_id)

    total = conn.execute(count_query, params).fetchone()[0]

    query += " ORDER BY customer_id ASC LIMIT ? OFFSET ?"
    query_params = list(params) + [limit, skip]
    
    rows = conn.execute(query, query_params).fetchall()
    customers = [dict(r) for r in rows]
    conn.close()
    return customers, total


def get_customer_by_id_from_db(customer_id: int, db_path: str = DEFAULT_DB_PATH) -> Optional[Dict[str, Any]]:
    """Retrieve single customer record by customer_id."""
    init_customer_table(db_path)
    conn = get_db_connection(db_path)
    row = conn.execute(
        "SELECT customer_id, gender, age, annual_income_k, spending_score, cluster_id FROM mall_customers WHERE customer_id = ?",
        (customer_id,)
    ).fetchone()
    conn.close()
    return dict(row) if row else None


def load_all_customers_df(db_path: str = DEFAULT_DB_PATH, fallback_csv: str = DEFAULT_PROCESSED_CSV) -> pd.DataFrame:
    """Load full customer dataframe from DB or fallback processed CSV."""
    if os.path.exists(db_path):
        conn = get_db_connection(db_path)
        try:
            df = pd.read_sql_query("SELECT * FROM mall_customers ORDER BY customer_id ASC", conn)
            conn.close()
            if len(df) > 0:
                return df
        except Exception:
            conn.close()

    if os.path.exists(fallback_csv):
        return pd.read_csv(fallback_csv)
    
    # Try raw CSV if processed does not exist yet
    if os.path.exists(DEFAULT_RAW_CSV):
        clean_df, _ = validate_and_clean_data(DEFAULT_RAW_CSV)
        clustered_df, _ = fit_customer_clusters(clean_df)
        return clustered_df

    return pd.DataFrame(columns=["customer_id", "gender", "age", "annual_income_k", "spending_score", "cluster_id"])
