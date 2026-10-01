"""Import, clean, cluster, and store Mall_Customers.csv data.

Usage:
    python src/import_customers.py [--raw data/raw/Mall_Customers.csv] [--db data/customers.db]
    python -m src.import_customers
"""
import argparse
import json
import logging
import os
import sys

# Ensure src is on sys.path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from customer_analytics import (
    DEFAULT_DB_PATH,
    DEFAULT_PROCESSED_CSV,
    DEFAULT_RAW_CSV,
    DEFAULT_METRICS_JSON,
    compute_customer_statistics,
    fit_customer_clusters,
    save_customers_to_db,
    validate_and_clean_data,
)

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)


def locate_raw_csv(specified_path: str = None) -> str:
    """Locate the Mall_Customers.csv file across common search locations."""
    candidates = []
    if specified_path:
        candidates.append(specified_path)
    
    candidates.extend([
        DEFAULT_RAW_CSV,
        os.path.join(os.path.dirname(__file__), "..", DEFAULT_RAW_CSV),
        os.path.join("data", "raw", "Mall_Customers.csv"),
        os.path.join(os.path.expanduser("~"), "Downloads", "Mall_Customers.csv"),
        "Mall_Customers.csv"
    ])
    
    for path in candidates:
        norm = os.path.normpath(path)
        if os.path.isfile(norm):
            return norm
            
    raise FileNotFoundError(
        f"Could not locate Mall_Customers.csv in any of: {[os.path.normpath(p) for p in candidates]}"
    )


def run_customer_import(
    raw_path: str = None,
    db_path: str = DEFAULT_DB_PATH,
    processed_csv_path: str = DEFAULT_PROCESSED_CSV,
    metrics_json_path: str = DEFAULT_METRICS_JSON,
    k_clusters: int = None
) -> dict:
    """Execute complete end-to-end customer dataset import pipeline.
    
    1. Locate Mall_Customers.csv
    2. Validate columns & data types
    3. Clean data & detect duplicates
    4. Run clustering evaluation and model fit
    5. Save/upsert to database
    6. Save processed dataset & cluster metrics
    7. Generate detailed report
    """
    logger.info("Step 1: Locating Mall_Customers.csv...")
    found_raw_path = locate_raw_csv(raw_path)
    logger.info("Found raw CSV at: %s", found_raw_path)

    # Copy to data/raw/Mall_Customers.csv if not already there
    target_raw = os.path.normpath(DEFAULT_RAW_CSV)
    if os.path.abspath(found_raw_path) != os.path.abspath(target_raw):
        os.makedirs(os.path.dirname(target_raw), exist_ok=True)
        import shutil
        shutil.copyfile(found_raw_path, target_raw)
        logger.info("Cached raw copy to: %s", target_raw)

    logger.info("Step 2 & 3: Validating schema, data types, cleaning, and checking duplicates...")
    clean_df, quality_report = validate_and_clean_data(found_raw_path)
    logger.info(
        "Data validation passed: %d records ready (Total missing: %d, Duplicate IDs: %d)",
        quality_report["cleaned_rows"],
        quality_report["total_missing"],
        quality_report["duplicate_customer_ids"]
    )

    logger.info("Step 4: Running K-Means clustering evaluation and fitting...")
    clustered_df, cluster_metadata = fit_customer_clusters(clean_df, n_clusters=k_clusters)
    selected_k = cluster_metadata["selected_k"]
    sil_score = cluster_metadata["silhouette_score"]
    logger.info("Clustering completed: Selected K=%d, Silhouette Score=%.4f", selected_k, sil_score)

    logger.info("Step 5: Storing records into database table 'mall_customers' at %s...", db_path)
    total_in_db = save_customers_to_db(clustered_df, db_path=db_path)
    logger.info("Database updated successfully: %d records in 'mall_customers'", total_in_db)

    logger.info("Step 6: Saving processed CSV to %s...", processed_csv_path)
    os.makedirs(os.path.dirname(os.path.abspath(processed_csv_path)), exist_ok=True)
    clustered_df.to_csv(processed_csv_path, index=False)

    stats = compute_customer_statistics(clustered_df)
    
    full_report = {
        "status": "success",
        "raw_file": found_raw_path,
        "database_file": db_path,
        "table_name": "mall_customers",
        "total_records_imported": total_in_db,
        "quality_report": quality_report,
        "clustering": cluster_metadata,
        "overall_statistics": stats
    }

    os.makedirs(os.path.dirname(os.path.abspath(metrics_json_path)), exist_ok=True)
    with open(metrics_json_path, "w", encoding="utf-8") as f:
        json.dump(full_report, f, indent=2)

    logger.info("Step 7: Import report saved to %s", metrics_json_path)
    return full_report


def main():
    parser = argparse.ArgumentParser(description="Import, clean, cluster, and store Mall_Customers.csv")
    parser.add_argument("--raw", default=None, help="Path to raw Mall_Customers.csv")
    parser.add_argument("--db", default=DEFAULT_DB_PATH, help="Path to SQLite database")
    parser.add_argument("--out", default=DEFAULT_PROCESSED_CSV, help="Path to output processed CSV")
    parser.add_argument("--k", type=int, default=None, help="Force specific number of clusters K (optional)")
    args = parser.parse_args()

    report = run_customer_import(
        raw_path=args.raw,
        db_path=args.db,
        processed_csv_path=args.out,
        k_clusters=args.k
    )
    print("\n" + "=" * 60)
    print("CUSTOMER IMPORT & CLUSTERING SUMMARY")
    print("=" * 60)
    print(f"Raw file:             {report['raw_file']}")
    print(f"Database Table:       {report['table_name']} ({report['database_file']})")
    print(f"Records Processed:    {report['total_records_imported']}")
    print(f"Optimal K:            {report['clustering']['selected_k']}")
    print(f"Silhouette Score:     {report['clustering']['silhouette_score']}")
    print(f"Cluster Breakdown:")
    for prof in report['clustering']['cluster_profiles']:
        print(f"  - {prof['cluster_name']}: {prof['customer_count']} customers ({prof['percentage']}%) | "
              f"Avg Age={prof['mean_age']}, Avg Income=${prof['mean_annual_income_k']}k, "
              f"Avg Spending={prof['mean_spending_score']} | [{prof['derived_description']}]")
    print("=" * 60)


if __name__ == "__main__":
    main()
