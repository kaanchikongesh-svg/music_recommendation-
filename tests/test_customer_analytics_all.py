"""Comprehensive test suite for Customer Analytics, Music Recommender, and FastAPI endpoints."""
import os
import sys
import tempfile
import unittest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "api"))

from customer_analytics import (
    DEFAULT_RAW_CSV,
    DataQualityValidationError,
    compute_customer_statistics,
    evaluate_clustering,
    fit_customer_clusters,
    get_customer_by_id_from_db,
    get_customers_from_db,
    save_customers_to_db,
    validate_and_clean_data,
)
from import_customers import locate_raw_csv, run_customer_import
from recommend import load_recommender
from main import app


class TestCustomerAnalytics(unittest.TestCase):
    def test_locate_raw_csv(self):
        path = locate_raw_csv()
        self.assertTrue(os.path.isfile(path))
        self.assertIn("Mall_Customers.csv", path)

    def test_real_dataset_validation(self):
        clean_df, report = validate_and_clean_data(DEFAULT_RAW_CSV)
        self.assertTrue(report["is_valid"])
        self.assertEqual(report["initial_rows"], 200)
        self.assertEqual(report["cleaned_rows"], 200)
        self.assertEqual(report["total_missing"], 0)
        self.assertEqual(report["duplicate_customer_ids"], 0)
        self.assertEqual(
            list(clean_df.columns),
            ["customer_id", "gender", "age", "annual_income_k", "spending_score"]
        )

    def test_missing_required_columns_fails(self):
        with tempfile.NamedTemporaryFile(mode="w", suffix=".csv", delete=False) as f:
            f.write("CustomerID,Gender,Age\n1,Male,20\n")
            temp_path = f.name
        try:
            with self.assertRaises(DataQualityValidationError):
                validate_and_clean_data(temp_path)
        finally:
            if os.path.exists(temp_path):
                os.remove(temp_path)

    def test_duplicate_customer_id_handling(self):
        with tempfile.NamedTemporaryFile(mode="w", suffix=".csv", delete=False) as f:
            f.write("CustomerID,Gender,Age,Annual Income (k$),Spending Score (1-100)\n"
                    "1,Male,20,15,39\n"
                    "1,Male,20,15,39\n"
                    "2,Female,22,16,80\n")
            temp_path = f.name
        try:
            clean_df, report = validate_and_clean_data(temp_path)
            self.assertEqual(report["duplicate_customer_ids"], 1)
            self.assertEqual(len(clean_df), 2)
            self.assertEqual(clean_df["customer_id"].tolist(), [1, 2])
        finally:
            if os.path.exists(temp_path):
                os.remove(temp_path)

    def test_invalid_numeric_ranges_handling(self):
        with tempfile.NamedTemporaryFile(mode="w", suffix=".csv", delete=False) as f:
            f.write("CustomerID,Gender,Age,Annual Income (k$),Spending Score (1-100)\n"
                    "1,Male,-5,15,39\n"     # Invalid age
                    "2,Female,22,-10,80\n"   # Invalid income
                    "3,Female,25,30,150\n"   # Invalid spending score (>100)
                    "4,Male,30,40,50\n")     # Valid
            temp_path = f.name
        try:
            clean_df, report = validate_and_clean_data(temp_path)
            self.assertEqual(report["invalid_age_records"], 1)
            self.assertEqual(report["invalid_income_records"], 1)
            self.assertEqual(report["invalid_spending_records"], 1)
            self.assertEqual(len(clean_df), 1)
            self.assertEqual(clean_df.iloc[0]["customer_id"], 4)
        finally:
            if os.path.exists(temp_path):
                os.remove(temp_path)

    def test_clustering_evaluation(self):
        clean_df, _ = validate_and_clean_data(DEFAULT_RAW_CSV)
        eval_res = evaluate_clustering(clean_df, min_k=2, max_k=8)
        self.assertIn("evaluations", eval_res)
        self.assertEqual(len(eval_res["evaluations"]), 7)
        self.assertTrue(2 <= eval_res["optimal_k"] <= 8)
        self.assertTrue(0.0 < eval_res["best_silhouette_score"] < 1.0)

    def test_fit_customer_clusters(self):
        clean_df, _ = validate_and_clean_data(DEFAULT_RAW_CSV)
        clustered_df, metadata = fit_customer_clusters(clean_df, n_clusters=6)
        
        self.assertIn("cluster_id", clustered_df.columns)
        self.assertIn("cluster_name", clustered_df.columns)
        self.assertEqual(len(clustered_df), 200)
        self.assertEqual(set(clustered_df["cluster_id"].unique()), {0, 1, 2, 3, 4, 5})
        self.assertEqual(metadata["selected_k"], 6)
        self.assertEqual(len(metadata["cluster_profiles"]), 6)
        
        # Verify neutral labels
        for prof in metadata["cluster_profiles"]:
            self.assertTrue(prof["cluster_name"].startswith("Cluster "))
            self.assertIn("customer_count", prof)
            self.assertGreater(prof["customer_count"], 0)

    def test_customer_statistics(self):
        clean_df, _ = validate_and_clean_data(DEFAULT_RAW_CSV)
        stats = compute_customer_statistics(clean_df)
        
        self.assertEqual(stats["total_customers"], 200)
        self.assertAlmostEqual(stats["average_age"], 38.85, delta=0.1)
        self.assertAlmostEqual(stats["average_annual_income_k"], 60.56, delta=0.1)
        self.assertAlmostEqual(stats["average_spending_score"], 50.2, delta=0.1)
        self.assertEqual(stats["gender_counts"]["Female"], 112)
        self.assertEqual(stats["gender_counts"]["Male"], 88)

    def test_database_persistence_and_idempotency(self):
        with tempfile.NamedTemporaryFile(suffix=".db", delete=False) as f:
            temp_db = f.name
        try:
            clean_df, _ = validate_and_clean_data(DEFAULT_RAW_CSV)
            clustered_df, _ = fit_customer_clusters(clean_df, n_clusters=5)
            
            # 1st save
            count1 = save_customers_to_db(clustered_df, db_path=temp_db)
            self.assertEqual(count1, 200)
            
            # 2nd save (idempotency check)
            count2 = save_customers_to_db(clustered_df, db_path=temp_db)
            self.assertEqual(count2, 200)
            
            # Query paginated
            customers, total = get_customers_from_db(db_path=temp_db, skip=0, limit=10)
            self.assertEqual(total, 200)
            self.assertEqual(len(customers), 10)
            self.assertEqual(customers[0]["customer_id"], 1)
            
            # Single customer query
            cust1 = get_customer_by_id_from_db(1, db_path=temp_db)
            self.assertIsNotNone(cust1)
            self.assertEqual(cust1["customer_id"], 1)
            self.assertEqual(cust1["gender"], "Male")
        finally:
            if os.path.exists(temp_db):
                os.remove(temp_db)


class TestMusicRecommender(unittest.TestCase):
    def test_music_recommender(self):
        if not os.path.exists("models/svd_model.pkl"):
            self.skipTest("Music models not generated yet.")
        rec = load_recommender()
        
        # Test popular
        popular = rec.popular_recommendations(5)
        self.assertEqual(len(popular), 5)
        self.assertIn("song_id", popular[0])
        self.assertIn("score", popular[0])
        
        # Test search
        hits = rec.search_songs("Song", limit=5)
        self.assertIsInstance(hits, list)
        
        # Test personalized
        user_recs = rec.recommend_for_user("user_0", 5)
        self.assertEqual(len(user_recs), 5)
        
        # Test similar
        first_song = popular[0]["song_id"]
        sim_recs = rec.recommend_similar([first_song], 5)
        self.assertLessEqual(len(sim_recs), 5)


class TestFastAPIServices(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        from fastapi.testclient import TestClient
        cls.client = TestClient(app)

    def test_health(self):
        res = self.client.get("/health")
        self.assertEqual(res.status_code, 200)
        self.assertEqual(res.json()["status"], "ok")

    def test_api_customers(self):
        res = self.client.get("/api/customers?skip=0&limit=10")
        self.assertEqual(res.status_code, 200)
        data = res.json()
        self.assertEqual(data["total"], 200)
        self.assertEqual(len(data["customers"]), 10)

    def test_api_customers_stats(self):
        res = self.client.get("/api/customers/stats")
        self.assertEqual(res.status_code, 200)
        data = res.json()
        self.assertEqual(data["statistics"]["total_customers"], 200)

    def test_api_customers_clusters(self):
        res = self.client.get("/api/customers/clusters?n_clusters=6")
        self.assertEqual(res.status_code, 200)
        data = res.json()
        self.assertEqual(data["clustering"]["selected_k"], 6)

    def test_api_customer_single(self):
        res = self.client.get("/api/customers/1")
        self.assertEqual(res.status_code, 200)
        self.assertEqual(res.json()["customer"]["customer_id"], 1)

        res404 = self.client.get("/api/customers/99999")
        self.assertEqual(res404.status_code, 404)


if __name__ == "__main__":
    unittest.main(verbosity=2)
