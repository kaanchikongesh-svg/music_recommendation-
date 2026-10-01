"""Wrapper module to allow `python -m backend.data.import_customers` invocation."""
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", "src"))
from import_customers import main, run_customer_import  # noqa: E402

if __name__ == "__main__":
    main()
