"""Test isolation: never touch the user's real data dir or live stores."""
import os
import tempfile

# Use a throwaway data dir so tests never pollute data/domains (the real registry).
os.environ["MALAR_DATA_DIR"] = tempfile.mkdtemp(prefix="malar_test_")
os.environ.setdefault("QDRANT_URL", ":memory:")
