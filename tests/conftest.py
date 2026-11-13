import os
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
os.environ.setdefault("DATABASE_URL", "postgresql://postgres:postgres@localhost:5432/modpipe_test")
os.environ.setdefault("MEDIA_DIR", str(ROOT / "var" / "test-media"))


@pytest.fixture(scope="session")
def pg():
    import psycopg
    url = os.environ["DATABASE_URL"]
    admin = url.rsplit("/", 1)[0] + "/postgres"
    try:
        with psycopg.connect(admin, autocommit=True) as c:
            c.execute("DROP DATABASE IF EXISTS modpipe_test WITH (FORCE)")
            c.execute("CREATE DATABASE modpipe_test")
    except psycopg.OperationalError as e:
        pytest.skip(f"no postgres: {e}")
    from modpipe import db
    db.migrate()
    yield db
    db.close()
