import os
from sqlalchemy import create_engine

def get_db_connection():
    # Loaded from environment variables
    db_url = os.getenv("DB_URL")
    engine = create_engine(db_url)
    return engine
