import os
import psycopg2


def get_db_connection():
    """Create a connection to the PostgreSQL database."""
    try:
        connection = psycopg2.connect(
            dsn=os.getenv("DATABASE_URL"),
        )
        return connection
    except Exception as e:
        print(f"Error connecting to the database: {e}")
        return None

