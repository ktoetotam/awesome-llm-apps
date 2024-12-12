import psycopg
from psycopg import sql

# Connect as a superuser (e.g., the 'postgres' role)
conn = psycopg.connect(
    dbname="ai",
    user="ai",
    password="ai",
    host="127.0.0.1",
    port="5532"
)
conn.autocommit = True  # Ensure immediate commit so CREATE DATABASE works properly

try:
    with conn.cursor() as cur:
        # Create the 'ai' schema if it does not exist
        cur.execute("CREATE SCHEMA IF NOT EXISTS ai;")

        # Create the 'recipes' table in the 'ai' schema
        cur.execute("""
        CREATE TABLE IF NOT EXISTS ai.data (
            id SERIAL PRIMARY KEY,
            name VARCHAR(255) NOT NULL,
            meta_data JSONB,
            content TEXT NOT NULL,
            embedding VECTOR(300),  -- Adjust the dimension as needed
            usage INTEGER,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        );
        """)

    print("Table created successfully.")

except psycopg.Error as e:
    print(f"Error: {e}")

finally:
    conn.close()