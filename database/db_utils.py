# database/db_utils.py
import psycopg2
from config import Config
import os

def get_db_connection():
    db_url = os.getenv("DATABASE_URL")
    conn = psycopg2.connect(db_url)
    conn.row_factory = psycopg2.Row
    return conn

def save_face_hash(user_id: str, face_hash: str):
    conn = get_db_connection()
    conn.execute("""
        INSERT OR REPLACE INTO users (id, face_hash)
        VALUES (?, ?)
        """, (user_id, face_hash))
    conn.commit()
    conn.close()

def get_all_hashes():
    conn = get_db_connection()
    hashes = conn.execute("SELECT id, face_hash FROM users").fetchall()
    conn.close()
    return hashes

def initialize_db():
    conn = get_db_connection()
    conn.execute("""
        CREATE TABLE IF NOT EXISTS users (
            id TEXT PRIMARY KEY,
            face_hash TEXT NOT NULL
        )
    """)
    conn.commit()
    conn.close()