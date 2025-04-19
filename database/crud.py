import psycopg2
from pgvector.psycopg2 import register_vector
from config import DB_CONFIG
from typing import Optional
def register_user(username, pin, full_name=None):
    try:
        conn = psycopg2.connect("dbname=postgres user=postgres password=sakeena123 host=localhost port=5432")
        cursor = conn.cursor()
        
        # Check if username already exists
        cursor.execute("SELECT id FROM users WHERE username = %s", (username,))
        if cursor.fetchone():
            raise ValueError("Username already exists")
        
        # Insert new user
        cursor.execute(
            "INSERT INTO users (username, pin, full_name) VALUES (%s, %s, %s) RETURNING id",
            (username, pin, full_name)
        )
        user_id = cursor.fetchone()[0]
        conn.commit()
        return user_id
    
    except Exception as e:
        conn.rollback()
        raise e
    finally:
        if conn:
            conn.close()



def save_face_embedding(user_id: int, embedding) -> None:
    """Save or update user's face embedding."""
    try:
        with psycopg2.connect(**DB_CONFIG) as conn:
            register_vector(conn)
            with conn.cursor() as cursor:
                cursor.execute("""
                    INSERT INTO face_embeddings (user_id, embedding)
                    VALUES (%s, %s)
                    ON CONFLICT (user_id) DO UPDATE
                    SET embedding = EXCLUDED.embedding;
                """, (user_id, embedding))
    except Exception as e:
        raise RuntimeError(f"Failed to save face embedding: {e}")

def find_face_match(live_embedding: list, threshold: float = 0.6) -> Optional[int]:
    """Find the closest face match in the database."""
    with psycopg2.connect(**DB_CONFIG) as conn:
        register_vector(conn)
        with conn.cursor() as cursor:
            cursor.execute("""
                SELECT user_id, embedding <-> %s AS distance
                FROM face_embeddings
                ORDER BY distance
                LIMIT 1;
            """, (live_embedding,))
            result = cursor.fetchone()
            if result and result[1] < threshold:
                return result[0]
            return None

def authenticate_pin(user_id: int, pin: str) -> bool:
    """Validate a user's PIN."""
    with psycopg2.connect(**DB_CONFIG) as conn:
        with conn.cursor() as cursor:
            cursor.execute("""
                SELECT user_id FROM users
                WHERE user_id = %s AND pin_hash = crypt(%s, pin_hash);
            """, (user_id, pin))
            return cursor.fetchone() is not None

def log_access_attempt(user_id: int, success: bool, method: str = "Face") -> None:
    """Log access attempts to the database."""
    with psycopg2.connect(**DB_CONFIG) as conn:
        with conn.cursor() as cursor:
            cursor.execute("""
                INSERT INTO access_logs (user_id, method, is_success)
                VALUES (%s, %s, %s);
            """, (user_id, method, success))
