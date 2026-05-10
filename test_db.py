from backend.db.database import engine
from sqlalchemy import text

def test_connection():
    print("Attempting to connect to Supabase Transaction Pooler...")
    try:
        with engine.connect() as connection:
            result = connection.execute(text("SELECT version();"))
            row = result.fetchone()
            print("---")
            print("SUCCESS: Connection successful!")
            print(f"Server version: {row[0]}")
            print("---")
    except Exception as e:
        print("---")
        print("ERROR: Connection failed!")
        print(f"Details: {e}")
        print("---")
        print("Checklist:")
        print("1. Is your password correct in .env?")
        print("2. Is the Transaction Pooler Host correct?")
        print("3. Is your internet connection active?")
        print("---")

if __name__ == "__main__":
    test_connection()
