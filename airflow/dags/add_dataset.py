import sqlite3

def move_to_dataset(db_path='/opt/airflow/dataset'):
    """
    Move all entries from 'prediction' to 'tuned' table and remove the entries from the 'prediction' table 
    up to the last serial number before the fine-tuning process started.
    """
    try:
        # Connect to the SQLite database
        conn = sqlite3.connect(db_path)
        cursor = conn.cursor()

        # Get the last serial number of entries in the 'prediction' table before tuning
        cursor.execute("""
            SELECT MAX(serial_number) FROM prediction;
        """)
        last_serial_number = cursor.fetchone()[0]
        print(f"Last serial number before tuning: {last_serial_number}")

        # Move all entries from 'prediction' to 'tuned' table
        cursor.execute("""
            INSERT INTO tuned SELECT * FROM prediction;
        """)

        # Remove entries from 'prediction' table up to the last serial number recorded before tuning
        cursor.execute("""
            DELETE FROM prediction WHERE serial_number <= ?;
        """, (last_serial_number,))

        # Commit changes
        conn.commit()
        print(f"Entries with serial number up to {last_serial_number} removed from 'prediction' and moved to 'tuned'.")

    except Exception as e:
        print(f"Error moving entries to 'tuned' table: {e}")
    finally:
        if conn:
            cursor.close()
            conn.close()
