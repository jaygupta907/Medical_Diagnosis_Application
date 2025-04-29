import sqlite3

def check_database(**kwargs):
    """
    Check if the number of entries in the database crosses the threshold.
    :return: Boolean indicating whether the threshold is crossed.
    """
    db_path = '/opt/airflow/uploads/predictions.db'  # Path to the SQLite database
    threshold = 100  # Set your threshold value here

    try:
        # Connect to the SQLite database
        conn = sqlite3.connect(db_path)
        cursor = conn.cursor()

        # Count rows in the 'prediction' table
        cursor.execute("SELECT COUNT(*) FROM prediction;")
        result = cursor.fetchone()
        cursor.close()
        conn.close()

        if result[0] > threshold:
            print(f"Threshold crossed: {result[0]} entries found. Proceeding with the API call.")
            return 'call_api_task'
        else:
            print(f"Threshold not crossed: {result[0]} entries. Skipping API call.")
            return 'skip_api_task'  
    except Exception as e:
        print(f"Error checking database: {e}")
        return 'skip_api_task'

