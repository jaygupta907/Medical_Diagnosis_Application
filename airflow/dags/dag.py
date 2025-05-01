from datetime import datetime, timedelta
from airflow.models.dag import DAG
from airflow.operators.python import PythonOperator
from airflow.operators.python import BranchPythonOperator
from airflow.utils.dates import days_ago
import requests  # To make HTTP requests to the FastAPI endpoint


# Now you can import the modules
from check_database import check_database
from call_api import call_app_api
from move_to_dataset import move_to_dataset


# Default arguments for the DAG
default_args = {
    'owner': 'airflow',
    'retries': 1,
    'retry_delay': timedelta(minutes=2),
}

# Function to call the localhost FastAPI endpoint for DVC and Git operations
def version_dataset():
    try:
        # Localhost URL for your FastAPI endpoint (update this if needed)
        url = "http://host.docker.internal:9200/perform_dvc_git_operations"
        
        
        # Making a POST request to the FastAPI endpoint
        response = requests.post(url)
        
        # Check if the request was successful
        if response.status_code == 200:
            print("Successfully triggered the local FastAPI endpoint.")
        else:
            print(f"Failed to trigger FastAPI endpoint. Status code: {response.status_code}")
    except Exception as e:
        print(f"An error occurred while calling the FastAPI endpoint: {str(e)}")

# Airflow DAG definition
with DAG(
    dag_id='finetuning',  # Unique DAG ID
    default_args=default_args,  # Default arguments for retries, etc.
    description='A simple test DAG to check database threshold and call API', 
    # schedule_interval=timedelta(minutes=10),  # Scheduling interval for the DAG 
    schedule_interval='@daily',  # Set to None for manual triggering
    start_date=datetime(2023, 1, 1),  # Start date of the DAG
    catchup=False,  # Skip past runs if the DAG was paused
    tags=['test'],  # Tags to organize and search for the DAG
) as dag:

    branch_task = BranchPythonOperator(
        task_id='branch_task',
        python_callable=check_database,
        provide_context=True,
    )

    call_api_task = PythonOperator(
        task_id='call_api_task',
        python_callable=call_app_api,
    )

    skip_api_task = PythonOperator(
        task_id='skip_api_task',
        python_callable=lambda: print("API call skipped due to threshold not being crossed."),
    )

    move_data_to_dataset = PythonOperator(
        task_id='move_to_dataset_task',
        python_callable=move_to_dataset,
    )

    versioning_task = PythonOperator( 
        task_id='versioning_task',
        python_callable=version_dataset,
        )

    # Setting up dependencies
    branch_task >> [call_api_task, skip_api_task]
    call_api_task >> move_data_to_dataset >> versioning_task
