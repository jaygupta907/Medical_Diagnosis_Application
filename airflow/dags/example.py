from datetime import datetime, timedelta
from airflow.models.dag import DAG
from airflow.operators.python import PythonOperator
import requests


default_args = {
    'owner': 'airflow',
    'retries': 1,
    'retry_delay': timedelta(minutes=2),
}

def call_app_api():
    response = requests.get("http://app:8000/retrain/")
    print(response.status_code, response.text)

with DAG(
    dag_id='test_hello_dag',
    default_args=default_args,
    description='A simple test DAG',
    schedule_interval='@daily',
    start_date=datetime(2023, 1, 1),
    catchup=False,
    tags=['test'],
) as dag:

    task = PythonOperator(
        task_id="call_app_service",
        python_callable=call_app_api,
    )

    task
