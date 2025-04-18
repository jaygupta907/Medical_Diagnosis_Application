from datetime import datetime, timedelta
from airflow.models.dag import DAG
from airflow.operators.python import PythonOperator

def print_hello():
    print("👋 Hello from Airflow!")

def print_date():
    print(f"📅 Current date is: {datetime.now()}")

default_args = {
    'owner': 'airflow',
    'retries': 1,
    'retry_delay': timedelta(minutes=2),
}

with DAG(
    dag_id='test_hello_dag',
    default_args=default_args,
    description='A simple test DAG',
    schedule_interval='@daily',
    start_date=datetime(2023, 1, 1),
    catchup=False,
    tags=['test'],
) as dag:

    task_hello = PythonOperator(
        task_id='say_hello',
        python_callable=print_hello,
    )

    task_date = PythonOperator(
        task_id='show_date',
        python_callable=print_date,
    )

    task_hello >> task_date
