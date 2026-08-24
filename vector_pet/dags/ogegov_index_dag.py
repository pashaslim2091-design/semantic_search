from airflow import DAG
from airflow.operators.python import PythonOperator
from datetime import datetime, timedelta
import sys
import os

# Добавляем путь к src, чтобы импортировать наши модули
sys.path.append('/opt/airflow/src')  # внутри контейнера airflow папка src монтируется

from parser import parse_ogegov
from embedder import index_articles

default_args = {
    'owner': 'airflow',
    'depends_on_past': False,
    'start_date': datetime(2025, 1, 1),
    'retries': 1,
    'retry_delay': timedelta(minutes=1),
}

dag = DAG(
    'ogegov_index',
    default_args=default_args,
    description='Индексация словаря Ожегова в Qdrant',
    schedule_interval='@once',  # запускаем только один раз (можно изменить)
    catchup=False,
)

parse_task = PythonOperator(
    task_id='parse_articles',
    python_callable=parse_ogegov,
    op_args=['/opt/airflow/data/ogegov.txt', '/opt/airflow/data/articles.json'],
    dag=dag,
)

index_task = PythonOperator(
    task_id='index_articles',
    python_callable=index_articles,
    op_args=['/opt/airflow/data/articles.json', 'ogegov', 'qdrant', 6333],
    dag=dag,
)

parse_task >> index_task
