FROM apache/airflow:2.10.5

COPY requirements.txt /requirements.txt

RUN pip install --no-cache-dir -r /requirements.txt

COPY ml /opt/airflow/ml
COPY pipeline /opt/airflow/pipeline
COPY dags /opt/airflow/dags