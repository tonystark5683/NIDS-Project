FROM python:3.7-slim


COPY requirements.txt /
RUN pip install --no-cache-dir -r /requirements.txt
COPY . /opt/project/

ENTRYPOINT ["python3", "/opt/project/project.py"]
