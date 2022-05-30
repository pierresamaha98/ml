FROM python:3.8-slim-buster
RUN apt-get update && \
    apt-get install -y --no-install-recommends libgomp1 \
    && rm -rf /var/lib/apt/lists/*
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt
COPY app.py Custom_transformer.py /usr/app/
EXPOSE 5000
WORKDIR /usr/app/
CMD python3 app.py