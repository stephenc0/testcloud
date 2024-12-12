FROM --platform=linux/amd64 python:3.9-slim

WORKDIR /app

COPY requirements.txt .
RUN pip install -r requirements.txt

COPY load_tester.py .

CMD ["python", "load_tester.py"]


