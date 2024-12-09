FROM --platform=linux/amd64 python:3.9-slim

WORKDIR /app

RUN pip install aiohttp numpy

COPY load_tester.py .

CMD ["python", "load_tester.py"]


