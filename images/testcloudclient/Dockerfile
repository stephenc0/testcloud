FROM --platform=linux/amd64 python:3.9-slim

WORKDIR /app
COPY load_tester.py .

RUN pip install aiohttp numpy

CMD ["python", "load_tester.py"]


