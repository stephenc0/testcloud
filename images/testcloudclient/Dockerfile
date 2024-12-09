FROM --platform=linux/amd64 python:3.9-slim

WORKDIR /app
COPY load_tester.py .

RUN pip install aiohttp numpy

ENV BASE_URL=http://myfakeapi:8082
ENV SUCCESS_RATE=95
ENV REQUESTS_PER_SECOND=10
ENV VARIANCE=0.1
ENV SUCCESS_ENDPOINT=/success
ENV FAIL_ENDPOINT=/fail

ENTRYPOINT ["python"]
CMD ["load_tester.py", \
     "--base-url", "$BASE_URL", \
     "--success-rate", "$SUCCESS_RATE", \
     "--requests-per-second", "$REQUESTS_PER_SECOND", \
     "--variance", "$VARIANCE", \
     "--success-endpoint", "$SUCCESS_ENDPOINT", \
     "--fail-endpoint", "$FAIL_ENDPOINT"]


