FROM --platform=linux/amd64 python:3.9-slim

WORKDIR /app
COPY load_tester.py .

RUN pip install aiohttp numpy

ENV TZ=America/New_York
RUN ln -snf /usr/share/zoneinfo/$TZ /etc/localtime && echo $TZ > /etc/timezone

ENV BASE_URL=http://myfakeapi:8082
ENV SUCCESS_RATE=95
ENV REQUESTS_PER_SECOND=10
ENV VARIANCE=0.1
ENV SUCCESS_ENDPOINT=/success
ENV FAIL_ENDPOINT=/fail

ENTRYPOINT ["python", "load_tester.py"]
CMD ["--base-url", "http://myfakeapi:8082", \
     "--success-rate", "95", \
     "--requests-per-second", "10", \
     "--variance", "0.1", \
     "--success-endpoint", "/success", \
     "--fail-endpoint", "/fail"]


