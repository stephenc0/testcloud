from fastapi import FastAPI, Response
from fastapi.responses import JSONResponse
from prometheus_client import Counter, Histogram, generate_latest, CONTENT_TYPE_LATEST
import random
import time
import os
import asyncio
from datetime import datetime
import numpy as np

app = FastAPI()

# Enhanced Prometheus metrics
REQUEST_COUNT = Counter('http_requests_total', 'Total HTTP requests', 
                       ['method', 'endpoint', 'status'])
REQUEST_DURATION = Histogram('http_request_duration_seconds',
                           'HTTP request duration in seconds',
                           ['method', 'endpoint', 'status'])

# Define error codes before conversion
ERROR_CODES = [400, 401, 403, 404, 500, 502, 503, 504]
# Pre-compute constants
ERROR_CODES = tuple(ERROR_CODES)  # Convert to tuple for faster lookup
BASE_LATENCY = float(os.getenv('BASE_LATENCY', '0.1'))
MU = np.log(BASE_LATENCY)
SIGMA = 0.05

# Pre-generate random number generator for better performance
RNG = np.random.default_rng()

async def simulate_latency():
    # Use pre-computed values and numpy's random generator
    latency = RNG.lognormal(MU, SIGMA)
    await asyncio.sleep(max(0, latency))

def record_metrics(method: str, endpoint: str, status_code: int, start_time: float):
    duration = time.time() - start_time
    REQUEST_COUNT.labels(method=method, endpoint=endpoint, status=status_code).inc()
    REQUEST_DURATION.labels(method=method, endpoint=endpoint, status=status_code).observe(duration)

@app.get("/success")
async def success():
    start_time = time.time()
    await simulate_latency()
    record_metrics('GET', '/success', 200, start_time)
    return JSONResponse(content={"status": "success"})

@app.get("/fail")
async def fail():
    start_time = time.time()
    await simulate_latency()
    status_code = int(RNG.choice(ERROR_CODES))  # Convert numpy.int64 to Python int
    record_metrics('GET', '/fail', status_code, start_time)
    return JSONResponse(
        content={"status": "error", "code": status_code},
        status_code=status_code
    )

@app.get("/metrics")
async def metrics():
    start_time = time.time()
    await simulate_latency()
    response = Response(
        content=generate_latest(),
        media_type=CONTENT_TYPE_LATEST
    )
    record_metrics('GET', '/metrics', 200, start_time)
    return response
