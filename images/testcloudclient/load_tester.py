import asyncio
import aiohttp
import logging
from typing import Dict, Optional
from datetime import datetime, time
import random
import math
import time as time_module  # Rename time module import to avoid conflict
from urllib.parse import urlparse

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class LoadTesterError(Exception):
    """Base exception for LoadTester errors"""
    pass

class LoadTester:
    def __init__(self, base_url: str, success_rate: float, requests_per_second: int, 
                 variance: float = 0.1, success_endpoint: str = '/success', 
                 fail_endpoint: str = '/fail', day_load_factor: float = 0.1,
                 week_load_factor: float = 0.1):
        # Validate URL
        try:
            parsed = urlparse(base_url)
            if not all([parsed.scheme, parsed.netloc]):
                raise ValueError("Invalid URL format")
            self.base_url = base_url
        except Exception as e:
            raise LoadTesterError(f"Invalid base_url: {str(e)}")

        # Validate variance (0-1 range)
        if not 0 <= variance <= 1:
            raise LoadTesterError(f"Invalid variance: must be between 0 and 1")
        
        # Validate load factors (-1 to 1 range)
        for factor, name in [(day_load_factor, 'day'), (week_load_factor, 'week')]:
            if not -1 <= factor <= 1:
                raise LoadTesterError(f"Invalid {name}_load_factor: must be between -1 and 1")

        self.success_rate = success_rate / 100
        self.variance = variance  # Remove min/max since we validate above
        self.base_requests = requests_per_second
        self.day_load_factor = day_load_factor
        self.week_load_factor = week_load_factor
        self.success_endpoint = success_endpoint
        self.fail_endpoint = fail_endpoint
        self.stats: Dict[str, int] = {"success": 0, "fail": 0, "errors": 0}
        self.last_was_success = None

        # Cache business hours times
        self.business_start = time(11, 0)
        self.business_end = time(13, 0)
        self.ramp_hours = 10  # Hours to ramp up/down

        logger.info(f"LoadTester initialized: day_factor={day_load_factor}, week_factor={week_load_factor}")

        # Pre-compute URLs
        self.success_url = f"{base_url}{success_endpoint}"
        self.fail_url = f"{base_url}{fail_endpoint}"
        
        # Pre-compute time constants
        self.business_start_hours = 11.0  # 11:00
        self.business_end_hours = 13.0    # 13:00
        self.ramp_start = self.business_start_hours - self.ramp_hours
        self.ramp_end = self.business_end_hours + self.ramp_hours
        
        # Pre-compute sigmoid ranges
        self.sigmoid_scale = 12.0
        self.sigmoid_offset = 6.0
        
        # Use sets for faster lookups
        self.weekday_set = frozenset([0, 1, 2, 3, 4])  # Monday-Friday
        self.peak_days = frozenset([1, 2, 3])          # Tuesday-Thursday
        
        # Cache for time-based calculations (refreshed every second)
        self._cache_time = 0
        self._cache = {}

    def sigmoid(self, x: float) -> float:
        """Sigmoid function for smooth transitions"""
        return 1 / (1 + math.exp(-x))

    def get_time_factor(self, current_hour: float) -> float:
        """Optimized time-based factor calculation"""
        current_hour = current_hour % 24
        
        if self.business_start_hours <= current_hour <= self.business_end_hours:
            return 1.0
        
        if current_hour < self.ramp_start or current_hour > self.ramp_end:
            return 0.0
            
        if current_hour < self.business_start_hours:
            x = self.sigmoid_scale * (current_hour - self.ramp_start) / self.ramp_hours - self.sigmoid_offset
        else:
            x = self.sigmoid_scale * (self.ramp_end - current_hour) / self.ramp_hours - self.sigmoid_offset
        
        return self.sigmoid(x)

    def get_week_factor(self, weekday: int, _: float) -> float:
        """Optimized weekday factor calculation"""
        if weekday not in self.weekday_set:
            return 0.0
        return 0.3 if weekday in self.peak_days else 0.15

    def apply_load_factor(self, base_rate: float, time_factor: float, week_factor: float) -> float:
        # Use weighted average where load factors determine the weight of time/week factors
        time_weight = self.day_load_factor
        week_weight = self.week_load_factor
        base_weight = 1 - (time_weight + week_weight)
        
        if base_weight < 0:
            # Normalize weights if they sum to > 1
            total = time_weight + week_weight
            time_weight /= total
            """Calculate current requests per second with seasonality"""
            base_weight = 0
        
        final_rate = base_rate * (
            base_weight + 
            time_weight * (1 + time_factor) +
            week_weight * (1 + week_factor)
        )
        return final_rate

    def get_current_request_rate(self) -> int:
        """Cached request rate calculation"""
        current_time = time_module.time()
        if current_time - self._cache_time >= 1.0:
            now = datetime.now()
            current_hour = now.hour + now.minute / 60
            weekday = now.weekday()
            
            time_factor = self.get_time_factor(current_hour)
            week_factor = self.get_week_factor(weekday, current_hour)
            
            current_rate = self.apply_load_factor(self.base_requests, time_factor, week_factor)
            
            self._cache = {
                'rate': max(1, round(current_rate)),
                'time_factor': time_factor,
                'week_factor': week_factor
            }
            self._cache_time = current_time
            
            # Batch logging to reduce I/O
            logger.info(
                f"Factors - Time: {time_factor:.2f}, Week: {week_factor:.2f} | "
                f"Load factors - Day: {self.day_load_factor}, Week: {self.week_load_factor}"
            )
            
        return self._cache['rate']

    def should_succeed(self) -> bool:
        """Simple success rate check with variance"""
        if self.last_was_success is None or random.random() > self.variance:
            should_succeed = random.random() < self.success_rate
        else:
            should_succeed = self.last_was_success
        
        self.last_was_success = should_succeed
        return should_succeed

    async def make_request(self, session: aiohttp.ClientSession) -> None:
        """Optimized request handling"""
        url = self.success_url if self.should_succeed() else self.fail_url
        endpoint = '/success' if url == self.success_url else '/fail'
        
        try:
            async with session.get(url, timeout=30) as response:
                self.stats[endpoint.strip("/")] += 1
        except Exception as e:
            self.stats["errors"] += 1
            logger.error(f"Request failed: url={url}, error={str(e)}")
            
    async def run(self) -> None:
        """Optimized task handling"""
        logger.info(f"Starting load test against {self.base_url}")
        
        timeout = aiohttp.ClientTimeout(
            total=1,        # Total timeout
            connect=0.2,    # Connection timeout
            sock_read=0.2   # Socket read timeout
        )
        
        async with aiohttp.ClientSession(timeout=timeout) as session:
            pending_tasks = set()
            
            while True:
                start_time = time_module.time()  # Use time_module instead of time
                current_requests = self.get_current_request_rate()
                
                # Clean up completed tasks
                pending_tasks = {task for task in pending_tasks if not task.done()}
                
                # Create new tasks
                new_tasks = {
                    asyncio.create_task(self.make_request(session))
                    for _ in range(current_requests)
                }
                pending_tasks.update(new_tasks)
                
                # Log stats
                logger.info(
                    f"Stats: {self.stats} | Target RPS: {current_requests} | "
                    f"Active tasks: {len(pending_tasks)}"
                )
                
                await asyncio.sleep(max(0, 1 - (time_module.time() - start_time)))

def main():
    import argparse
    import os
    
    parser = argparse.ArgumentParser(description='Load tester for fake API')
    parser.add_argument('--base-url', default=os.getenv('BASE_URL', 'http://myfakeapi:8080'),
                      help='Base URL of the fake API')
    parser.add_argument('--success-rate', type=float, default=float(os.getenv('SUCCESS_RATE', '80')),
                      help='Percentage of requests that should succeed (0-100)')
    parser.add_argument('--requests-per-second', type=int, default=int(os.getenv('REQUESTS_PER_SECOND', '10')),
                      help='Number of requests to generate per second')
    parser.add_argument('--variance', type=float, default=float(os.getenv('VARIANCE', '0.1')),
                      help='Variance factor for request grouping (0-1). Higher values mean more grouped requests.')
    parser.add_argument('--day-load-factor', type=float, 
                       default=float(os.getenv('DAY_LOAD_FACTOR', '0.1')),
                       help='Daily load variation factor (0-1)')
    parser.add_argument('--week-load-factor', type=float, 
                       default=float(os.getenv('WEEK_LOAD_FACTOR', '0.1')),
                       help='Weekly load variation factor (0-1)')
    parser.add_argument('--success-endpoint', default=os.getenv('SUCCESS_ENDPOINT', '/success'),
                      help='Success endpoint path')
    parser.add_argument('--fail-endpoint', default=os.getenv('FAIL_ENDPOINT', '/fail'),  # Fixed extra parenthesis
                      help='Fail endpoint path')
    
    args = parser.parse_args()
    
    tester = LoadTester(
        args.base_url,
        args.success_rate,
        args.requests_per_second,
        args.variance,
        args.success_endpoint,
        args.fail_endpoint,
        args.day_load_factor,
        args.week_load_factor
    )
    asyncio.run(tester.run())

if __name__ == "__main__":
    main()
