import time
import threading
from typing import Callable, Optional
from datetime import datetime
import re


class Scheduler:
    def __init__(self):
        self.jobs = {}
        self._running = False
        self._thread: Optional[threading.Thread] = None

    def add_job(self, job_id: str, func: Callable, cron_expr: str = None,
                interval_seconds: int = None):
        job = {
            'func': func,
            'cron_expr': cron_expr,
            'interval_seconds': interval_seconds,
            'last_run': None,
            'next_run': None,
        }
        self.jobs[job_id] = job
        self._schedule_next(job)

    def remove_job(self, job_id: str):
        if job_id in self.jobs:
            del self.jobs[job_id]

    def _schedule_next(self, job: dict):
        if job['cron_expr']:
            job['next_run'] = self._cron_to_next_time(job['cron_expr'])
        elif job['interval_seconds']:
            if job['last_run']:
                job['next_run'] = job['last_run'] + job['interval_seconds']
            else:
                job['next_run'] = time.time() + job['interval_seconds']

    def _cron_to_next_time(self, cron_expr: str) -> float:
        parts = cron_expr.strip().split()
        if len(parts) != 5:
            return time.time() + 3600

        minute, hour, day, month, day_of_week = parts

        now = datetime.now()
        next_time = now.replace(second=0, microsecond=0)

        from datetime import timedelta
        next_time += timedelta(minutes=1)

        for _ in range(60 * 24 * 365):
            if self._matches_cron(next_time, minute, hour, day, month, day_of_week):
                return next_time.timestamp()
            next_time += timedelta(minutes=1)

        return time.time() + 3600

    def _matches_cron(self, dt: datetime, minute: str, hour: str, day: str,
                      month: str, day_of_week: str) -> bool:
        if not self._matches_field(dt.minute, minute, 0, 59):
            return False
        if not self._matches_field(dt.hour, hour, 0, 23):
            return False
        if not self._matches_field(dt.day, day, 1, 31):
            return False
        if not self._matches_field(dt.month, month, 1, 12):
            return False
        if not self._matches_field(dt.weekday(), day_of_week, 0, 6):
            return False
        return True

    def _matches_field(self, value: int, expr: str, min_val: int, max_val: int) -> bool:
        if expr == '*':
            return True

        if ',' in expr:
            return any(self._matches_field(value, p, min_val, max_val) for p in expr.split(','))

        if '-' in expr:
            start, end = expr.split('-')
            return min_val <= int(start) <= value <= int(end) <= max_val

        if expr.startswith('*/'):
            step = int(expr[2:])
            return value % step == 0

        try:
            return value == int(expr)
        except ValueError:
            return False

    def start(self):
        if self._running:
            return

        self._running = True
        self._thread = threading.Thread(target=self._run, daemon=True)
        self._thread.start()

    def stop(self):
        self._running = False
        if self._thread:
            self._thread.join(timeout=5)

    def _run(self):
        while self._running:
            now = time.time()

            for job_id, job in self.jobs.items():
                if job['next_run'] and now >= job['next_run']:
                    try:
                        job['func']()
                    except Exception as e:
                        print(f"Job {job_id} 执行失败: {e}")
                    finally:
                        job['last_run'] = now
                        self._schedule_next(job)

            time.sleep(1)

    def run_now(self, job_id: str):
        if job_id in self.jobs:
            job = self.jobs[job_id]
            try:
                job['func']()
                job['last_run'] = time.time()
                self._schedule_next(job)
            except Exception as e:
                print(f"Job {job_id} 执行失败: {e}")

    def get_job_status(self) -> dict:
        status = {}
        for job_id, job in self.jobs.items():
            status[job_id] = {
                'last_run': datetime.fromtimestamp(job['last_run']).strftime('%Y-%m-%d %H:%M:%S') if job['last_run'] else None,
                'next_run': datetime.fromtimestamp(job['next_run']).strftime('%Y-%m-%d %H:%M:%S') if job['next_run'] else None,
                'cron_expr': job['cron_expr'],
                'interval_seconds': job['interval_seconds'],
            }
        return status


class IncrementalUpdater:
    def __init__(self, recommender_system, interval_seconds: int = 3600):
        self.recommender_system = recommender_system
        self.interval_seconds = interval_seconds
        self.last_update = None

    def update(self):
        try:
            self.recommender_system.incremental_update()
            self.last_update = time.time()
            print(f"增量更新完成: {datetime.now()}")
        except Exception as e:
            print(f"增量更新失败: {e}")

    def retrain(self):
        try:
            self.recommender_system.retrain()
            self.last_update = time.time()
            print(f"全量训练完成: {datetime.now()}")
        except Exception as e:
            print(f"全量训练失败: {e}")
