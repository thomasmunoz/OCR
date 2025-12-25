"""
ORACLE-DESIGNED REDIS QUEUE
============================
Distributed queue for multi-container scaling.
Drop-in replacement for SQLite queue.
"""

import os
import json
import uuid
import time
from datetime import datetime, timedelta
from typing import Optional, Dict, List, Any, Callable
from dataclasses import dataclass, asdict
import logging
import threading

try:
    import redis
    REDIS_AVAILABLE = True
except ImportError:
    REDIS_AVAILABLE = False

from job_queue.local_queue import Job, JobStatus, JobPriority

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class RedisQueue:
    """
    Redis-based distributed job queue.
    Supports multiple workers across containers.
    """

    # Redis keys
    QUEUE_KEY = "idp:queue"           # Sorted set (priority queue)
    JOBS_KEY = "idp:jobs"             # Hash (job details)
    PROCESSING_KEY = "idp:processing" # Set (jobs being processed)
    STATS_KEY = "idp:stats"           # Hash (statistics)
    EVENTS_KEY = "idp:events"         # Stream (real-time events)
    LOCKS_KEY = "idp:locks"           # For distributed locking

    def __init__(self, redis_url: str = None):
        if not REDIS_AVAILABLE:
            raise ImportError("redis package not installed. Run: pip install redis")

        self.redis_url = redis_url or os.getenv("IDP_REDIS_URL", "redis://localhost:6379/0")
        self.redis = redis.from_url(self.redis_url, decode_responses=True)
        self._lock = threading.Lock()

        # Test connection
        try:
            self.redis.ping()
            logger.info(f"Connected to Redis: {self.redis_url}")
        except redis.ConnectionError as e:
            raise ConnectionError(f"Cannot connect to Redis: {e}")

    def submit(self, file_path: str, priority: JobPriority = JobPriority.NORMAL,
               metadata: Dict = None) -> Job:
        """Submit a new job to the queue"""
        import hashlib
        from pathlib import Path

        path = Path(file_path)
        if not path.exists():
            raise FileNotFoundError(f"File not found: {file_path}")

        # Calculate file hash
        with open(file_path, 'rb') as f:
            file_hash = hashlib.sha256(f.read()).hexdigest()

        job = Job(
            job_id=str(uuid.uuid4()),
            file_path=str(path.absolute()),
            file_name=path.name,
            file_size=path.stat().st_size,
            file_hash=file_hash,
            priority=priority,
            metadata=metadata or {}
        )

        # Store job data
        job_data = job.to_dict()
        self.redis.hset(self.JOBS_KEY, job.job_id, json.dumps(job_data))

        # Add to priority queue (higher priority = higher score)
        # Score = priority * 1000000000 + (MAX_TIME - timestamp) for FIFO within priority
        score = priority.value * 1_000_000_000 + (2_000_000_000 - int(time.time()))
        self.redis.zadd(self.QUEUE_KEY, {job.job_id: score})

        # Update stats
        self.redis.hincrby(self.STATS_KEY, "jobs_submitted", 1)
        self.redis.hincrby(self.STATS_KEY, f"submitted_{datetime.utcnow().date()}", 1)

        # Log event
        self._log_event(job.job_id, "submitted", {"file_name": job.file_name})

        logger.info(f"Job submitted: {job.job_id}")
        return job

    def get_next(self, visibility_timeout_seconds: int = 300,
                 worker_id: str = None) -> Optional[Job]:
        """
        Get next job for processing (atomic operation).
        Uses WATCH/MULTI for race condition prevention.
        """
        worker_id = worker_id or str(uuid.uuid4())[:8]

        with self.redis.pipeline() as pipe:
            try:
                # Watch the queue for changes
                pipe.watch(self.QUEUE_KEY)

                # Get highest priority job
                result = self.redis.zrevrange(self.QUEUE_KEY, 0, 0)
                if not result:
                    pipe.unwatch()
                    return None

                job_id = result[0]

                # Check if already being processed
                if self.redis.sismember(self.PROCESSING_KEY, job_id):
                    pipe.unwatch()
                    return self.get_next(visibility_timeout_seconds, worker_id)

                # Atomic transaction
                pipe.multi()
                pipe.zrem(self.QUEUE_KEY, job_id)
                pipe.sadd(self.PROCESSING_KEY, job_id)
                pipe.execute()

                # Get job data
                job_data = self.redis.hget(self.JOBS_KEY, job_id)
                if not job_data:
                    self.redis.srem(self.PROCESSING_KEY, job_id)
                    return None

                job = Job.from_dict(json.loads(job_data))
                job.status = JobStatus.PROCESSING
                job.started_at = datetime.utcnow().isoformat()

                # Update job
                self.redis.hset(self.JOBS_KEY, job_id, json.dumps(job.to_dict()))

                # Set visibility timeout (auto-release if worker crashes)
                self.redis.setex(
                    f"{self.LOCKS_KEY}:{job_id}",
                    visibility_timeout_seconds,
                    worker_id
                )

                self._log_event(job_id, "processing_started", {"worker": worker_id})
                return job

            except redis.WatchError:
                # Another worker got the job, try again
                return self.get_next(visibility_timeout_seconds, worker_id)

    def update_progress(self, job_id: str, progress: float, stage: str,
                        pages_processed: int = None, extra: Dict = None):
        """Update job progress"""
        job_data = self.redis.hget(self.JOBS_KEY, job_id)
        if not job_data:
            return

        job = Job.from_dict(json.loads(job_data))
        job.progress = progress
        job.current_stage = stage
        if pages_processed is not None:
            job.pages_processed = pages_processed

        self.redis.hset(self.JOBS_KEY, job_id, json.dumps(job.to_dict()))

        # Refresh lock
        lock_key = f"{self.LOCKS_KEY}:{job_id}"
        if self.redis.exists(lock_key):
            self.redis.expire(lock_key, 300)

        self._log_event(job_id, "progress", {
            "progress": progress,
            "stage": stage,
            **(extra or {})
        })

    def complete_ocr(self, job_id: str, ocr_result_path: str, ocr_model: str):
        """Mark OCR stage as complete"""
        job_data = self.redis.hget(self.JOBS_KEY, job_id)
        if not job_data:
            return

        job = Job.from_dict(json.loads(job_data))
        job.status = JobStatus.OCR_COMPLETE
        job.ocr_result_path = ocr_result_path
        job.ocr_model = ocr_model
        job.progress = 0.5
        job.current_stage = "organizing"

        self.redis.hset(self.JOBS_KEY, job_id, json.dumps(job.to_dict()))
        self._log_event(job_id, "ocr_complete", {"ocr_model": ocr_model})

    def complete(self, job_id: str, final_result_path: str, confidence_score: float,
                 org_model: str):
        """Mark job as fully completed"""
        job_data = self.redis.hget(self.JOBS_KEY, job_id)
        if not job_data:
            return

        job = Job.from_dict(json.loads(job_data))

        # Calculate processing time
        started = job.started_at
        if started:
            start_dt = datetime.fromisoformat(started)
            processing_time = (datetime.utcnow() - start_dt).total_seconds()
        else:
            processing_time = 0

        job.status = JobStatus.COMPLETED
        job.final_result_path = final_result_path
        job.confidence_score = confidence_score
        job.org_model = org_model
        job.progress = 1.0
        job.current_stage = "completed"
        job.completed_at = datetime.utcnow().isoformat()
        job.processing_time_seconds = processing_time

        # Update job
        self.redis.hset(self.JOBS_KEY, job_id, json.dumps(job.to_dict()))

        # Remove from processing
        self.redis.srem(self.PROCESSING_KEY, job_id)
        self.redis.delete(f"{self.LOCKS_KEY}:{job_id}")

        # Update stats
        self.redis.hincrby(self.STATS_KEY, "jobs_completed", 1)
        self.redis.hincrbyfloat(self.STATS_KEY, "total_processing_time", processing_time)

        self._log_event(job_id, "completed", {
            "confidence": confidence_score,
            "processing_time": processing_time
        })

        logger.info(f"Job completed: {job_id}")

    def fail(self, job_id: str, error_message: str, retry: bool = True):
        """Mark job as failed, optionally retry"""
        job_data = self.redis.hget(self.JOBS_KEY, job_id)
        if not job_data:
            return

        job = Job.from_dict(json.loads(job_data))

        if retry and job.retry_count < job.max_retries:
            # Retry - put back in queue
            job.retry_count += 1
            job.status = JobStatus.PENDING
            job.error_message = error_message
            job.current_stage = "retry_queued"

            self.redis.hset(self.JOBS_KEY, job_id, json.dumps(job.to_dict()))

            # Re-add to queue with same priority
            score = job.priority.value * 1_000_000_000 + (2_000_000_000 - int(time.time()))
            self.redis.zadd(self.QUEUE_KEY, {job_id: score})

            self._log_event(job_id, "retry", {
                "error": error_message,
                "attempt": job.retry_count
            })
        else:
            # Final failure
            job.status = JobStatus.FAILED
            job.error_message = error_message
            job.completed_at = datetime.utcnow().isoformat()
            job.current_stage = "failed"

            self.redis.hset(self.JOBS_KEY, job_id, json.dumps(job.to_dict()))
            self.redis.hincrby(self.STATS_KEY, "jobs_failed", 1)

            self._log_event(job_id, "failed", {"error": error_message})

        # Remove from processing
        self.redis.srem(self.PROCESSING_KEY, job_id)
        self.redis.delete(f"{self.LOCKS_KEY}:{job_id}")

    def get_job(self, job_id: str) -> Optional[Job]:
        """Get job by ID"""
        job_data = self.redis.hget(self.JOBS_KEY, job_id)
        if not job_data:
            return None
        return Job.from_dict(json.loads(job_data))

    def get_jobs(self, status: JobStatus = None, limit: int = 100,
                 offset: int = 0) -> List[Job]:
        """Get jobs, optionally filtered by status"""
        # Get all job IDs
        all_job_ids = self.redis.hkeys(self.JOBS_KEY)

        jobs = []
        for job_id in all_job_ids:
            job_data = self.redis.hget(self.JOBS_KEY, job_id)
            if job_data:
                job = Job.from_dict(json.loads(job_data))
                if status is None or job.status == status:
                    jobs.append(job)

        # Sort by priority and created_at
        jobs.sort(key=lambda j: (j.priority.value, j.created_at), reverse=True)

        return jobs[offset:offset + limit]

    def get_stats(self, days: int = 7) -> Dict:
        """Get queue statistics"""
        stats = self.redis.hgetall(self.STATS_KEY)

        # Count by status
        all_jobs = self.get_jobs(limit=10000)
        status_counts = {}
        for job in all_jobs:
            status_counts[job.status.value] = status_counts.get(job.status.value, 0) + 1

        pending = self.redis.zcard(self.QUEUE_KEY)
        processing = self.redis.scard(self.PROCESSING_KEY)

        total_jobs = len(all_jobs)
        completed = status_counts.get('completed', 0)
        failed = status_counts.get('failed', 0)

        # Get recent events
        events = self.get_events(limit=20)

        return {
            "queue": {
                "total_jobs": total_jobs,
                "pending": pending,
                "processing": processing,
                "completed": completed,
                "failed": failed
            },
            "rates": {
                "success_rate": completed / max(1, completed + failed) * 100,
                "failure_rate": failed / max(1, completed + failed) * 100
            },
            "totals": {
                "submitted": int(stats.get("jobs_submitted", 0)),
                "completed": int(stats.get("jobs_completed", 0)),
                "failed": int(stats.get("jobs_failed", 0)),
                "total_processing_time": float(stats.get("total_processing_time", 0))
            },
            "recent_events": events
        }

    def get_events(self, job_id: str = None, limit: int = 50) -> List[Dict]:
        """Get events from stream"""
        try:
            # Read from stream
            events = self.redis.xrevrange(self.EVENTS_KEY, count=limit)
            result = []
            for event_id, data in events:
                if job_id is None or data.get("job_id") == job_id:
                    result.append({
                        "event_id": event_id,
                        "job_id": data.get("job_id"),
                        "event_type": data.get("event_type"),
                        "event_data": json.loads(data.get("event_data", "{}")),
                        "created_at": data.get("created_at")
                    })
            return result
        except:
            return []

    def cancel(self, job_id: str):
        """Cancel a pending job"""
        self.redis.zrem(self.QUEUE_KEY, job_id)

        job_data = self.redis.hget(self.JOBS_KEY, job_id)
        if job_data:
            job = Job.from_dict(json.loads(job_data))
            job.status = JobStatus.CANCELLED
            job.current_stage = "cancelled"
            self.redis.hset(self.JOBS_KEY, job_id, json.dumps(job.to_dict()))

        self._log_event(job_id, "cancelled", {})

    def delete_job(self, job_id: str):
        """Permanently delete a job from Redis"""
        # Remove from queue
        self.redis.zrem(self.QUEUE_KEY, job_id)
        # Remove from processing set
        self.redis.srem(self.PROCESSING_KEY, job_id)
        # Remove job data
        self.redis.hdel(self.JOBS_KEY, job_id)
        # Remove any locks
        self.redis.delete(f"{self.LOCKS_KEY}:{job_id}")

    def cleanup(self, days_old: int = 30):
        """Remove old completed/failed jobs"""
        cutoff = datetime.utcnow() - timedelta(days=days_old)

        all_jobs = self.get_jobs(limit=100000)
        for job in all_jobs:
            if job.status in [JobStatus.COMPLETED, JobStatus.FAILED, JobStatus.CANCELLED]:
                if job.created_at:
                    created = datetime.fromisoformat(job.created_at)
                    if created < cutoff:
                        self.redis.hdel(self.JOBS_KEY, job.job_id)

        # Trim events stream
        self.redis.xtrim(self.EVENTS_KEY, maxlen=10000, approximate=True)

    def _log_event(self, job_id: str, event_type: str, event_data: Dict):
        """Log event to stream"""
        try:
            self.redis.xadd(self.EVENTS_KEY, {
                "job_id": job_id,
                "event_type": event_type,
                "event_data": json.dumps(event_data),
                "created_at": datetime.utcnow().isoformat()
            }, maxlen=10000)
        except:
            pass  # Non-critical

    def health_check(self) -> Dict:
        """Check Redis health"""
        try:
            self.redis.ping()
            info = self.redis.info()
            return {
                "status": "healthy",
                "connected_clients": info.get("connected_clients"),
                "used_memory": info.get("used_memory_human"),
                "uptime_days": info.get("uptime_in_days")
            }
        except Exception as e:
            return {"status": "unhealthy", "error": str(e)}


def get_queue(queue_type: str = None):
    """Factory function to get appropriate queue"""
    queue_type = queue_type or os.getenv("IDP_QUEUE_TYPE", "sqlite")

    if queue_type == "redis":
        return RedisQueue()
    else:
        from job_queue.local_queue import LocalQueue
        return LocalQueue()
