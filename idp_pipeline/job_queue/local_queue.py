"""
ORACLE-DESIGNED LOCAL QUEUE SYSTEM
==================================
SQLite-based queue that mimics AWS SQS functionality.
100% on-premise, no external dependencies.
"""

import sqlite3
import json
import uuid
import time
import threading
from pathlib import Path
from datetime import datetime, timedelta
from typing import Optional, Dict, List, Any, Callable
from dataclasses import dataclass, asdict
from enum import Enum
from contextlib import contextmanager

class JobStatus(Enum):
    """Job status states"""
    PENDING = "pending"
    PROCESSING = "processing"
    OCR_COMPLETE = "ocr_complete"
    ORGANIZING = "organizing"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"

class JobPriority(Enum):
    """Job priority levels"""
    LOW = 0
    NORMAL = 1
    HIGH = 2
    URGENT = 3

@dataclass
class Job:
    """Job in the queue"""
    job_id: str
    file_path: str
    file_name: str
    file_size: int
    file_hash: str

    status: JobStatus = JobStatus.PENDING
    priority: JobPriority = JobPriority.NORMAL

    # Model selection
    ocr_model: Optional[str] = None
    org_model: Optional[str] = None

    # Progress tracking
    progress: float = 0.0  # 0.0 - 1.0
    current_stage: str = "queued"
    pages_processed: int = 0
    total_pages: int = 1

    # Results
    ocr_result_path: Optional[str] = None
    final_result_path: Optional[str] = None
    confidence_score: Optional[float] = None

    # Timing
    created_at: str = ""
    started_at: Optional[str] = None
    completed_at: Optional[str] = None
    processing_time_seconds: Optional[float] = None

    # Error handling
    error_message: Optional[str] = None
    retry_count: int = 0
    max_retries: int = 3

    # Metadata
    metadata: Dict = None

    def __post_init__(self):
        if not self.created_at:
            self.created_at = datetime.utcnow().isoformat()
        if self.metadata is None:
            self.metadata = {}

    def to_dict(self) -> Dict:
        d = asdict(self)
        d['status'] = self.status.value
        d['priority'] = self.priority.value
        return d

    @classmethod
    def from_dict(cls, d: Dict) -> 'Job':
        d['status'] = JobStatus(d['status'])
        d['priority'] = JobPriority(d['priority'])
        return cls(**d)


class LocalQueue:
    """
    SQLite-based job queue with SQS-like functionality.
    Thread-safe, persistent, with visibility timeout support.
    """

    def __init__(self, db_path: str = "idp_queue.db"):
        self.db_path = Path(db_path)
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self._lock = threading.Lock()
        self._init_db()

    @contextmanager
    def _get_connection(self):
        """Thread-safe connection context manager"""
        conn = sqlite3.connect(str(self.db_path), timeout=30)
        conn.row_factory = sqlite3.Row
        try:
            yield conn
            conn.commit()
        except Exception as e:
            conn.rollback()
            raise e
        finally:
            conn.close()

    def _init_db(self):
        """Initialize database schema"""
        with self._get_connection() as conn:
            cursor = conn.cursor()

            # Jobs table
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS jobs (
                    job_id TEXT PRIMARY KEY,
                    file_path TEXT NOT NULL,
                    file_name TEXT NOT NULL,
                    file_size INTEGER,
                    file_hash TEXT,
                    status TEXT DEFAULT 'pending',
                    priority INTEGER DEFAULT 1,
                    ocr_model TEXT,
                    org_model TEXT,
                    progress REAL DEFAULT 0.0,
                    current_stage TEXT DEFAULT 'queued',
                    pages_processed INTEGER DEFAULT 0,
                    total_pages INTEGER DEFAULT 1,
                    ocr_result_path TEXT,
                    final_result_path TEXT,
                    confidence_score REAL,
                    created_at TEXT,
                    started_at TEXT,
                    completed_at TEXT,
                    processing_time_seconds REAL,
                    error_message TEXT,
                    retry_count INTEGER DEFAULT 0,
                    max_retries INTEGER DEFAULT 3,
                    metadata TEXT,
                    visibility_timeout TEXT
                )
            """)

            # Indexes for fast queries
            cursor.execute("CREATE INDEX IF NOT EXISTS idx_status ON jobs(status)")
            cursor.execute("CREATE INDEX IF NOT EXISTS idx_priority ON jobs(priority)")
            cursor.execute("CREATE INDEX IF NOT EXISTS idx_created ON jobs(created_at)")

            # Stats table
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS stats (
                    stat_date TEXT PRIMARY KEY,
                    jobs_submitted INTEGER DEFAULT 0,
                    jobs_completed INTEGER DEFAULT 0,
                    jobs_failed INTEGER DEFAULT 0,
                    total_pages_processed INTEGER DEFAULT 0,
                    avg_confidence REAL DEFAULT 0.0,
                    total_processing_seconds REAL DEFAULT 0.0
                )
            """)

            # Events table (for real-time updates)
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS events (
                    event_id INTEGER PRIMARY KEY AUTOINCREMENT,
                    job_id TEXT,
                    event_type TEXT,
                    event_data TEXT,
                    created_at TEXT DEFAULT CURRENT_TIMESTAMP
                )
            """)

    def submit(self, file_path: str, priority: JobPriority = JobPriority.NORMAL,
               metadata: Dict = None) -> Job:
        """Submit a new job to the queue"""
        import hashlib

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

        with self._get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute("""
                INSERT INTO jobs (
                    job_id, file_path, file_name, file_size, file_hash,
                    status, priority, created_at, metadata
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
            """, (
                job.job_id, job.file_path, job.file_name, job.file_size, job.file_hash,
                job.status.value, job.priority.value, job.created_at, json.dumps(job.metadata)
            ))

            # Update daily stats
            today = datetime.utcnow().date().isoformat()
            cursor.execute("""
                INSERT INTO stats (stat_date, jobs_submitted)
                VALUES (?, 1)
                ON CONFLICT(stat_date) DO UPDATE SET jobs_submitted = jobs_submitted + 1
            """, (today,))

            # Log event
            self._log_event(cursor, job.job_id, "submitted", {"file_name": job.file_name})

        return job

    def get_next(self, visibility_timeout_seconds: int = 300) -> Optional[Job]:
        """
        Get next job for processing (SQS-like behavior).
        Job becomes invisible to other workers for visibility_timeout.
        """
        with self._get_connection() as conn:
            cursor = conn.cursor()

            # Get highest priority pending job
            cursor.execute("""
                SELECT * FROM jobs
                WHERE status = 'pending'
                  AND (visibility_timeout IS NULL OR visibility_timeout < datetime('now'))
                ORDER BY priority DESC, created_at ASC
                LIMIT 1
            """)

            row = cursor.fetchone()
            if not row:
                return None

            # Set visibility timeout
            timeout = datetime.utcnow() + timedelta(seconds=visibility_timeout_seconds)
            cursor.execute("""
                UPDATE jobs
                SET status = 'processing',
                    started_at = ?,
                    visibility_timeout = ?
                WHERE job_id = ?
            """, (datetime.utcnow().isoformat(), timeout.isoformat(), row['job_id']))

            self._log_event(cursor, row['job_id'], "processing_started", {})

            return self._row_to_job(row)

    def update_progress(self, job_id: str, progress: float, stage: str,
                        pages_processed: int = None, extra: Dict = None):
        """Update job progress"""
        with self._get_connection() as conn:
            cursor = conn.cursor()

            updates = ["progress = ?", "current_stage = ?"]
            values = [progress, stage]

            if pages_processed is not None:
                updates.append("pages_processed = ?")
                values.append(pages_processed)

            values.append(job_id)

            cursor.execute(f"""
                UPDATE jobs SET {', '.join(updates)} WHERE job_id = ?
            """, values)

            self._log_event(cursor, job_id, "progress", {
                "progress": progress,
                "stage": stage,
                **(extra or {})
            })

    def complete_ocr(self, job_id: str, ocr_result_path: str, ocr_model: str):
        """Mark OCR stage as complete"""
        with self._get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute("""
                UPDATE jobs SET
                    status = 'ocr_complete',
                    ocr_result_path = ?,
                    ocr_model = ?,
                    progress = 0.5,
                    current_stage = 'organizing'
                WHERE job_id = ?
            """, (ocr_result_path, ocr_model, job_id))

            self._log_event(cursor, job_id, "ocr_complete", {"ocr_model": ocr_model})

    def complete(self, job_id: str, final_result_path: str, confidence_score: float,
                 org_model: str):
        """Mark job as fully completed"""
        now = datetime.utcnow().isoformat()

        with self._get_connection() as conn:
            cursor = conn.cursor()

            # Get started_at for duration calculation
            cursor.execute("SELECT started_at FROM jobs WHERE job_id = ?", (job_id,))
            row = cursor.fetchone()
            started = row['started_at'] if row else now

            # Calculate processing time
            start_dt = datetime.fromisoformat(started) if started else datetime.utcnow()
            processing_time = (datetime.utcnow() - start_dt).total_seconds()

            cursor.execute("""
                UPDATE jobs SET
                    status = 'completed',
                    final_result_path = ?,
                    confidence_score = ?,
                    org_model = ?,
                    progress = 1.0,
                    current_stage = 'completed',
                    completed_at = ?,
                    processing_time_seconds = ?,
                    visibility_timeout = NULL
                WHERE job_id = ?
            """, (final_result_path, confidence_score, org_model, now, processing_time, job_id))

            # Update daily stats
            today = datetime.utcnow().date().isoformat()
            cursor.execute("""
                INSERT INTO stats (stat_date, jobs_completed, avg_confidence, total_processing_seconds)
                VALUES (?, 1, ?, ?)
                ON CONFLICT(stat_date) DO UPDATE SET
                    jobs_completed = jobs_completed + 1,
                    avg_confidence = (avg_confidence * jobs_completed + ?) / (jobs_completed + 1),
                    total_processing_seconds = total_processing_seconds + ?
            """, (today, confidence_score, processing_time, confidence_score, processing_time))

            self._log_event(cursor, job_id, "completed", {
                "confidence": confidence_score,
                "processing_time": processing_time
            })

    def fail(self, job_id: str, error_message: str, retry: bool = True):
        """Mark job as failed, optionally retry"""
        with self._get_connection() as conn:
            cursor = conn.cursor()

            # Get current retry count
            cursor.execute("SELECT retry_count, max_retries FROM jobs WHERE job_id = ?", (job_id,))
            row = cursor.fetchone()

            if row and retry and row['retry_count'] < row['max_retries']:
                # Retry - put back in queue
                cursor.execute("""
                    UPDATE jobs SET
                        status = 'pending',
                        retry_count = retry_count + 1,
                        error_message = ?,
                        visibility_timeout = NULL,
                        current_stage = 'retry_queued'
                    WHERE job_id = ?
                """, (error_message, job_id))

                self._log_event(cursor, job_id, "retry", {
                    "error": error_message,
                    "attempt": row['retry_count'] + 1
                })
            else:
                # Final failure
                cursor.execute("""
                    UPDATE jobs SET
                        status = 'failed',
                        error_message = ?,
                        completed_at = ?,
                        current_stage = 'failed'
                    WHERE job_id = ?
                """, (error_message, datetime.utcnow().isoformat(), job_id))

                # Update stats
                today = datetime.utcnow().date().isoformat()
                cursor.execute("""
                    INSERT INTO stats (stat_date, jobs_failed)
                    VALUES (?, 1)
                    ON CONFLICT(stat_date) DO UPDATE SET jobs_failed = jobs_failed + 1
                """, (today,))

                self._log_event(cursor, job_id, "failed", {"error": error_message})

    def get_job(self, job_id: str) -> Optional[Job]:
        """Get job by ID"""
        with self._get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute("SELECT * FROM jobs WHERE job_id = ?", (job_id,))
            row = cursor.fetchone()
            return self._row_to_job(row) if row else None

    def get_jobs(self, status: JobStatus = None, limit: int = 100,
                 offset: int = 0) -> List[Job]:
        """Get jobs, optionally filtered by status"""
        with self._get_connection() as conn:
            cursor = conn.cursor()

            if status:
                cursor.execute("""
                    SELECT * FROM jobs WHERE status = ?
                    ORDER BY priority DESC, created_at DESC
                    LIMIT ? OFFSET ?
                """, (status.value, limit, offset))
            else:
                cursor.execute("""
                    SELECT * FROM jobs
                    ORDER BY priority DESC, created_at DESC
                    LIMIT ? OFFSET ?
                """, (limit, offset))

            return [self._row_to_job(row) for row in cursor.fetchall()]

    def get_stats(self, days: int = 7) -> Dict:
        """Get queue statistics"""
        with self._get_connection() as conn:
            cursor = conn.cursor()

            # Queue counts
            cursor.execute("""
                SELECT status, COUNT(*) as count FROM jobs GROUP BY status
            """)
            status_counts = {row['status']: row['count'] for row in cursor.fetchall()}

            # Daily stats
            cursor.execute("""
                SELECT * FROM stats
                WHERE stat_date >= date('now', ?)
                ORDER BY stat_date DESC
            """, (f'-{days} days',))
            daily = [dict(row) for row in cursor.fetchall()]

            # Recent events
            cursor.execute("""
                SELECT * FROM events
                ORDER BY created_at DESC LIMIT 20
            """)
            events = [dict(row) for row in cursor.fetchall()]

            # Calculate totals
            total_jobs = sum(status_counts.values())
            completed = status_counts.get('completed', 0)
            failed = status_counts.get('failed', 0)
            pending = status_counts.get('pending', 0)
            processing = status_counts.get('processing', 0)

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
                "daily_stats": daily,
                "recent_events": events
            }

    def get_events(self, job_id: str = None, limit: int = 50) -> List[Dict]:
        """Get events for real-time updates"""
        with self._get_connection() as conn:
            cursor = conn.cursor()

            if job_id:
                cursor.execute("""
                    SELECT * FROM events WHERE job_id = ?
                    ORDER BY created_at DESC LIMIT ?
                """, (job_id, limit))
            else:
                cursor.execute("""
                    SELECT * FROM events
                    ORDER BY created_at DESC LIMIT ?
                """, (limit,))

            return [dict(row) for row in cursor.fetchall()]

    def cancel(self, job_id: str):
        """Cancel a pending job"""
        with self._get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute("""
                UPDATE jobs SET status = 'cancelled', current_stage = 'cancelled'
                WHERE job_id = ? AND status = 'pending'
            """, (job_id,))
            self._log_event(cursor, job_id, "cancelled", {})

    def delete_job(self, job_id: str):
        """Permanently delete a job from the database"""
        with self._get_connection() as conn:
            cursor = conn.cursor()
            # Delete associated events first
            cursor.execute("DELETE FROM events WHERE job_id = ?", (job_id,))
            # Delete the job
            cursor.execute("DELETE FROM jobs WHERE job_id = ?", (job_id,))

    def cleanup(self, days_old: int = 30):
        """Remove old completed/failed jobs"""
        with self._get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute("""
                DELETE FROM jobs
                WHERE status IN ('completed', 'failed', 'cancelled')
                  AND created_at < datetime('now', ?)
            """, (f'-{days_old} days',))

            # Also cleanup old events
            cursor.execute("""
                DELETE FROM events WHERE created_at < datetime('now', ?)
            """, (f'-{days_old} days',))

    def _log_event(self, cursor, job_id: str, event_type: str, event_data: Dict):
        """Log an event"""
        cursor.execute("""
            INSERT INTO events (job_id, event_type, event_data)
            VALUES (?, ?, ?)
        """, (job_id, event_type, json.dumps(event_data)))

    def _row_to_job(self, row) -> Job:
        """Convert database row to Job object"""
        return Job(
            job_id=row['job_id'],
            file_path=row['file_path'],
            file_name=row['file_name'],
            file_size=row['file_size'],
            file_hash=row['file_hash'],
            status=JobStatus(row['status']),
            priority=JobPriority(row['priority']),
            ocr_model=row['ocr_model'],
            org_model=row['org_model'],
            progress=row['progress'],
            current_stage=row['current_stage'],
            pages_processed=row['pages_processed'],
            total_pages=row['total_pages'],
            ocr_result_path=row['ocr_result_path'],
            final_result_path=row['final_result_path'],
            confidence_score=row['confidence_score'],
            created_at=row['created_at'],
            started_at=row['started_at'],
            completed_at=row['completed_at'],
            processing_time_seconds=row['processing_time_seconds'],
            error_message=row['error_message'],
            retry_count=row['retry_count'],
            max_retries=row['max_retries'],
            metadata=json.loads(row['metadata']) if row['metadata'] else {}
        )


class QueueWorker:
    """
    Worker that processes jobs from the queue.
    Can run multiple instances for parallel processing.
    """

    def __init__(self, queue: LocalQueue, process_func: Callable[[Job], None],
                 worker_id: str = None):
        self.queue = queue
        self.process_func = process_func
        self.worker_id = worker_id or str(uuid.uuid4())[:8]
        self.running = False

    def start(self, poll_interval: float = 1.0):
        """Start processing jobs"""
        self.running = True
        print(f"Worker {self.worker_id} started")

        while self.running:
            job = self.queue.get_next()

            if job:
                print(f"Worker {self.worker_id} processing job {job.job_id}")
                try:
                    self.process_func(job)
                except Exception as e:
                    self.queue.fail(job.job_id, str(e))
                    print(f"Worker {self.worker_id} failed job {job.job_id}: {e}")
            else:
                time.sleep(poll_interval)

    def stop(self):
        """Stop the worker"""
        self.running = False
        print(f"Worker {self.worker_id} stopped")


# CLI interface
if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Local Queue Management")
    subparsers = parser.add_subparsers(dest="command")

    # Submit command
    submit_parser = subparsers.add_parser("submit", help="Submit a job")
    submit_parser.add_argument("file", help="File to process")
    submit_parser.add_argument("--priority", choices=["low", "normal", "high", "urgent"],
                               default="normal")

    # Status command
    status_parser = subparsers.add_parser("status", help="Show queue status")

    # List command
    list_parser = subparsers.add_parser("list", help="List jobs")
    list_parser.add_argument("--status", choices=["pending", "processing", "completed", "failed"])
    list_parser.add_argument("--limit", type=int, default=10)

    args = parser.parse_args()

    queue = LocalQueue()

    if args.command == "submit":
        priority_map = {
            "low": JobPriority.LOW,
            "normal": JobPriority.NORMAL,
            "high": JobPriority.HIGH,
            "urgent": JobPriority.URGENT
        }
        job = queue.submit(args.file, priority_map[args.priority])
        print(f"Job submitted: {job.job_id}")

    elif args.command == "status":
        stats = queue.get_stats()
        print(f"\nQueue Status:")
        print(f"  Pending: {stats['queue']['pending']}")
        print(f"  Processing: {stats['queue']['processing']}")
        print(f"  Completed: {stats['queue']['completed']}")
        print(f"  Failed: {stats['queue']['failed']}")
        print(f"  Success Rate: {stats['rates']['success_rate']:.1f}%")

    elif args.command == "list":
        status = JobStatus(args.status) if args.status else None
        jobs = queue.get_jobs(status, limit=args.limit)
        for job in jobs:
            print(f"{job.job_id[:8]}  {job.status.value:12}  {job.file_name}")
