"""
ORACLE-DESIGNED WEB API SERVER
==============================
FastAPI server with real-time WebSocket updates.
Serves the dashboard UI and handles file uploads.
"""

import os
import sys
import json
import uuid
import asyncio
from pathlib import Path
from typing import Dict, List, Optional
from datetime import datetime
import logging

from fastapi import FastAPI, File, UploadFile, HTTPException, BackgroundTasks, WebSocket, WebSocketDisconnect
from fastapi.responses import HTMLResponse, JSONResponse, FileResponse
from fastapi.staticfiles import StaticFiles
from fastapi.middleware.cors import CORSMiddleware
import uvicorn

sys.path.insert(0, str(Path(__file__).parent.parent))

from job_queue.local_queue import JobStatus, JobPriority
from job_queue.redis_queue import get_queue
from api.pipeline import IDPPipeline

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Initialize
app = FastAPI(
    title="IDP Pipeline API",
    description="Intelligent Document Processing - 100% On-Premise",
    version="1.0.0"
)

# CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Paths - use environment variables for Docker, fallback to local
BASE_DIR = Path(__file__).parent.parent
UPLOAD_DIR = Path(os.getenv("IDP_UPLOAD_DIR", str(BASE_DIR / "uploads")))
OUTPUT_DIR = Path(os.getenv("IDP_OUTPUT_DIR", str(BASE_DIR / "output")))
UPLOAD_DIR.mkdir(parents=True, exist_ok=True)
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# Pipeline instance
pipeline = IDPPipeline(
    queue_db=str(BASE_DIR / "idp_queue.db"),
    output_dir=str(OUTPUT_DIR),
    available_vram_gb=16.0,
    priority="balanced"
)

# WebSocket connections
active_connections: List[WebSocket] = []


async def broadcast_update(message: Dict):
    """Broadcast update to all connected clients"""
    for connection in active_connections:
        try:
            await connection.send_json(message)
        except:
            pass


@app.get("/", response_class=HTMLResponse)
async def dashboard():
    """Serve the dashboard UI"""
    return get_dashboard_html()


@app.get("/favicon.ico")
async def favicon():
    """Serve favicon"""
    favicon_path = BASE_DIR / "static" / "favicon.ico"
    if favicon_path.exists():
        return FileResponse(str(favicon_path), media_type="image/x-icon")
    raise HTTPException(404, "Favicon not found")


@app.get("/health")
async def health_check():
    """Health check endpoint for Docker/Kubernetes"""
    try:
        # Check queue health
        queue_health = {"status": "healthy"}
        if hasattr(pipeline.queue, 'health_check'):
            queue_health = pipeline.queue.health_check()

        return {
            "status": "healthy",
            "service": "idp-api",
            "timestamp": datetime.utcnow().isoformat(),
            "queue": queue_health,
            "version": "1.0.0"
        }
    except Exception as e:
        return JSONResponse(
            status_code=503,
            content={"status": "unhealthy", "error": str(e)}
        )


@app.get("/ready")
async def readiness_check():
    """Readiness probe for Kubernetes"""
    return {"ready": True}


@app.post("/api/upload")
async def upload_file(
    file: UploadFile = File(...),
    priority: str = "normal",
    background_tasks: BackgroundTasks = None
):
    """Upload a file for processing"""
    # Save uploaded file
    file_id = str(uuid.uuid4())[:8]
    ext = Path(file.filename).suffix
    saved_path = UPLOAD_DIR / f"{file_id}{ext}"

    with open(saved_path, "wb") as f:
        content = await file.read()
        f.write(content)

    # Submit to pipeline
    job = pipeline.submit(
        str(saved_path),
        priority,
        metadata={"original_filename": file.filename}
    )

    await broadcast_update({
        "type": "job_submitted",
        "job_id": job.job_id,
        "file_name": file.filename
    })

    return {
        "job_id": job.job_id,
        "file_name": file.filename,
        "status": "submitted"
    }


@app.get("/api/files/{job_id}")
async def serve_file(job_id: str):
    """Serve the uploaded file for viewing inline in browser"""
    from fastapi.responses import FileResponse

    job = pipeline.queue.get_job(job_id)
    if not job:
        raise HTTPException(404, "Job not found")

    file_path = Path(job.file_path)
    if not file_path.exists():
        raise HTTPException(404, "File not found")

    # Get original filename from metadata
    original_filename = job.metadata.get("original_filename", job.file_name) if job.metadata else job.file_name

    # Determine content type
    suffix = file_path.suffix.lower()
    content_types = {
        '.pdf': 'application/pdf',
        '.png': 'image/png',
        '.jpg': 'image/jpeg',
        '.jpeg': 'image/jpeg',
        '.tiff': 'image/tiff',
        '.tif': 'image/tiff',
        '.bmp': 'image/bmp'
    }
    content_type = content_types.get(suffix, 'application/octet-stream')

    # Return file with inline disposition to display in browser
    return FileResponse(
        path=str(file_path),
        media_type=content_type,
        headers={
            "Content-Disposition": f"inline; filename=\"{original_filename}\""
        }
    )


@app.get("/api/jobs")
async def list_jobs(status: str = None, limit: int = 50):
    """List jobs"""
    status_enum = JobStatus(status) if status else None
    jobs = pipeline.queue.get_jobs(status_enum, limit)
    return [job.to_dict() for job in jobs]


@app.get("/api/jobs/{job_id}")
async def get_job(job_id: str):
    """Get job details"""
    job = pipeline.queue.get_job(job_id)
    if not job:
        raise HTTPException(404, "Job not found")
    return job.to_dict()


@app.get("/api/jobs/{job_id}/result")
async def get_result(job_id: str):
    """Get job result"""
    job = pipeline.queue.get_job(job_id)
    if not job:
        raise HTTPException(404, "Job not found")
    if job.status != JobStatus.COMPLETED:
        raise HTTPException(400, f"Job not completed: {job.status.value}")

    if job.final_result_path and Path(job.final_result_path).exists():
        with open(job.final_result_path, 'r') as f:
            return json.load(f)

    raise HTTPException(404, "Result file not found")


@app.get("/api/jobs/{job_id}/ocr-text")
async def get_ocr_text(job_id: str):
    """Get raw OCR text from job"""
    job = pipeline.queue.get_job(job_id)
    if not job:
        raise HTTPException(404, "Job not found")

    # Check if OCR stage is complete
    allowed_statuses = [JobStatus.COMPLETED, JobStatus.PROCESSING]
    if job.status not in allowed_statuses and job.current_stage not in ["ocr_complete", "organizing", "completed"]:
        if not job.ocr_result_path:
            raise HTTPException(400, f"OCR not complete yet. Status: {job.status.value}, Stage: {job.current_stage}")

    # Try to get OCR result file
    if job.ocr_result_path and Path(job.ocr_result_path).exists():
        with open(job.ocr_result_path, 'r') as f:
            ocr_data = json.load(f)
            # Extract raw text from OCR result
            raw_text = ""
            if isinstance(ocr_data, dict):
                # Check common field names for text content
                if "full_text" in ocr_data:
                    raw_text = ocr_data["full_text"]
                elif "text" in ocr_data:
                    raw_text = ocr_data["text"]
                elif "raw_text" in ocr_data:
                    raw_text = ocr_data["raw_text"]
                elif "pages" in ocr_data:
                    # Multi-page document - extract raw_text from each page
                    for page in ocr_data["pages"]:
                        if isinstance(page, dict):
                            page_text = page.get("raw_text", page.get("text", ""))
                            if page_text:
                                raw_text += page_text + "\n\n"
                        elif isinstance(page, str):
                            raw_text += page + "\n\n"
                else:
                    # Fallback: dump as JSON
                    raw_text = json.dumps(ocr_data, indent=2)

                # Get model from metadata if available
                ocr_model = job.ocr_model or ocr_data.get("metadata", {}).get("ocr_model_used", "unknown")
            elif isinstance(ocr_data, str):
                raw_text = ocr_data
                ocr_model = job.ocr_model or "unknown"
            else:
                raw_text = str(ocr_data)
                ocr_model = job.ocr_model or "unknown"

            return {
                "job_id": job_id,
                "ocr_model": ocr_model,
                "raw_text": raw_text.strip(),
                "file_name": job.file_name
            }

    # Fallback: try to get from final result
    if job.final_result_path and Path(job.final_result_path).exists():
        with open(job.final_result_path, 'r') as f:
            final_data = json.load(f)
            raw_text = final_data.get("raw_text", "") or final_data.get("text", "")
            if not raw_text and "pages" in final_data:
                for page in final_data["pages"]:
                    if isinstance(page, dict):
                        raw_text += page.get("text", "") + "\n\n"
            return {
                "job_id": job_id,
                "ocr_model": job.ocr_model or "unknown",
                "raw_text": raw_text.strip() if raw_text else "No raw text available",
                "file_name": job.file_name
            }

    raise HTTPException(404, "OCR result not found")


@app.delete("/api/jobs/{job_id}")
async def cancel_job(job_id: str):
    """Cancel a job"""
    pipeline.queue.cancel(job_id)
    return {"status": "cancelled"}


@app.get("/api/stats")
async def get_stats():
    """Get queue statistics"""
    return pipeline.queue.get_stats()


@app.get("/api/status")
async def get_status():
    """Get pipeline status"""
    return pipeline.get_status()


@app.get("/api/models")
async def get_models():
    """Get available models"""
    from config.models_config import OCR_MODELS, ORGANIZATION_MODELS

    return {
        "ocr_models": {k: {"name": v.name, "repo": v.hf_repo, "vram": v.vram_required_gb}
                       for k, v in OCR_MODELS.items()},
        "organization_models": {k: {"name": v.name, "repo": v.hf_repo, "vram": v.vram_required_gb}
                                for k, v in ORGANIZATION_MODELS.items()}
    }


@app.delete("/api/cleanup")
async def cleanup_failed_jobs():
    """
    Delete all incomplete and failed jobs along with their associated files.
    Cleans up: queue entries, uploaded files, output files, temp files.
    """
    import shutil
    import tempfile

    cleaned = {
        "jobs_removed": 0,
        "upload_files_deleted": 0,
        "output_files_deleted": 0,
        "temp_files_deleted": 0,
        "bytes_freed": 0,
        "errors": []
    }

    try:
        # Get all failed and incomplete jobs
        failed_statuses = [JobStatus.FAILED, JobStatus.CANCELLED]
        incomplete_statuses = [JobStatus.PENDING, JobStatus.PROCESSING, JobStatus.OCR_COMPLETE]

        jobs_to_clean = []
        for status in failed_statuses + incomplete_statuses:
            try:
                status_jobs = pipeline.queue.get_jobs(status, limit=1000)
                jobs_to_clean.extend(status_jobs)
            except Exception as e:
                cleaned["errors"].append(f"Error getting {status.value} jobs: {str(e)}")

        for job in jobs_to_clean:
            try:
                # Delete uploaded file
                if job.file_path and Path(job.file_path).exists():
                    file_size = Path(job.file_path).stat().st_size
                    Path(job.file_path).unlink()
                    cleaned["upload_files_deleted"] += 1
                    cleaned["bytes_freed"] += file_size

                # Delete OCR result file
                if job.ocr_result_path and Path(job.ocr_result_path).exists():
                    file_size = Path(job.ocr_result_path).stat().st_size
                    Path(job.ocr_result_path).unlink()
                    cleaned["output_files_deleted"] += 1
                    cleaned["bytes_freed"] += file_size

                # Delete final result file
                if job.final_result_path and Path(job.final_result_path).exists():
                    file_size = Path(job.final_result_path).stat().st_size
                    Path(job.final_result_path).unlink()
                    cleaned["output_files_deleted"] += 1
                    cleaned["bytes_freed"] += file_size

                # Delete compact result file if exists
                compact_path = OUTPUT_DIR / f"{job.job_id}_compact.json"
                if compact_path.exists():
                    file_size = compact_path.stat().st_size
                    compact_path.unlink()
                    cleaned["output_files_deleted"] += 1
                    cleaned["bytes_freed"] += file_size

                # Remove job from queue/database
                if hasattr(pipeline.queue, 'delete_job'):
                    pipeline.queue.delete_job(job.job_id)
                else:
                    # Fallback: mark as cancelled so cleanup() can remove it
                    pipeline.queue.cancel(job.job_id)

                cleaned["jobs_removed"] += 1

            except Exception as e:
                cleaned["errors"].append(f"Error cleaning job {job.job_id}: {str(e)}")

        # Clean temp directory for OCR page images
        temp_dir = Path(tempfile.gettempdir())
        for temp_file in temp_dir.glob("ocr_page_*"):
            try:
                file_size = temp_file.stat().st_size
                temp_file.unlink()
                cleaned["temp_files_deleted"] += 1
                cleaned["bytes_freed"] += file_size
            except Exception as e:
                cleaned["errors"].append(f"Error deleting temp file {temp_file}: {str(e)}")

        # Trigger queue cleanup for old records
        if hasattr(pipeline.queue, 'cleanup'):
            pipeline.queue.cleanup(days_old=0)  # Clean all cancelled/failed immediately

        # Broadcast update
        await broadcast_update({
            "type": "cleanup_complete",
            "jobs_removed": cleaned["jobs_removed"],
            "files_deleted": cleaned["upload_files_deleted"] + cleaned["output_files_deleted"]
        })

        cleaned["bytes_freed_mb"] = round(cleaned["bytes_freed"] / (1024 * 1024), 2)
        return cleaned

    except Exception as e:
        logger.error(f"Cleanup failed: {e}")
        raise HTTPException(500, f"Cleanup failed: {str(e)}")


@app.post("/api/process/start")
async def start_processing(background_tasks: BackgroundTasks):
    """Start background processing"""
    if pipeline.is_processing:
        return {"status": "already_running"}

    async def process_loop():
        while True:
            try:
                job = pipeline.queue.get_next()
                if job:
                    await broadcast_update({
                        "type": "job_started",
                        "job_id": job.job_id
                    })

                    def progress_cb(p, msg):
                        asyncio.create_task(broadcast_update({
                            "type": "progress",
                            "job_id": job.job_id,
                            "progress": p,
                            "message": msg
                        }))

                    result = pipeline.process_job(job, progress_cb)

                    await broadcast_update({
                        "type": "job_completed",
                        "job_id": job.job_id,
                        "confidence": result.overall_confidence
                    })
                else:
                    await asyncio.sleep(2)
            except Exception as e:
                logger.error(f"Processing error: {e}")
                await asyncio.sleep(2)

    background_tasks.add_task(process_loop)
    return {"status": "started"}


@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket):
    """WebSocket for real-time updates"""
    await websocket.accept()
    active_connections.append(websocket)

    try:
        while True:
            # Keep connection alive, receive any client messages
            data = await websocket.receive_text()
            # Echo back stats periodically if requested
            if data == "stats":
                stats = pipeline.queue.get_stats()
                await websocket.send_json({"type": "stats", "data": stats})
    except WebSocketDisconnect:
        active_connections.remove(websocket)


def get_dashboard_html() -> str:
    """Generate the dashboard HTML"""
    return '''<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>IDP Pipeline Dashboard</title>
    <link rel="icon" type="image/x-icon" href="/favicon.ico">
    <style>
        * { margin: 0; padding: 0; box-sizing: border-box; }
        body {
            font-family: -apple-system, BlinkMacSystemFont, 'SF Pro Display', sans-serif;
            background: linear-gradient(135deg, #0f0f1a 0%, #1a1a2e 50%, #16213e 100%);
            min-height: 100vh;
            color: #e0e0e0;
        }
        .container { max-width: 1400px; margin: 0 auto; padding: 2rem; }

        header {
            display: flex;
            justify-content: space-between;
            align-items: center;
            margin-bottom: 2rem;
            padding: 1.5rem;
            background: rgba(255,255,255,0.05);
            border-radius: 16px;
            border: 1px solid rgba(255,255,255,0.1);
        }
        .logo {
            font-size: 1.8rem;
            font-weight: 700;
            background: linear-gradient(135deg, #00d4ff, #7c3aed);
            -webkit-background-clip: text;
            -webkit-text-fill-color: transparent;
        }
        .status-badge {
            padding: 0.5rem 1rem;
            border-radius: 20px;
            font-size: 0.9rem;
            font-weight: 600;
        }
        .status-online { background: #10b981; color: white; }
        .status-offline { background: #ef4444; color: white; }

        .grid { display: grid; grid-template-columns: 1fr 1fr; gap: 1.5rem; margin-bottom: 2rem; }
        @media (max-width: 900px) { .grid { grid-template-columns: 1fr; } }

        .card {
            background: rgba(255,255,255,0.05);
            border-radius: 16px;
            padding: 1.5rem;
            border: 1px solid rgba(255,255,255,0.1);
        }
        .card h2 {
            font-size: 1.1rem;
            margin-bottom: 1rem;
            color: #a0a0a0;
            text-transform: uppercase;
            letter-spacing: 1px;
        }

        /* Upload Area */
        .upload-area {
            border: 2px dashed rgba(124,58,237,0.5);
            border-radius: 12px;
            padding: 3rem;
            text-align: center;
            cursor: pointer;
            transition: all 0.3s;
        }
        .upload-area:hover, .upload-area.dragover {
            border-color: #7c3aed;
            background: rgba(124,58,237,0.1);
        }
        .upload-icon { font-size: 3rem; margin-bottom: 1rem; }
        .upload-text { color: #a0a0a0; }
        .upload-text strong { color: #7c3aed; }
        #fileInput { display: none; }

        /* Stats */
        .stats-grid {
            display: grid;
            grid-template-columns: repeat(4, 1fr);
            gap: 1rem;
        }
        .stat-item {
            background: rgba(0,0,0,0.2);
            padding: 1rem;
            border-radius: 10px;
            text-align: center;
        }
        .stat-value {
            font-size: 2rem;
            font-weight: 700;
            background: linear-gradient(135deg, #00d4ff, #7c3aed);
            -webkit-background-clip: text;
            -webkit-text-fill-color: transparent;
        }
        .stat-label { color: #6b7280; font-size: 0.85rem; margin-top: 0.3rem; }

        /* Jobs Table */
        .jobs-table {
            width: 100%;
            border-collapse: collapse;
        }
        .jobs-table th, .jobs-table td {
            padding: 1rem;
            text-align: left;
            border-bottom: 1px solid rgba(255,255,255,0.1);
        }
        .jobs-table th { color: #6b7280; font-weight: 500; font-size: 0.85rem; text-transform: uppercase; }
        .jobs-table tr:hover { background: rgba(255,255,255,0.02); }

        .badge {
            padding: 0.3rem 0.8rem;
            border-radius: 20px;
            font-size: 0.8rem;
            font-weight: 600;
        }
        .badge-pending { background: #fbbf24; color: #1a1a2e; }
        .badge-processing { background: #3b82f6; color: white; }
        .badge-completed { background: #10b981; color: white; }
        .badge-failed { background: #ef4444; color: white; }
        .badge-ocr_complete { background: #8b5cf6; color: white; }
        .badge-cancelled { background: #6b7280; color: white; }

        /* Model tags */
        .model-tag {
            display: inline-block;
            padding: 0.2rem 0.5rem;
            border-radius: 4px;
            font-size: 0.7rem;
            font-weight: 500;
            margin-top: 0.3rem;
            background: rgba(124,58,237,0.2);
            color: #a78bfa;
            border: 1px solid rgba(124,58,237,0.3);
        }
        .model-tag.ocr { background: rgba(59,130,246,0.2); color: #93c5fd; border-color: rgba(59,130,246,0.3); }
        .model-tag.org { background: rgba(16,185,129,0.2); color: #6ee7b7; border-color: rgba(16,185,129,0.3); }

        /* Progress percentage */
        .progress-container {
            display: flex;
            align-items: center;
            gap: 0.5rem;
        }
        .progress-percent {
            font-size: 0.85rem;
            font-weight: 600;
            color: #7c3aed;
            min-width: 40px;
        }

        /* Progress Bar */
        .progress-bar {
            height: 6px;
            background: rgba(255,255,255,0.1);
            border-radius: 3px;
            overflow: hidden;
        }
        .progress-fill {
            height: 100%;
            background: linear-gradient(90deg, #7c3aed, #00d4ff);
            transition: width 0.3s;
        }

        /* Actions */
        .btn {
            padding: 0.6rem 1.2rem;
            border: none;
            border-radius: 8px;
            cursor: pointer;
            font-weight: 600;
            transition: all 0.3s;
        }
        .btn-primary {
            background: linear-gradient(135deg, #7c3aed, #2563eb);
            color: white;
        }
        .btn-primary:hover { transform: translateY(-2px); box-shadow: 0 4px 12px rgba(124,58,237,0.4); }
        .btn-sm { padding: 0.4rem 0.8rem; font-size: 0.85rem; }
        .btn-outline {
            background: transparent;
            border: 1px solid #7c3aed;
            color: #7c3aed;
        }
        .btn-danger {
            background: linear-gradient(135deg, #ef4444, #dc2626);
            color: white;
        }
        .btn-danger:hover { transform: translateY(-2px); box-shadow: 0 4px 12px rgba(239,68,68,0.4); }

        /* Toast */
        .toast {
            position: fixed;
            bottom: 2rem;
            right: 2rem;
            padding: 1rem 1.5rem;
            background: #1a1a2e;
            border: 1px solid rgba(255,255,255,0.1);
            border-radius: 10px;
            box-shadow: 0 10px 40px rgba(0,0,0,0.5);
            transform: translateY(100px);
            opacity: 0;
            transition: all 0.3s;
        }
        .toast.show { transform: translateY(0); opacity: 1; }
        .toast-success { border-left: 4px solid #10b981; }
        .toast-error { border-left: 4px solid #ef4444; }

        /* Connection indicator */
        .ws-status {
            width: 10px;
            height: 10px;
            border-radius: 50%;
            display: inline-block;
            margin-right: 0.5rem;
        }
        .ws-connected { background: #10b981; }
        .ws-disconnected { background: #ef4444; }
    </style>
</head>
<body>
    <div class="container">
        <header>
            <div class="logo">IDP Pipeline</div>
            <div>
                <span class="ws-status" id="wsStatus"></span>
                <span id="wsStatusText">Connecting...</span>
            </div>
        </header>

        <div class="grid">
            <!-- Upload Card -->
            <div class="card">
                <h2>Upload Document</h2>
                <div class="upload-area" id="uploadArea" onclick="document.getElementById('fileInput').click()">
                    <div class="upload-icon">📄</div>
                    <p class="upload-text">Drag & drop or <strong>click to upload</strong></p>
                    <p style="color:#6b7280; font-size:0.85rem; margin-top:0.5rem;">PDF, PNG, JPG, TIFF supported</p>
                </div>
                <input type="file" id="fileInput" accept=".pdf,.png,.jpg,.jpeg,.tiff,.tif,.bmp" onchange="uploadFile(this.files[0])">
            </div>

            <!-- Stats Card -->
            <div class="card">
                <h2>Queue Statistics</h2>
                <div class="stats-grid">
                    <div class="stat-item">
                        <div class="stat-value" id="statPending">0</div>
                        <div class="stat-label">Pending</div>
                    </div>
                    <div class="stat-item">
                        <div class="stat-value" id="statProcessing">0</div>
                        <div class="stat-label">Processing</div>
                    </div>
                    <div class="stat-item">
                        <div class="stat-value" id="statCompleted">0</div>
                        <div class="stat-label">Completed</div>
                    </div>
                    <div class="stat-item">
                        <div class="stat-value" id="statFailed">0</div>
                        <div class="stat-label">Failed</div>
                    </div>
                </div>
                <div style="margin-top:1rem; display: flex; gap: 0.5rem; flex-wrap: wrap;">
                    <button class="btn btn-primary" onclick="startProcessing()">Start Processing</button>
                    <button class="btn btn-outline" onclick="refreshStats()">Refresh</button>
                    <button class="btn btn-danger" onclick="cleanupJobs()" title="Delete failed, cancelled, and incomplete jobs">
                        🗑️ Cleanup Failed/Incomplete
                    </button>
                </div>
            </div>
        </div>

        <!-- Jobs Table -->
        <div class="card">
            <h2>Recent Jobs</h2>
            <table class="jobs-table">
                <thead>
                    <tr>
                        <th>ID</th>
                        <th>File</th>
                        <th>Status</th>
                        <th>Model</th>
                        <th>Progress</th>
                        <th>Confidence</th>
                        <th>Actions</th>
                    </tr>
                </thead>
                <tbody id="jobsTable">
                    <tr><td colspan="7" style="text-align:center;color:#6b7280;">No jobs yet</td></tr>
                </tbody>
            </table>
        </div>
    </div>

    <div class="toast" id="toast"></div>

    <script>
        let ws = null;
        let jobs = [];

        // WebSocket connection
        function connectWebSocket() {
            const protocol = window.location.protocol === 'https:' ? 'wss:' : 'ws:';
            ws = new WebSocket(`${protocol}//${window.location.host}/ws`);

            ws.onopen = () => {
                document.getElementById('wsStatus').className = 'ws-status ws-connected';
                document.getElementById('wsStatusText').textContent = 'Connected';
            };

            ws.onclose = () => {
                document.getElementById('wsStatus').className = 'ws-status ws-disconnected';
                document.getElementById('wsStatusText').textContent = 'Disconnected';
                setTimeout(connectWebSocket, 3000);
            };

            ws.onmessage = (event) => {
                const data = JSON.parse(event.data);
                handleWSMessage(data);
            };
        }

        function handleWSMessage(data) {
            switch(data.type) {
                case 'job_submitted':
                    showToast(`Job submitted: ${data.file_name}`, 'success');
                    refreshJobs();
                    refreshStats();
                    break;
                case 'job_started':
                    showToast(`Processing: ${data.job_id}`, 'success');
                    refreshJobs();
                    refreshStats();
                    break;
                case 'progress':
                    updateJobProgress(data.job_id, data.progress, data.message);
                    // Refresh full job list periodically during progress to catch status changes
                    if (!window._lastJobRefresh || Date.now() - window._lastJobRefresh > 2000) {
                        window._lastJobRefresh = Date.now();
                        refreshJobs();
                    }
                    break;
                case 'job_completed':
                    showToast(`Completed! Confidence: ${(data.confidence*100).toFixed(1)}%`, 'success');
                    refreshJobs();
                    refreshStats();
                    break;
                case 'job_failed':
                    showToast(`Job failed: ${data.error || 'Unknown error'}`, 'error');
                    refreshJobs();
                    refreshStats();
                    break;
                case 'cleanup_complete':
                    showToast(`Cleanup: ${data.jobs_removed} jobs removed`, 'success');
                    refreshJobs();
                    refreshStats();
                    break;
                case 'stats':
                    updateStats(data.data);
                    break;
            }
        }

        // Upload
        async function uploadFile(file) {
            if (!file) return;

            const formData = new FormData();
            formData.append('file', file);

            try {
                const response = await fetch('/api/upload', {
                    method: 'POST',
                    body: formData
                });
                const result = await response.json();
                showToast(`Uploaded: ${result.file_name}`, 'success');
                refreshJobs();
                refreshStats();
            } catch (error) {
                showToast('Upload failed: ' + error.message, 'error');
            }
        }

        // Drag & drop
        const uploadArea = document.getElementById('uploadArea');
        uploadArea.addEventListener('dragover', (e) => {
            e.preventDefault();
            uploadArea.classList.add('dragover');
        });
        uploadArea.addEventListener('dragleave', () => {
            uploadArea.classList.remove('dragover');
        });
        uploadArea.addEventListener('drop', (e) => {
            e.preventDefault();
            uploadArea.classList.remove('dragover');
            if (e.dataTransfer.files.length > 0) {
                uploadFile(e.dataTransfer.files[0]);
            }
        });

        // Jobs list
        async function refreshJobs() {
            try {
                const response = await fetch('/api/jobs?limit=20');
                jobs = await response.json();
                renderJobs();
            } catch (error) {
                console.error('Failed to fetch jobs:', error);
            }
        }

        function renderJobs() {
            const tbody = document.getElementById('jobsTable');
            if (jobs.length === 0) {
                tbody.innerHTML = '<tr><td colspan="7" style="text-align:center;color:#6b7280;">No jobs yet</td></tr>';
                return;
            }

            tbody.innerHTML = jobs.map(job => {
                // Format model tags
                let modelTags = '';
                if (job.ocr_model) {
                    const ocrModelShort = job.ocr_model.split('/').pop().substring(0, 15);
                    modelTags += `<span class="model-tag ocr" title="${job.ocr_model}">OCR: ${ocrModelShort}</span>`;
                }
                if (job.org_model) {
                    const orgModelShort = job.org_model.split('/').pop().substring(0, 15);
                    modelTags += `<br><span class="model-tag org" title="${job.org_model}">ORG: ${orgModelShort}</span>`;
                }
                if (!modelTags) {
                    modelTags = '<span style="color:#6b7280;">-</span>';
                }

                // Calculate progress percentage
                const progressPercent = Math.round(job.progress * 100);

                // Determine actions based on status
                let actions = '-';
                const canViewOcr = ['ocr_complete', 'organizing', 'completed'].includes(job.current_stage) ||
                                   job.status === 'completed' || job.ocr_result_path;

                if (job.status === 'completed') {
                    actions = `
                        <button class="btn btn-sm btn-primary" onclick="viewResult('${job.job_id}')">Result</button>
                        <button class="btn btn-sm btn-outline" onclick="viewOcrText('${job.job_id}')" style="margin-left:4px;">Raw</button>
                    `;
                } else if (canViewOcr) {
                    actions = `<button class="btn btn-sm btn-outline" onclick="viewOcrText('${job.job_id}')">Raw Text</button>`;
                } else if (job.status === 'pending') {
                    actions = `<button class="btn btn-sm btn-outline" onclick="cancelJob('${job.job_id}')">Cancel</button>`;
                }

                // Extract page info from current_stage (e.g., "OCR page 3/10")
                let pagesInfo = '';
                let pageBasedProgress = null;
                const pageMatch = job.current_stage?.match(/page\s+(\d+)\/(\d+)/i);
                if (pageMatch) {
                    const currentPage = parseInt(pageMatch[1]);
                    const totalPages = parseInt(pageMatch[2]);
                    pagesInfo = `${currentPage}/${totalPages} pages`;
                    // Calculate progress based on pages (more accurate during OCR)
                    pageBasedProgress = Math.round((currentPage / totalPages) * 100);
                } else if (job.total_pages > 1) {
                    pagesInfo = `${job.pages_processed || 0}/${job.total_pages} pages`;
                    pageBasedProgress = Math.round(((job.pages_processed || 0) / job.total_pages) * 100);
                }

                // Use page-based progress if available and we're in OCR stage, otherwise use job.progress
                const displayProgress = (pageBasedProgress !== null && job.current_stage?.toLowerCase().includes('ocr'))
                    ? pageBasedProgress
                    : progressPercent;

                // Format stage with page info
                let stageDisplay = job.current_stage || '-';
                if (pagesInfo && !stageDisplay.includes('/')) {
                    stageDisplay = `${pagesInfo} - ${stageDisplay}`;
                }

                // Get original filename from metadata
                const displayName = job.metadata?.original_filename || job.file_name;
                const truncatedName = displayName.length > 40 ? displayName.substring(0, 37) + '...' : displayName;

                return `
                <tr id="job-${job.job_id}">
                    <td><code>${job.job_id.substring(0, 8)}</code></td>
                    <td>
                        <a href="/api/files/${job.job_id}" target="_blank" title="${displayName}" style="color: #7c3aed; text-decoration: none;">
                            ${truncatedName}
                        </a>
                    </td>
                    <td><span class="badge badge-${job.status}">${job.status}</span></td>
                    <td>${modelTags}</td>
                    <td>
                        <div class="progress-container">
                            <div class="progress-bar" style="flex:1;">
                                <div class="progress-fill" style="width: ${displayProgress}%"></div>
                            </div>
                            <span class="progress-percent">${displayProgress}%</span>
                        </div>
                        <small style="color:#6b7280;">${stageDisplay}</small>
                    </td>
                    <td>${job.confidence_score ? (job.confidence_score * 100).toFixed(1) + '%' : '-'}</td>
                    <td>${actions}</td>
                </tr>
            `}).join('');
        }

        function updateJobProgress(jobId, progress, message) {
            const row = document.getElementById(`job-${jobId}`);
            if (row) {
                const progressBar = row.querySelector('.progress-fill');
                const stageText = row.querySelector('small');
                if (progressBar) progressBar.style.width = `${progress * 100}%`;
                if (stageText) stageText.textContent = message;
            }
        }

        // Stats
        async function refreshStats() {
            try {
                const response = await fetch('/api/stats');
                const stats = await response.json();
                updateStats(stats);
            } catch (error) {
                console.error('Failed to fetch stats:', error);
            }
        }

        function updateStats(stats) {
            document.getElementById('statPending').textContent = stats.queue?.pending || 0;
            document.getElementById('statProcessing').textContent = stats.queue?.processing || 0;
            document.getElementById('statCompleted').textContent = stats.queue?.completed || 0;
            document.getElementById('statFailed').textContent = stats.queue?.failed || 0;
        }

        // Actions
        async function startProcessing() {
            try {
                await fetch('/api/process/start', { method: 'POST' });
                showToast('Processing started', 'success');
            } catch (error) {
                showToast('Failed to start: ' + error.message, 'error');
            }
        }

        async function viewResult(jobId) {
            try {
                // First get job info for filename
                const jobResponse = await fetch(`/api/jobs/${jobId}`);
                const job = await jobResponse.json();
                const fileName = job.metadata?.original_filename || job.file_name || jobId;

                const response = await fetch(`/api/jobs/${jobId}/result`);
                if (!response.ok) {
                    const err = await response.json();
                    throw new Error(err.detail || 'Failed to load result');
                }
                const result = await response.json();

                // Open in new window with styled JSON content
                const win = window.open('', '_blank');
                win.document.write(`
                    <html><head><title>Organized JSON: ${fileName}</title>
                    <style>
                        body{background:#0f0f1a;color:#e0e0e0;font-family:-apple-system,BlinkMacSystemFont,sans-serif;padding:20px;margin:0;}
                        .header{background:linear-gradient(135deg,#10b981,#059669);padding:15px 20px;margin:-20px -20px 20px -20px;color:white;}
                        h1{margin:0;font-size:1.3rem;}
                        .meta{font-size:0.85rem;opacity:0.9;margin-top:5px;}
                        .badge{display:inline-block;background:rgba(255,255,255,0.2);padding:3px 8px;border-radius:4px;font-size:0.8rem;margin-left:10px;}
                        pre{white-space:pre-wrap;word-wrap:break-word;background:rgba(255,255,255,0.05);padding:20px;border-radius:8px;line-height:1.6;font-family:'Menlo',monospace;font-size:0.85rem;}
                        .string{color:#10b981;}
                        .number{color:#3b82f6;}
                        .boolean{color:#f59e0b;}
                        .null{color:#6b7280;}
                        .key{color:#7c3aed;}
                    </style></head>
                    <body>
                        <div class="header">
                            <h1>Organized JSON Result<span class="badge">Model: ${job.org_model || 'N/A'}</span></h1>
                            <div class="meta">File: ${fileName} | Job: ${jobId}</div>
                        </div>
                        <pre>${JSON.stringify(result, null, 2)}</pre>
                    </body></html>
                `);
            } catch (error) {
                showToast('Failed to load result: ' + error.message, 'error');
            }
        }

        async function viewOcrText(jobId) {
            try {
                // Get job info for original filename
                const jobResponse = await fetch(`/api/jobs/${jobId}`);
                const job = await jobResponse.json();
                const fileName = job.metadata?.original_filename || job.file_name || jobId;

                const response = await fetch(`/api/jobs/${jobId}/ocr-text`);
                if (!response.ok) {
                    const err = await response.json();
                    throw new Error(err.detail || 'Failed to load OCR text');
                }
                const result = await response.json();

                // Open in new window with styled content
                const win = window.open('', '_blank');
                win.document.write(`
                    <html><head><title>Raw OCR Text: ${fileName}</title>
                    <style>
                        body{background:#0f0f1a;color:#e0e0e0;font-family:-apple-system,BlinkMacSystemFont,sans-serif;padding:20px;margin:0;}
                        .header{background:linear-gradient(135deg,#7c3aed,#2563eb);padding:15px 20px;margin:-20px -20px 20px -20px;color:white;}
                        h1{margin:0;font-size:1.3rem;}
                        .meta{font-size:0.85rem;opacity:0.9;margin-top:5px;}
                        .model-badge{display:inline-block;background:rgba(255,255,255,0.2);padding:3px 8px;border-radius:4px;font-size:0.8rem;margin-left:10px;}
                        pre{white-space:pre-wrap;word-wrap:break-word;background:rgba(255,255,255,0.05);padding:20px;border-radius:8px;line-height:1.6;}
                        .stats{display:flex;gap:20px;margin:15px 0;padding:10px;background:rgba(255,255,255,0.05);border-radius:8px;}
                        .stat{text-align:center;}
                        .stat-value{font-size:1.5rem;font-weight:bold;color:#7c3aed;}
                        .stat-label{font-size:0.75rem;color:#a0a0a0;}
                    </style></head>
                    <body>
                        <div class="header">
                            <h1>Raw OCR/Extracted Text<span class="model-badge">Model: ${result.ocr_model}</span></h1>
                            <div class="meta">File: ${fileName} | Job: ${result.job_id}</div>
                        </div>
                        <div class="stats">
                            <div class="stat">
                                <div class="stat-value">${(result.raw_text || '').length.toLocaleString()}</div>
                                <div class="stat-label">Characters</div>
                            </div>
                            <div class="stat">
                                <div class="stat-value">${(result.raw_text || '').split(/\\s+/).filter(w => w).length.toLocaleString()}</div>
                                <div class="stat-label">Words</div>
                            </div>
                            <div class="stat">
                                <div class="stat-value">${(result.raw_text || '').split(/\\n/).length.toLocaleString()}</div>
                                <div class="stat-label">Lines</div>
                            </div>
                        </div>
                        <pre>${result.raw_text || 'No text extracted'}</pre>
                    </body></html>
                `);
            } catch (error) {
                showToast('Failed to load OCR text: ' + error.message, 'error');
            }
        }

        async function cancelJob(jobId) {
            try {
                await fetch(`/api/jobs/${jobId}`, { method: 'DELETE' });
                showToast('Job cancelled', 'success');
                refreshJobs();
            } catch (error) {
                showToast('Failed to cancel', 'error');
            }
        }

        async function cleanupJobs() {
            if (!confirm('This will permanently delete all failed, cancelled, pending, and processing jobs along with their files. Continue?')) {
                return;
            }

            try {
                const response = await fetch('/api/cleanup', { method: 'DELETE' });
                const result = await response.json();

                if (result.errors && result.errors.length > 0) {
                    console.warn('Cleanup warnings:', result.errors);
                }

                showToast(`Cleaned up: ${result.jobs_removed} jobs, ${result.upload_files_deleted + result.output_files_deleted} files (${result.bytes_freed_mb} MB freed)`, 'success');
                refreshJobs();
                refreshStats();
            } catch (error) {
                showToast('Cleanup failed: ' + error.message, 'error');
            }
        }

        // Toast
        function showToast(message, type = 'success') {
            const toast = document.getElementById('toast');
            toast.textContent = message;
            toast.className = `toast toast-${type} show`;
            setTimeout(() => toast.classList.remove('show'), 3000);
        }

        // Init
        connectWebSocket();
        refreshJobs();
        refreshStats();
        setInterval(refreshJobs, 3000);  // Refresh jobs every 3 seconds for real-time status
        setInterval(refreshStats, 5000); // Refresh stats every 5 seconds
    </script>
</body>
</html>'''


def run_server(host: str = "0.0.0.0", port: int = 8080):
    """Run the server"""
    uvicorn.run(app, host=host, port=port)


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--host", default="0.0.0.0")
    parser.add_argument("--port", type=int, default=8080)
    args = parser.parse_args()

    print(f"Starting IDP Pipeline Server on http://{args.host}:{args.port}")
    run_server(args.host, args.port)
