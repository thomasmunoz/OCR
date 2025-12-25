"""
ORACLE-DESIGNED IDP PIPELINE
============================
Main processing pipeline that orchestrates:
1. File analysis
2. Smart model selection
3. OCR extraction
4. JSON organization
5. Results storage
"""

import os
import sys
import json
import time
from pathlib import Path
from typing import Optional, Dict, Callable
from datetime import datetime
import logging
import threading

sys.path.insert(0, str(Path(__file__).parent.parent))

from config.models_config import OCR_MODELS, ORGANIZATION_MODELS, RECOMMENDED_CONFIG
from config.intermediate_format import OCRIntermediateResult, FinalJSONOutput
from models.smart_router import SmartModelRouter, FileAnalysis, ModelSelection, Priority
from models.ocr_engine import UnifiedOCRProcessor
from models.organizer_engine import UnifiedOrganizer
from job_queue.local_queue import Job, JobStatus, JobPriority
from job_queue.redis_queue import get_queue

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class IDPPipeline:
    """
    Intelligent Document Processing Pipeline.
    100% on-premise, automatic model selection.
    """

    def __init__(self,
                 queue_db: str = "idp_queue.db",
                 output_dir: str = None,
                 available_vram_gb: float = None,
                 priority: str = None,
                 queue_type: str = None):
        """
        Initialize the IDP pipeline.

        Args:
            queue_db: Path to SQLite queue database (if using sqlite)
            output_dir: Directory for output files
            available_vram_gb: Available GPU VRAM (default from IDP_VRAM_GB env, 4.0)
            priority: Processing priority (quality/balanced/speed)
            queue_type: Queue type (redis/sqlite, auto-detected from env)
        """
        # Get VRAM limit from env (default 4GB for Docker compatibility)
        available_vram_gb = available_vram_gb or float(os.getenv("IDP_VRAM_GB", "4.0"))
        priority = priority or os.getenv("IDP_PRIORITY", "balanced")

        # Get queue based on environment or parameter
        queue_type = queue_type or os.getenv("IDP_QUEUE_TYPE", "sqlite")
        if queue_type == "redis":
            from job_queue.redis_queue import RedisQueue
            self.queue = RedisQueue()
        else:
            from job_queue.local_queue import LocalQueue
            self.queue = LocalQueue(queue_db)

        # Output directory from env or parameter
        output_dir = output_dir or os.getenv("IDP_OUTPUT_DIR", "output")
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

        # Smart router for model selection
        priority_map = {"quality": Priority.QUALITY, "balanced": Priority.BALANCED, "speed": Priority.SPEED}
        self.router = SmartModelRouter(available_vram_gb, priority_map.get(priority, Priority.BALANCED))

        # Lazy-loaded engines
        self._ocr_engine = None
        self._org_engine = None
        self._current_ocr_model = None
        self._current_org_model = None

        # Processing state
        self.is_processing = False
        self._stop_requested = False

    def submit(self, file_path: str, priority: str = "normal",
               metadata: Dict = None) -> Job:
        """
        Submit a file for processing.

        Args:
            file_path: Path to document
            priority: Job priority (low/normal/high/urgent)
            metadata: Optional metadata

        Returns:
            Job object
        """
        priority_map = {
            "low": JobPriority.LOW,
            "normal": JobPriority.NORMAL,
            "high": JobPriority.HIGH,
            "urgent": JobPriority.URGENT
        }

        job = self.queue.submit(
            file_path,
            priority_map.get(priority, JobPriority.NORMAL),
            metadata
        )

        logger.info(f"Job submitted: {job.job_id} - {job.file_name}")
        return job

    def process_job(self, job: Job, progress_callback: Callable = None,
                    enhanced_output: bool = True,
                    use_v21: bool = True,
                    use_ai_cascade: bool = True) -> FinalJSONOutput:
        """
        Process a single job through the full pipeline.

        Args:
            job: Job to process
            progress_callback: Optional callback(progress, message)
            enhanced_output: If True, use enhanced JSON format (default)
            use_v21: If True, use v2.1 format with visual content (tables/diagrams/charts)
            use_ai_cascade: If True, escalate to better AI for low-confidence extractions

        Returns:
            FinalJSONOutput (or enhanced dict if enhanced_output=True)
        """
        start_time = time.time()

        try:
            # STAGE 1: Analyze file and select models
            if progress_callback:
                progress_callback(0.05, "Analyzing file...")

            analysis = self.router.analyze_file(job.file_path)
            selection = self.router.select_models(analysis)

            logger.info(f"Selected models: OCR={selection.ocr_model.name}, Org={selection.organization_model.name}")

            # Update job with model info
            self.queue.update_progress(
                job.job_id, 0.08, "models_selected",
                extra={"ocr_model": selection.ocr_model.name, "org_model": selection.organization_model.name}
            )

            # STAGE 1.5: Ensure models are downloaded (with progress bar)
            if progress_callback:
                progress_callback(0.08, "Checking model availability...")

            def download_progress(current, total, model_name, status):
                if status == "downloading":
                    if total > 0:
                        pct = current / total * 100
                        if progress_callback:
                            progress_callback(0.08, f"Downloading {model_name}: {pct:.1f}%")
                        self.queue.update_progress(
                            job.job_id, 0.08,
                            f"Downloading {model_name}: {pct:.1f}%"
                        )
                elif status == "ready":
                    if progress_callback:
                        progress_callback(0.09, f"{model_name} ready")

            models_ready = self.router.ensure_models_ready(selection, download_progress)
            if not models_ready:
                raise RuntimeError("Failed to download required models")

            # STAGE 2: Load/switch models if needed
            if progress_callback:
                progress_callback(0.1, f"Loading {selection.ocr_model.name}...")

            ocr_model_key = self._find_model_key(selection.ocr_model.hf_repo, OCR_MODELS)
            org_model_key = self._find_model_key(selection.organization_model.hf_repo, ORGANIZATION_MODELS)

            if self._current_ocr_model != ocr_model_key:
                if self._ocr_engine:
                    self._ocr_engine.engine.unload_model()
                self._ocr_engine = UnifiedOCRProcessor(ocr_model_key)
                self._current_ocr_model = ocr_model_key

            # STAGE 3: OCR
            if progress_callback:
                progress_callback(0.15, "Starting OCR...")

            def ocr_progress(p, msg):
                actual_progress = 0.15 + p * 0.4  # OCR is 15-55%
                if progress_callback:
                    progress_callback(actual_progress, msg)
                self.queue.update_progress(job.job_id, actual_progress, msg)

            ocr_result = self._ocr_engine.process_document(
                job.file_path,
                job.job_id,
                ocr_progress
            )

            # Build enhanced OCR data if requested
            enhanced_ocr_data = None
            if enhanced_output:
                if progress_callback:
                    progress_callback(0.55, "Building enhanced metadata...")

                # Use v2.1 with visual content extraction if requested
                if use_v21:
                    if progress_callback:
                        progress_callback(0.56, "Building v2.1 enhanced output with visual content...")
                    enhanced_ocr_data = self._ocr_engine.build_enhanced_output_v21(
                        ocr_result,
                        job.file_path,
                        use_ai_cascade=use_ai_cascade
                    )
                    logger.info(f"V2.1 output: schema={enhanced_ocr_data.get('schema_version')}, "
                               f"tables={enhanced_ocr_data.get('visual_content', {}).get('total_tables', 0)}")
                else:
                    # Fall back to v2.0
                    enhanced_ocr_data = self._ocr_engine.build_enhanced_output(
                        ocr_result,
                        job.file_path
                    )

            # Save intermediate result
            ocr_output_path = self.output_dir / f"{job.job_id}_ocr.json"
            with open(ocr_output_path, 'w', encoding='utf-8') as f:
                f.write(ocr_result.to_json())

            self.queue.complete_ocr(job.job_id, str(ocr_output_path), selection.ocr_model.name)

            # STAGE 4: Organization
            if progress_callback:
                progress_callback(0.6, f"Loading {selection.organization_model.name}...")

            # CRITICAL: Unload OCR model BEFORE loading organizer to free memory
            # This is essential for CPU/low-VRAM environments
            if self._ocr_engine and self._ocr_engine.engine:
                logger.info("Unloading OCR model to free memory for organizer")
                self._ocr_engine.engine.unload_model()
                self._ocr_engine = None
                self._current_ocr_model = None

                # Aggressive memory cleanup
                import gc
                import torch
                gc.collect()
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
                    torch.cuda.synchronize()

                # Optional brief pause for memory reclaim (configurable, default 0)
                cleanup_delay = float(os.getenv("IDP_MEMORY_CLEANUP_DELAY", "0"))
                if cleanup_delay > 0:
                    time.sleep(cleanup_delay)
                gc.collect()
                logger.info("Memory cleanup completed")

            if self._current_org_model != org_model_key:
                if self._org_engine:
                    self._org_engine.engine.unload_model()
                self._org_engine = UnifiedOrganizer(org_model_key)
                self._current_org_model = org_model_key

            def org_progress(p, msg):
                actual_progress = 0.6 + p * 0.35  # Org is 60-95%
                if progress_callback:
                    progress_callback(actual_progress, msg)
                self.queue.update_progress(job.job_id, actual_progress, msg)

            # Use enhanced processing if available
            if enhanced_output and enhanced_ocr_data:
                final_result = self._org_engine.process_enhanced(
                    ocr_result,
                    enhanced_ocr_data,
                    org_progress
                )
                overall_confidence = final_result.get("overall_confidence", 0.8)
            else:
                final_result = self._org_engine.process(ocr_result, org_progress)
                overall_confidence = final_result.overall_confidence

            # STAGE 5: Save final result
            if progress_callback:
                progress_callback(0.95, "Saving results...")

            final_output_path = self.output_dir / f"{job.job_id}_final.json"
            with open(final_output_path, 'w', encoding='utf-8') as f:
                if enhanced_output and isinstance(final_result, dict):
                    json.dump(final_result, f, indent=2, default=str)
                else:
                    f.write(final_result.to_json())

            # Also save compact version
            compact_output_path = self.output_dir / f"{job.job_id}_compact.json"
            with open(compact_output_path, 'w', encoding='utf-8') as f:
                if enhanced_output and isinstance(final_result, dict):
                    # Create compact version from enhanced output
                    compact = {
                        "schema_version": final_result.get("schema_version", "2.0"),
                        "job_id": final_result.get("job_id"),
                        "document_type": final_result.get("document_type"),
                        "total_pages": final_result.get("total_pages"),
                        "primary_language": final_result.get("primary_language"),
                        "signature_status": final_result.get("signatures", {}).get("status"),
                        "overall_confidence": final_result.get("overall_confidence"),
                        "total_words": final_result.get("total_words"),
                        "total_characters": final_result.get("total_characters"),
                        "pages_summary": [
                            {
                                "page": p.get("page_number"),
                                "language": p.get("language", {}).get("detected"),
                                "confidence": p.get("quality", {}).get("extraction_confidence"),
                                "words": p.get("statistics", {}).get("word_count"),
                                "chars": p.get("statistics", {}).get("char_count"),
                                "tables": p.get("statistics", {}).get("table_count", 0)
                            }
                            for p in final_result.get("pages", [])
                        ]
                    }

                    # Add v2.1 visual content summary if available
                    visual_content = final_result.get("visual_content")
                    if visual_content:
                        compact["visual_content_summary"] = {
                            "total_tables": visual_content.get("total_tables", 0),
                            "total_diagrams": visual_content.get("total_diagrams", 0),
                            "total_charts": visual_content.get("total_charts", 0)
                        }

                        # Include table reconstruction formats in compact output
                        reconstruction = final_result.get("reconstruction", {})
                        if reconstruction.get("tables_markdown"):
                            compact["tables_markdown"] = reconstruction["tables_markdown"]

                    json.dump(compact, f, indent=2)
                else:
                    f.write(final_result.to_compact_json())

            # Mark complete
            self.queue.complete(
                job.job_id,
                str(final_output_path),
                overall_confidence,
                selection.organization_model.name
            )

            if progress_callback:
                progress_callback(1.0, "Complete!")

            logger.info(f"Job completed: {job.job_id} in {time.time() - start_time:.1f}s")
            return final_result

        except Exception as e:
            logger.error(f"Job failed: {job.job_id} - {e}")
            self.queue.fail(job.job_id, str(e))
            raise

    def process_next(self, progress_callback: Callable = None) -> Optional[FinalJSONOutput]:
        """
        Get and process the next job in queue.

        Returns:
            FinalJSONOutput if job processed, None if queue empty
        """
        job = self.queue.get_next()
        if job:
            return self.process_job(job, progress_callback)
        return None

    def start_worker(self, poll_interval: float = 2.0,
                     progress_callback: Callable = None):
        """
        Start continuous worker loop.

        Args:
            poll_interval: Seconds between queue checks
            progress_callback: Optional progress callback
        """
        self.is_processing = True
        self._stop_requested = False

        logger.info("Worker started")

        while not self._stop_requested:
            try:
                result = self.process_next(progress_callback)
                if result is None:
                    # No jobs, wait
                    time.sleep(poll_interval)
            except Exception as e:
                logger.error(f"Worker error: {e}")
                time.sleep(poll_interval)

        self.is_processing = False
        logger.info("Worker stopped")

    def stop_worker(self):
        """Stop the worker loop"""
        self._stop_requested = True

    def get_status(self) -> Dict:
        """Get pipeline status"""
        stats = self.queue.get_stats()
        return {
            "is_processing": self.is_processing,
            "current_ocr_model": self._current_ocr_model,
            "current_org_model": self._current_org_model,
            "queue_stats": stats,
            "output_dir": str(self.output_dir)
        }

    def _find_model_key(self, hf_repo: str, models_dict: Dict) -> str:
        """Find model key by HuggingFace repo"""
        for key, model in models_dict.items():
            if model.hf_repo == hf_repo:
                return key
        # Return default
        return list(models_dict.keys())[0]


class PipelineWorkerThread(threading.Thread):
    """Background worker thread for the pipeline"""

    def __init__(self, pipeline: IDPPipeline, progress_callback: Callable = None):
        super().__init__(daemon=True)
        self.pipeline = pipeline
        self.progress_callback = progress_callback

    def run(self):
        self.pipeline.start_worker(progress_callback=self.progress_callback)

    def stop(self):
        self.pipeline.stop_worker()


# Simple CLI interface
if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="IDP Pipeline")
    subparsers = parser.add_subparsers(dest="command")

    # Submit command
    submit = subparsers.add_parser("submit", help="Submit file")
    submit.add_argument("file", help="File to process")
    submit.add_argument("--priority", default="normal")

    # Process command
    process = subparsers.add_parser("process", help="Process next job")

    # Worker command
    worker = subparsers.add_parser("worker", help="Start worker")

    # Status command
    status = subparsers.add_parser("status", help="Show status")

    args = parser.parse_args()

    pipeline = IDPPipeline()

    if args.command == "submit":
        job = pipeline.submit(args.file, args.priority)
        print(f"Submitted: {job.job_id}")

    elif args.command == "process":
        def show_progress(p, msg):
            print(f"[{p*100:5.1f}%] {msg}")

        result = pipeline.process_next(show_progress)
        if result:
            print(f"\nCompleted! Output: {pipeline.output_dir}")
        else:
            print("No jobs in queue")

    elif args.command == "worker":
        print("Starting worker... (Ctrl+C to stop)")
        try:
            pipeline.start_worker()
        except KeyboardInterrupt:
            pipeline.stop_worker()

    elif args.command == "status":
        status = pipeline.get_status()
        print(json.dumps(status, indent=2))
