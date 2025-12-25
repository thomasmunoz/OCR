#!/usr/bin/env python3
"""
IDP PIPELINE - Main Entry Point
===============================
100% On-Premise Intelligent Document Processing
"""

import argparse
import sys
from pathlib import Path

def main():
    parser = argparse.ArgumentParser(
        description="IDP Pipeline - Intelligent Document Processing",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Start web UI
  python run.py serve

  # Submit a file
  python run.py submit invoice.pdf

  # Process one job
  python run.py process

  # Start background worker
  python run.py worker

  # Check status
  python run.py status

  # Download recommended models
  python run.py download
"""
    )

    subparsers = parser.add_subparsers(dest="command", help="Commands")

    # Serve command
    serve_parser = subparsers.add_parser("serve", help="Start web UI server")
    serve_parser.add_argument("--host", default="127.0.0.1", help="Host to bind")
    serve_parser.add_argument("--port", type=int, default=8080, help="Port to bind")

    # Submit command
    submit_parser = subparsers.add_parser("submit", help="Submit file for processing")
    submit_parser.add_argument("file", help="File to process")
    submit_parser.add_argument("--priority", choices=["low", "normal", "high", "urgent"], default="normal")

    # Process command
    process_parser = subparsers.add_parser("process", help="Process next job")

    # Worker command
    worker_parser = subparsers.add_parser("worker", help="Start background worker")

    # Status command
    status_parser = subparsers.add_parser("status", help="Show queue status")

    # Download command
    download_parser = subparsers.add_parser("download", help="Download recommended models")
    download_parser.add_argument("--all", action="store_true", help="Download all models")

    # Analyze command
    analyze_parser = subparsers.add_parser("analyze", help="Analyze file and suggest models")
    analyze_parser.add_argument("file", help="File to analyze")

    args = parser.parse_args()

    if args.command == "serve":
        from api.server import run_server
        print(f"Starting IDP Pipeline UI at http://{args.host}:{args.port}")
        run_server(args.host, args.port)

    elif args.command == "submit":
        from api.pipeline import IDPPipeline
        pipeline = IDPPipeline()
        job = pipeline.submit(args.file, args.priority)
        print(f"Job submitted: {job.job_id}")
        print(f"File: {job.file_name}")
        print(f"Priority: {args.priority}")

    elif args.command == "process":
        from api.pipeline import IDPPipeline
        pipeline = IDPPipeline()

        def progress(p, msg):
            bar = "█" * int(p * 30) + "░" * (30 - int(p * 30))
            print(f"\r[{bar}] {p*100:5.1f}% {msg}", end="", flush=True)

        result = pipeline.process_next(progress)
        print()
        if result:
            print(f"\nCompleted!")
            print(f"Document type: {result.document_type}")
            print(f"Confidence: {result.overall_confidence*100:.1f}%")
        else:
            print("No jobs in queue")

    elif args.command == "worker":
        from api.pipeline import IDPPipeline
        pipeline = IDPPipeline()
        print("Starting worker... (Ctrl+C to stop)")
        try:
            pipeline.start_worker()
        except KeyboardInterrupt:
            pipeline.stop_worker()
            print("\nWorker stopped")

    elif args.command == "status":
        from api.pipeline import IDPPipeline
        import json
        pipeline = IDPPipeline()
        status = pipeline.get_status()
        print(json.dumps(status, indent=2))

    elif args.command == "download":
        from config.models_config import OCR_MODELS, ORGANIZATION_MODELS, RECOMMENDED_CONFIG
        import subprocess

        models_to_download = []

        if args.all:
            for m in OCR_MODELS.values():
                models_to_download.append(m.hf_repo)
            for m in ORGANIZATION_MODELS.values():
                models_to_download.append(m.hf_repo)
        else:
            # Just recommended
            ocr_key = RECOMMENDED_CONFIG["ocr_default"]
            org_key = RECOMMENDED_CONFIG["org_default"]
            models_to_download.append(OCR_MODELS[ocr_key].hf_repo)
            models_to_download.append(ORGANIZATION_MODELS[org_key].hf_repo)

        print(f"Downloading {len(models_to_download)} models...")
        for repo in models_to_download:
            print(f"\n>>> {repo}")
            subprocess.run(["huggingface-cli", "download", repo], check=False)

    elif args.command == "analyze":
        from models.smart_router import SmartModelRouter
        router = SmartModelRouter()
        analysis = router.analyze_file(args.file)
        selection = router.select_models(analysis)

        print(f"\n{'='*50}")
        print("FILE ANALYSIS")
        print(f"{'='*50}")
        print(f"Format: {analysis.file_format}")
        print(f"Size: {analysis.file_size_bytes / 1024:.1f} KB")
        print(f"Language: {analysis.detected_language.value}")
        print(f"Pages: {analysis.page_count}")
        print(f"Complexity: {analysis.complexity_score}/100")

        print(f"\n{'='*50}")
        print("RECOMMENDED MODELS")
        print(f"{'='*50}")
        print(f"OCR: {selection.ocr_model.name}")
        print(f"      {selection.ocr_model.hf_repo}")
        print(f"Organization: {selection.organization_model.name}")
        print(f"      {selection.organization_model.hf_repo}")
        print(f"\nSelection confidence: {selection.confidence*100:.0f}%")

        print(f"\n{'='*50}")
        print("DOWNLOAD COMMANDS")
        print(f"{'='*50}")
        for cmd in router.get_download_commands(selection):
            print(cmd)

    else:
        parser.print_help()

if __name__ == "__main__":
    main()
