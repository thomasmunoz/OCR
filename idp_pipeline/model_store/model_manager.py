"""
Model Manager - Download and Cache HuggingFace Models
======================================================
Provides:
- Local model storage in model_store/
- Progress bar during downloads
- Model availability checking
- Automatic download on first use
"""

import os
import sys
import json
import shutil
import logging
from pathlib import Path
from typing import Optional, Dict, List, Callable
from dataclasses import dataclass, asdict
from datetime import datetime
from enum import Enum
import threading

# Initialize logger
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Add parent path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from config.models_config import OCR_MODELS, ORGANIZATION_MODELS, OCRModel, OrganizationModel


class ModelStatus(Enum):
    """Status of a model in the local store"""
    NOT_DOWNLOADED = "not_downloaded"
    DOWNLOADING = "downloading"
    READY = "ready"
    ERROR = "error"


@dataclass
class ModelInfo:
    """Information about a locally stored model"""
    model_key: str
    model_name: str
    hf_repo: str
    model_type: str  # "ocr" or "organization"
    status: ModelStatus
    local_path: Optional[str] = None
    size_bytes: Optional[int] = None
    downloaded_at: Optional[str] = None
    error_message: Optional[str] = None


class ProgressCallback:
    """Progress callback wrapper for download progress"""

    def __init__(self, model_name: str, callback: Optional[Callable] = None):
        self.model_name = model_name
        self.callback = callback
        self.current = 0
        self.total = 0

    def __call__(self, current: int, total: int):
        self.current = current
        self.total = total
        if self.callback:
            self.callback(current, total, self.model_name)


class ModelManager:
    """
    Manages local model storage and downloads.

    Usage:
        manager = ModelManager()

        # Check if model is available
        if not manager.is_model_ready("qwen2_vl_2b"):
            # Download with progress
            manager.download_model("qwen2_vl_2b", progress_callback=my_callback)

        # Get local path
        path = manager.get_model_path("qwen2_vl_2b")
    """

    def __init__(self, store_path: Optional[str] = None):
        """
        Initialize ModelManager.

        Args:
            store_path: Path to store models. Defaults to ./model_store or IDP_MODELS_DIR env
        """
        if store_path:
            self.store_path = Path(store_path)
        else:
            # Check env, then default to model_store in project
            env_path = os.getenv("IDP_MODELS_DIR")
            if env_path:
                self.store_path = Path(env_path)
            else:
                self.store_path = Path(__file__).parent

        self.store_path.mkdir(parents=True, exist_ok=True)
        self.registry_path = self.store_path / "registry.json"
        self._registry: Dict[str, ModelInfo] = {}
        self._download_locks: Dict[str, threading.Lock] = {}
        self._global_lock = threading.Lock()

        # Load existing registry
        self._load_registry()

        logger.info(f"ModelManager initialized. Store path: {self.store_path}")

    def _load_registry(self):
        """Load model registry from disk"""
        if self.registry_path.exists():
            try:
                with open(self.registry_path, 'r') as f:
                    data = json.load(f)
                    for key, info in data.items():
                        info['status'] = ModelStatus(info['status'])
                        self._registry[key] = ModelInfo(**info)
            except Exception as e:
                logger.warning(f"Could not load registry: {e}")
                self._registry = {}

    def _save_registry(self):
        """Save model registry to disk"""
        try:
            data = {}
            for key, info in self._registry.items():
                info_dict = asdict(info)
                info_dict['status'] = info.status.value
                data[key] = info_dict
            with open(self.registry_path, 'w') as f:
                json.dump(data, f, indent=2)
        except Exception as e:
            logger.error(f"Could not save registry: {e}")

    def _get_lock(self, model_key: str) -> threading.Lock:
        """Get or create lock for a model"""
        with self._global_lock:
            if model_key not in self._download_locks:
                self._download_locks[model_key] = threading.Lock()
            return self._download_locks[model_key]

    def get_model_config(self, model_key: str) -> Optional[OCRModel | OrganizationModel]:
        """Get model configuration by key"""
        if model_key in OCR_MODELS:
            return OCR_MODELS[model_key]
        elif model_key in ORGANIZATION_MODELS:
            return ORGANIZATION_MODELS[model_key]
        return None

    def get_model_type(self, model_key: str) -> Optional[str]:
        """Get model type (ocr or organization)"""
        if model_key in OCR_MODELS:
            return "ocr"
        elif model_key in ORGANIZATION_MODELS:
            return "organization"
        return None

    def list_available_models(self) -> Dict[str, List[str]]:
        """List all available models by type"""
        return {
            "ocr": list(OCR_MODELS.keys()),
            "organization": list(ORGANIZATION_MODELS.keys())
        }

    def get_model_status(self, model_key: str) -> ModelStatus:
        """Get status of a model"""
        if model_key in self._registry:
            info = self._registry[model_key]
            # Verify the path still exists
            if info.status == ModelStatus.READY and info.local_path:
                if not Path(info.local_path).exists():
                    info.status = ModelStatus.NOT_DOWNLOADED
                    info.local_path = None
                    self._save_registry()
            return info.status
        return ModelStatus.NOT_DOWNLOADED

    def is_model_ready(self, model_key: str) -> bool:
        """Check if a model is downloaded and ready"""
        return self.get_model_status(model_key) == ModelStatus.READY

    def get_model_path(self, model_key: str) -> Optional[str]:
        """Get local path for a downloaded model"""
        if model_key in self._registry:
            info = self._registry[model_key]
            if info.status == ModelStatus.READY and info.local_path:
                return info.local_path
        return None

    def get_model_info(self, model_key: str) -> Optional[ModelInfo]:
        """Get full info for a model"""
        return self._registry.get(model_key)

    def download_model(
        self,
        model_key: str,
        progress_callback: Optional[Callable[[int, int, str], None]] = None,
        force: bool = False
    ) -> bool:
        """
        Download a model from HuggingFace.

        Args:
            model_key: Key of the model (e.g., "qwen2_vl_2b")
            progress_callback: Optional callback(current_bytes, total_bytes, model_name)
            force: Force re-download even if exists

        Returns:
            True if download successful, False otherwise
        """
        config = self.get_model_config(model_key)
        if not config:
            logger.error(f"Unknown model: {model_key}")
            return False

        model_type = self.get_model_type(model_key)

        # Check if already downloading or ready
        lock = self._get_lock(model_key)
        with lock:
            status = self.get_model_status(model_key)

            if status == ModelStatus.READY and not force:
                logger.info(f"Model {model_key} already downloaded")
                return True

            if status == ModelStatus.DOWNLOADING:
                logger.info(f"Model {model_key} is already being downloaded")
                return False

            # Update status
            self._registry[model_key] = ModelInfo(
                model_key=model_key,
                model_name=config.name,
                hf_repo=config.hf_repo,
                model_type=model_type,
                status=ModelStatus.DOWNLOADING
            )
            self._save_registry()

        try:
            # Import huggingface_hub
            from huggingface_hub import snapshot_download, HfApi
            from tqdm import tqdm

            logger.info(f"Downloading {config.name} from {config.hf_repo}...")

            # Create model directory
            model_dir = self.store_path / model_key
            model_dir.mkdir(parents=True, exist_ok=True)

            # Create progress bar
            pbar = None
            last_progress = [0]

            def tqdm_progress(current, total):
                nonlocal pbar
                if pbar is None and total > 0:
                    pbar = tqdm(
                        total=total,
                        unit='B',
                        unit_scale=True,
                        unit_divisor=1024,
                        desc=f"Downloading {config.name}",
                        ncols=100
                    )
                if pbar:
                    pbar.update(current - last_progress[0])
                    last_progress[0] = current

                if progress_callback:
                    progress_callback(current, total, config.name)

            # Download using huggingface_hub
            local_path = snapshot_download(
                repo_id=config.hf_repo,
                local_dir=str(model_dir),
                local_dir_use_symlinks=False,
                resume_download=True
            )

            if pbar:
                pbar.close()

            # Calculate size
            total_size = sum(
                f.stat().st_size for f in model_dir.rglob('*') if f.is_file()
            )

            # Update registry
            with lock:
                self._registry[model_key] = ModelInfo(
                    model_key=model_key,
                    model_name=config.name,
                    hf_repo=config.hf_repo,
                    model_type=model_type,
                    status=ModelStatus.READY,
                    local_path=str(model_dir),
                    size_bytes=total_size,
                    downloaded_at=datetime.now().isoformat()
                )
                self._save_registry()

            logger.info(f"Successfully downloaded {config.name} to {model_dir}")
            logger.info(f"Size: {total_size / (1024**3):.2f} GB")

            return True

        except Exception as e:
            logger.error(f"Failed to download {model_key}: {e}")

            with lock:
                self._registry[model_key] = ModelInfo(
                    model_key=model_key,
                    model_name=config.name,
                    hf_repo=config.hf_repo,
                    model_type=model_type,
                    status=ModelStatus.ERROR,
                    error_message=str(e)
                )
                self._save_registry()

            return False

    def ensure_model_ready(
        self,
        model_key: str,
        progress_callback: Optional[Callable[[int, int, str], None]] = None
    ) -> str:
        """
        Ensure a model is downloaded and return its path.
        Downloads if not present.

        Args:
            model_key: Key of the model
            progress_callback: Optional progress callback

        Returns:
            Local path to the model

        Raises:
            RuntimeError: If model cannot be downloaded
        """
        if not self.is_model_ready(model_key):
            logger.info(f"Model {model_key} not found locally. Downloading...")
            success = self.download_model(model_key, progress_callback)
            if not success:
                raise RuntimeError(f"Failed to download model: {model_key}")

        path = self.get_model_path(model_key)
        if not path:
            raise RuntimeError(f"Model path not found: {model_key}")

        return path

    def delete_model(self, model_key: str) -> bool:
        """Delete a downloaded model"""
        lock = self._get_lock(model_key)
        with lock:
            if model_key in self._registry:
                info = self._registry[model_key]
                if info.local_path and Path(info.local_path).exists():
                    try:
                        shutil.rmtree(info.local_path)
                        logger.info(f"Deleted model: {model_key}")
                    except Exception as e:
                        logger.error(f"Failed to delete model files: {e}")
                        return False

                del self._registry[model_key]
                self._save_registry()
                return True
        return False

    def get_store_stats(self) -> Dict:
        """Get statistics about the model store"""
        total_size = 0
        model_count = 0

        for key, info in self._registry.items():
            if info.status == ModelStatus.READY and info.size_bytes:
                total_size += info.size_bytes
                model_count += 1

        return {
            "store_path": str(self.store_path),
            "total_models": model_count,
            "total_size_gb": round(total_size / (1024**3), 2),
            "models": {
                k: {
                    "name": v.model_name,
                    "status": v.status.value,
                    "size_gb": round(v.size_bytes / (1024**3), 2) if v.size_bytes else None
                }
                for k, v in self._registry.items()
            }
        }

    def print_status(self):
        """Print formatted status of all models"""
        print("\n" + "=" * 70)
        print("MODEL STORE STATUS")
        print("=" * 70)
        print(f"Store Path: {self.store_path}")
        print()

        stats = self.get_store_stats()
        print(f"Total Models: {stats['total_models']}")
        print(f"Total Size: {stats['total_size_gb']} GB")
        print()

        print("OCR Models:")
        print("-" * 50)
        for key in OCR_MODELS.keys():
            status = self.get_model_status(key)
            config = OCR_MODELS[key]
            status_icon = "✓" if status == ModelStatus.READY else "✗"
            print(f"  [{status_icon}] {key}: {config.name}")
            if status == ModelStatus.READY:
                info = self._registry.get(key)
                if info and info.size_bytes:
                    print(f"      Size: {info.size_bytes / (1024**3):.2f} GB")

        print("\nOrganization Models:")
        print("-" * 50)
        for key in ORGANIZATION_MODELS.keys():
            status = self.get_model_status(key)
            config = ORGANIZATION_MODELS[key]
            status_icon = "✓" if status == ModelStatus.READY else "✗"
            print(f"  [{status_icon}] {key}: {config.name}")
            if status == ModelStatus.READY:
                info = self._registry.get(key)
                if info and info.size_bytes:
                    print(f"      Size: {info.size_bytes / (1024**3):.2f} GB")

        print("=" * 70)


# Singleton instance
_manager_instance: Optional[ModelManager] = None
_manager_lock = threading.Lock()


def get_model_manager(store_path: Optional[str] = None) -> ModelManager:
    """Get or create the singleton ModelManager instance"""
    global _manager_instance
    with _manager_lock:
        if _manager_instance is None:
            _manager_instance = ModelManager(store_path)
        return _manager_instance


# CLI Interface
if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Model Manager CLI")
    subparsers = parser.add_subparsers(dest="command")

    # Status command
    status_parser = subparsers.add_parser("status", help="Show model status")

    # List command
    list_parser = subparsers.add_parser("list", help="List available models")

    # Download command
    download_parser = subparsers.add_parser("download", help="Download a model")
    download_parser.add_argument("model", help="Model key to download")
    download_parser.add_argument("--force", action="store_true", help="Force re-download")

    # Delete command
    delete_parser = subparsers.add_parser("delete", help="Delete a downloaded model")
    delete_parser.add_argument("model", help="Model key to delete")

    args = parser.parse_args()

    manager = get_model_manager()

    if args.command == "status":
        manager.print_status()

    elif args.command == "list":
        models = manager.list_available_models()
        print("\nAvailable OCR Models:")
        for key in models["ocr"]:
            config = OCR_MODELS[key]
            print(f"  - {key}: {config.name} ({config.vram_required_gb}GB VRAM)")
        print("\nAvailable Organization Models:")
        for key in models["organization"]:
            config = ORGANIZATION_MODELS[key]
            print(f"  - {key}: {config.name} ({config.vram_required_gb}GB VRAM)")

    elif args.command == "download":
        print(f"\nDownloading {args.model}...")
        success = manager.download_model(args.model, force=args.force)
        if success:
            print(f"\n✓ Successfully downloaded {args.model}")
        else:
            print(f"\n✗ Failed to download {args.model}")
            sys.exit(1)

    elif args.command == "delete":
        confirm = input(f"Delete model {args.model}? [y/N]: ")
        if confirm.lower() == 'y':
            if manager.delete_model(args.model):
                print(f"✓ Deleted {args.model}")
            else:
                print(f"✗ Failed to delete {args.model}")
        else:
            print("Cancelled")

    else:
        parser.print_help()
