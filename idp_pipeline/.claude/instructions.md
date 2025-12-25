# IDP Pipeline - Project Instructions

## Reports Location

**All HTML reports must be saved to the parent OCR reports folder:**

```
/Users/tomahawk/DEV/DEVX/OCR/reports/
```

**NOT in the project subfolder:**
```
/Users/tomahawk/DEV/DEVX/OCR/idp_pipeline/reports/  (INCORRECT)
```

This centralizes all OCR project reports in one location for easy access.

## Project Structure

- **Working directory:** `/Users/tomahawk/DEV/DEVX/OCR/idp_pipeline/`
- **Reports directory:** `/Users/tomahawk/DEV/DEVX/OCR/reports/`
- **Docker:** Uses docker-compose for multi-container deployment
- **Queue:** SQLite (local) or Redis (distributed)

## Key Files

- `api/server.py` - FastAPI server with dashboard
- `api/pipeline.py` - Main processing pipeline
- `job_queue/local_queue.py` - SQLite queue implementation
- `job_queue/redis_queue.py` - Redis queue implementation
- `config/models_config.py` - AI model configurations
