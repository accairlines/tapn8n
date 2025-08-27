# PPT Generation Service

A Django-based service for processing Outlook emails and generating PowerPoint presentations using RAG (Retrieval-Augmented Generation).

## Overview

This service provides:
- Email parsing and indexing from Outlook .msg files
- Vector storage using Qdrant
- RAG-based question answering using LLaMA
- Automatic PowerPoint generation from email content

## Architecture

- **Django REST API**: Main service interface
- **Email Parser**: Processes .msg files using extract-msg
- **Text Chunker**: Splits content into manageable chunks
- **Vector Database**: Qdrant for similarity search
- **LLaMA Integration**: Uses existing llama container for embeddings and generation
- **PPT Builder**: Creates PowerPoint presentations from RAG results

## API Endpoints

### Health Check
- `GET /healthz/` - Service health status

### Email Processing
- `POST /reindex/` - Trigger email reindexing
- `POST /ask/` - Ask questions about emails
- `POST /generate-ppt/` - Generate PowerPoint presentation

## Configuration

Environment variables:
- `OLLAMA_BASE_URL`: LLaMA service URL (default: http://llama:11434)
- `QDRANT_URL`: Qdrant service URL (default: http://qdrant:6333)
- `DATA_DIR`: Email data directory (default: /data/outlook)
- `LLAMA_MODEL`: LLaMA model name (default: llama3.1:8b)
- `EMBED_MODEL`: Embedding model (default: nomic-embed-text)

## Usage

### 1. Place Outlook .msg files in the data directory
```
data/emails/
├── email1.msg
├── email2.msg
└── ...
```

### 2. Trigger reindexing
```bash
curl -X POST http://localhost:8001/reindex/
```

### 3. Ask questions
```bash
curl -X POST http://localhost:8001/ask/ \
  -H "Content-Type: application/json" \
  -d '{"question": "What are the main topics discussed in the emails?"}'
```

### 4. Generate PowerPoint
```bash
curl -X POST http://localhost:8001/generate-ppt/ \
  -H "Content-Type: application/json" \
  -d '{"question": "Summarize the key findings", "slide_count": 5}'
```

## Development

### Running locally
```bash
cd services/pptgeneration
pip install -r requirements.txt
python manage.py runserver 8001
```

### Running with Docker
```bash
docker build -f Dockerfile.django -t pptgeneration .
docker run -p 8001:8001 pptgeneration
```

## Dependencies

- Python 3.11+
- Django 4.2+
- Django REST Framework
- Qdrant Client
- extract-msg
- BeautifulSoup4
- python-pptx
- httpx

## Notes

- The service integrates with existing llama and qdrant containers
- Email processing is optimized for Portuguese language content
- PowerPoint generation includes source citations
- All operations are logged for monitoring and debugging
