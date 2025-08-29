#!/usr/bin/env python
"""
Email ingestion script for processing Outlook .msg files.
This script can be run independently or called from the Django service.
"""

import os
import sys
import logging
import argparse
from pathlib import Path
from datetime import datetime

# Add the parent directory to Python path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Django setup
os.environ.setdefault('DJANGO_SETTINGS_MODULE', 'pptgeneration.settings')

import django
django.setup()

from rag.tasks import reindex_emails
from rag.utils import get_system_info


def main():
    """Main ingestion function."""
    parser = argparse.ArgumentParser(description='Process Outlook .msg files for RAG indexing')
    parser.add_argument('--force', action='store_true', help='Force reindexing of all files')
    parser.add_argument('--data-dir', type=str, help='Override data directory path')
    parser.add_argument('--verbose', '-v', action='store_true', help='Enable verbose logging')
    
    args = parser.parse_args()
    
    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)
    
    try:
        logger.info("Starting email ingestion process")
        
        # Log system information
        system_info = get_system_info()
        logger.info(f"System info: {system_info}")
        
        # Start reindexing
        start_time = datetime.now()
        
        result = reindex_emails(force=args.force)
        
        end_time = datetime.now()
        
        # Log results
        processing_time = (end_time - start_time).total_seconds()
        # Log to database
        logger.info(f"Email ingestion completed, {str({
            'force': args.force,
            'processing_time': processing_time,
            'result': result,
            'system_info': system_info
        })}")
        
        return 0
        
    except Exception as e:
        logger.error(f"Ingestion failed: {e}")
                
        return 1


if __name__ == '__main__':
    sys.exit(main())
