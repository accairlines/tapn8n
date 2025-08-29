import os
import logging
import hashlib
from pathlib import Path
from datetime import datetime
from typing import List, Dict, Any
import re
import base64
import extract_msg
from bs4 import BeautifulSoup
from django.conf import settings

logger = logging.getLogger(__name__)


class EmailParser:
    """Parser for Outlook .msg files."""
    
    def __init__(self):
        self.data_dir = Path(os.environ.get('DATA_DIR', ''))
        self.supported_extensions = {'.msg'}
    
    def parse_all_emails(self, force=False) -> List[Dict[str, Any]]:
        """Parse all email files in the data directory."""
        emails = []
        
        logger.info(f"Data directory: {self.data_dir}")
        if not self.data_dir.exists():
            logger.warning(f"Data directory does not exist: {self.data_dir}")
            return emails
        
        for file_path in self.data_dir.rglob('*.msg'):
            try:
                email_data = self.parse_html_file(file_path, force=force)
                if email_data:
                    emails.append(email_data)
            except Exception as e:
                logger.error(f"Failed to parse {file_path}: {e}")
                continue
        
        logger.info(f"Parsed {len(emails)} email files")
        return emails
    
    def parse_html_file(self, file_path: Path, force=False) -> Dict[str, Any]:
        """Parse a single HTML file."""
        try:
            # Generate file hash for change detection
            file_hash = self._get_file_hash(file_path)
            
            # Check if we need to reprocess
            if not force and self._is_already_processed(file_path, file_hash):
                logger.debug(f"Skipping already processed file: {file_path}")
                return None
            
            # Read and parse the HTML file
            with open(file_path, 'r', encoding='utf-8') as f:
                file_content = f.read()
                header_content, html_content = file_content.split('==================================================')
            # Extract elements from header content
            subject_content = header_content.strip().split('\n')[0] if header_content else ''
            received_content = header_content.strip().split('\n')[2] if header_content else ''
            message_id_content = header_content.strip().split('\n')[5] if header_content else ''
            
            # Extract and separate image content from HTML
            image_content = None
            text_content = html_content
            
            # Find embedded base64 images
            img_match = re.search(r'<img[^>]*src="data:image/[^;]+;base64,([^"]+)"', html_content)
            if img_match:
                # Extract the base64 image data
                base64_data = img_match.group(1)
                try:
                    # Convert base64 to bytes
                    image_content = base64.b64decode(base64_data)
                    # Remove the img tag from text content
                    text_content = re.sub(r'<img[^>]+>', '', html_content)
                except Exception as e:
                    logger.warning(f"Failed to decode embedded image: {e}")
                        
            # Extract basic metadata
            html_data = {
                'filename': file_path.name,
                'subject': subject_content.replace('Subject: ', ''),
                'date_received': self._parse_date(received_content.replace('Received: ', '')),
                'msg_id': message_id_content.replace('Message ID: ', ''),
                'content_text': text_content,
                'content_images': image_content
            }
                        
            logger.debug(f"Successfully parsed: {file_path}")
            return html_data
            
        except Exception as e:
            logger.error(f"Failed to parse {file_path}: {e}")
            raise
    
    def _get_file_hash(self, file_path: Path) -> str:
        """Generate SHA256 hash of file content."""
        try:
            with open(file_path, 'rb') as f:
                content = f.read()
                return hashlib.sha256(content).hexdigest()
        except Exception as e:
            logger.error(f"Failed to generate hash for {file_path}: {e}")
            return ""
    
    def _is_already_processed(self, file_path: Path, file_hash: str) -> bool:
        """Check if file has already been processed."""
        # This could be enhanced with database lookup
        # For now, return False to always process
        return False
    
    def _parse_date(self, date_str: str) -> datetime:
        """Parse email date string."""
        if not date_str:
            return datetime.now()
        
        try:
            # Try different date formats
            date_formats = [
                '%a, %d %b %Y %H:%M:%S %z',
                '%d %b %Y %H:%M:%S %z',
                '%Y-%m-%d %H:%M:%S',
                '%Y-%m-%dT%H:%M:%SZ',
                '%d/%m/%Y %H:%M:%S'
            ]
            
            for fmt in date_formats:
                try:
                    return datetime.strptime(date_str, fmt)
                except ValueError:
                    continue
            
            # If all formats fail, return current time
            logger.warning(f"Could not parse date: {date_str}")
            return datetime.now()
            
        except Exception as e:
            logger.error(f"Date parsing error: {e}")
            return datetime.now()
    
    