import os
import logging
import hashlib
from pathlib import Path
from datetime import datetime
from typing import List, Dict, Any
import extract_msg
from bs4 import BeautifulSoup
from django.conf import settings

logger = logging.getLogger(__name__)


class EmailParser:
    """Parser for Outlook .msg files."""
    
    def __init__(self):
        self.data_dir = Path(settings.DATA_DIR)
        self.supported_extensions = {'.msg'}
    
    def parse_all_emails(self, force=False) -> List[Dict[str, Any]]:
        """Parse all email files in the data directory."""
        emails = []
        
        if not self.data_dir.exists():
            logger.warning(f"Data directory does not exist: {self.data_dir}")
            return emails
        
        for file_path in self.data_dir.rglob('*.msg'):
            try:
                email_data = self.parse_email_file(file_path, force=force)
                if email_data:
                    emails.append(email_data)
            except Exception as e:
                logger.error(f"Failed to parse {file_path}: {e}")
                continue
        
        logger.info(f"Parsed {len(emails)} email files")
        return emails
    
    def parse_email_file(self, file_path: Path, force=False) -> Dict[str, Any]:
        """Parse a single .msg file."""
        try:
            # Generate file hash for change detection
            file_hash = self._get_file_hash(file_path)
            
            # Check if we need to reprocess
            if not force and self._is_already_processed(file_path, file_hash):
                logger.debug(f"Skipping already processed file: {file_path}")
                return None
            
            # Parse the .msg file
            msg = extract_msg.Message(file_path)
            
            # Extract basic metadata
            email_data = {
                'file_path': str(file_path),
                'file_hash': file_hash,
                'subject': msg.subject or 'Sem assunto',
                'sender': msg.sender or 'Remetente desconhecido',
                'recipient': msg.to or 'Destinatário desconhecido',
                'date_received': self._parse_date(msg.date),
                'msg_id': msg.message_id or f"msg_{file_hash[:8]}",
                'content': self._clean_content(msg.body),
                'html_content': self._clean_html_content(msg.htmlBody),
                'attachments': self._extract_attachments(msg),
                'headers': dict(msg.header) if hasattr(msg, 'header') else {}
            }
            
            # Clean up
            msg.close()
            
            logger.debug(f"Successfully parsed: {file_path}")
            return email_data
            
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
    
    def _clean_content(self, content: str) -> str:
        """Clean plain text content."""
        if not content:
            return ""
        
        # Remove extra whitespace and normalize
        lines = content.split('\n')
        cleaned_lines = []
        
        for line in lines:
            line = line.strip()
            if line:
                cleaned_lines.append(line)
        
        return '\n'.join(cleaned_lines)
    
    def _clean_html_content(self, html_content: str) -> str:
        """Clean HTML content and extract text."""
        if not html_content:
            return ""
        
        try:
            soup = BeautifulSoup(html_content, 'html.parser')
            
            # Remove script and style elements
            for script in soup(["script", "style"]):
                script.decompose()
            
            # Extract text
            text = soup.get_text()
            
            # Clean up whitespace
            lines = (line.strip() for line in text.splitlines())
            chunks = (phrase.strip() for line in lines for phrase in line.split("  "))
            text = ' '.join(chunk for chunk in chunks if chunk)
            
            return text
            
        except Exception as e:
            logger.error(f"HTML cleaning failed: {e}")
            return html_content
    
    def _extract_attachments(self, msg) -> List[Dict[str, str]]:
        """Extract attachment information."""
        attachments = []
        
        try:
            for attachment in msg.attachments:
                att_data = {
                    'filename': attachment.longFilename or attachment.shortFilename or 'unknown',
                    'content_type': getattr(attachment, 'contentType', 'unknown'),
                    'size': getattr(attachment, 'size', 0)
                }
                attachments.append(att_data)
        except Exception as e:
            logger.warning(f"Failed to extract attachments: {e}")
        
        return attachments
