import os
import logging
import base64
from datetime import datetime, timedelta
from pathlib import Path
from typing import List, Dict, Optional, Set
import json

from O365 import Account
from exchangelib import Credentials, Account as ExchangeAccount, DELEGATE, Configuration
from exchangelib.protocol import BaseProtocol, NoVerifyHTTPAdapter

# Disable SSL verification for development (remove in production)
BaseProtocol.HTTP_ADAPTER_CLS = NoVerifyHTTPAdapter

logger = logging.getLogger(__name__)


class Office365EmailSync:
    """
    Class to synchronize emails from Office 365 and save them as .msg files
    for indexing by Qdrant and use with Llama.
    """
    
    def __init__(self, outlook_folder_path: Optional[str] = None):
        """
        Initialize the Office 365 email sync.
        
        Args:
            outlook_folder_path: Path to the Outlook folder for storing .msg files
        """
        self.outlook_folder_path = Path(outlook_folder_path or os.environ.get('DATA_DIR', '/data/outlook'))
        self.outlook_folder_path.mkdir(parents=True, exist_ok=True)
        
        # Office 365 credentials from environment variables
        self.client_id = os.environ.get('OFFICE365_CLIENT_ID')
        self.client_secret = os.environ.get('OFFICE365_CLIENT_SECRET')
        self.tenant_id = os.environ.get('OFFICE365_TENANT_ID')
        self.email = os.environ.get('OFFICE365_EMAIL')
        self.password = os.environ.get('OFFICE365_PASSWORD')
        
        # Alternative: Exchange credentials
        self.exchange_email = os.environ.get('EXCHANGE_EMAIL')
        self.exchange_password = os.environ.get('EXCHANGE_PASSWORD')
        self.exchange_server = os.environ.get('EXCHANGE_SERVER', 'outlook.office365.com')
        
        if not self._validate_credentials():
            raise ValueError("Office 365 credentials not properly configured")
    
    def _validate_credentials(self) -> bool:
        """Validate that at least one set of credentials is available."""
        o365_configured = all([self.client_id, self.client_secret, self.tenant_id])
        exchange_configured = all([self.exchange_email, self.exchange_password])
        
        if not o365_configured and not exchange_configured:
            logger.error("Neither O365 nor Exchange credentials are configured")
            return False
        
        return True
    
    def _get_o365_account(self) -> Optional[Account]:
        """Get O365 account using client credentials flow."""
        try:
            if not all([self.client_id, self.client_secret, self.tenant_id]):
                return None
                
            credentials = (self.client_id, self.client_secret)
            account = Account(credentials, tenant_id=self.tenant_id)
            
            # Try to authenticate
            if account.authenticate(scopes=['Mail.Read']):
                logger.info("Successfully authenticated with O365 using client credentials")
                return account
            else:
                logger.warning("Failed to authenticate with O365 using client credentials")
                return None
                
        except Exception as e:
            logger.error(f"Error authenticating with O365: {e}")
            return None
    
    def _get_exchange_account(self) -> Optional[ExchangeAccount]:
        """Get Exchange account using username/password."""
        try:
            if not all([self.exchange_email, self.exchange_password]):
                return None
                
            credentials = Credentials(self.exchange_email, self.exchange_password)
            config = Configuration(service_endpoint=f'https://{self.exchange_server}/EWS/Exchange.asmx')
            
            account = ExchangeAccount(
                primary_smtp_address=self.exchange_email,
                credentials=credentials,
                config=config,
                access_type=DELEGATE
            )
            
            logger.info("Successfully authenticated with Exchange")
            return account
            
        except Exception as e:
            logger.error(f"Error authenticating with Exchange: {e}")
            return None
    
    def _get_email_identifier(self, email_data: Dict) -> str:
        """Generate a unique identifier for an email."""
        # Use message ID if available, otherwise create one from subject and date
        if email_data.get('message_id'):
            return email_data['message_id']
        
        # Create identifier from subject, sender, and date
        subject = email_data.get('subject', 'No Subject').replace(' ', '_')[:50]
        sender = email_data.get('sender', 'Unknown').replace('@', '_at_').replace('.', '_')
        date_str = email_data.get('date_received', datetime.now()).strftime('%Y%m%d_%H%M%S')
        
        return f"{date_str}_{sender}_{subject}"
    
    def _get_existing_files(self, days: int = 7) -> Set[str]:
        """
        Get list of existing .msg files from the last N days.
        
        Args:
            days: Number of days to look back
            
        Returns:
            Set of email identifiers (without .msg extension)
        """
        cutoff_date = datetime.now() - timedelta(days=days)
        existing_files = set()
        
        try:
            for file_path in self.outlook_folder_path.glob("*.msg"):
                # Check if file is from the last N days
                file_stat = file_path.stat()
                if datetime.fromtimestamp(file_stat.st_mtime) >= cutoff_date:
                    # Remove .msg extension to get identifier
                    identifier = file_path.stem
                    existing_files.add(identifier)
                    
        except Exception as e:
            logger.error(f"Error reading existing files: {e}")
        
        logger.info(f"Found {len(existing_files)} existing .msg files from last {days} days")
        return existing_files
    
    def _download_email_as_msg(self, email_data: Dict, account) -> Optional[str]:
        """
        Download an email and save it as a .msg file.
        
        Args:
            email_data: Dictionary containing email information
            account: O365 or Exchange account object
            
        Returns:
            Path to saved .msg file if successful, None otherwise
        """
        try:
            email_id = email_data.get('id')
            if not email_id:
                logger.warning("Email ID not found, skipping")
                return None
            
            # Get the full email message
            if hasattr(account, 'mailbox'):  # O365 account
                message = account.mailbox().get_message(email_id)
                if not message:
                    logger.warning(f"Could not retrieve message with ID: {email_id}")
                    return None
                
                # Convert to .msg format using O365
                msg_content = message.to_message()
                
            elif hasattr(account, 'inbox'):  # Exchange account
                message = account.inbox.get(id=email_id)
                if not message:
                    logger.warning(f"Could not retrieve message with ID: {email_id}")
                    return None
                
                # For Exchange, we need to create a .msg file manually
                # This is a simplified approach - in production you might want to use a library like win32com
                msg_content = self._create_msg_from_exchange_message(message)
                
            else:
                logger.error("Unknown account type")
                return None
            
            # Generate filename
            identifier = self._get_email_identifier(email_data)
            filename = f"{identifier}.msg"
            file_path = self.outlook_folder_path / filename
            
            # Save the .msg file
            with open(file_path, 'wb') as f:
                if isinstance(msg_content, bytes):
                    f.write(msg_content)
                else:
                    f.write(msg_content.encode('utf-8'))
            
            logger.info(f"Saved email as .msg file: {file_path}")
            return str(file_path)
            
        except Exception as e:
            logger.error(f"Error downloading email as .msg: {e}")
            return None
    
    def _create_msg_from_exchange_message(self, message) -> bytes:
        """
        Create a .msg file from an Exchange message.
        This is a simplified implementation - in production you might want to use win32com.
        
        Args:
            message: Exchange message object
            
        Returns:
            Bytes representing the .msg file content
        """
        # This is a placeholder implementation
        # In a real scenario, you would use win32com or similar to create proper .msg files
        # For now, we'll create a simple text representation
        
        msg_content = f"""From: {message.sender.email_address if message.sender else 'Unknown'}
To: {', '.join([r.email_address for r in message.to_recipients]) if message.to_recipients else 'Unknown'}
Subject: {message.subject or 'No Subject'}
Date: {message.datetime_received or 'Unknown'}
Message-ID: {message.message_id or 'Unknown'}

{message.body or 'No body content'}

--- This is a simplified .msg representation ---
"""
        
        return msg_content.encode('utf-8')
    
    def _get_emails_from_o365(self, days: int = 7) -> List[Dict]:
        """
        Get emails from Office 365 for the last N days.
        
        Args:
            days: Number of days to look back
            
        Returns:
            List of email data dictionaries
        """
        account = self._get_o365_account()
        if not account:
            logger.warning("Could not authenticate with O365, trying Exchange")
            return []
        
        try:
            mailbox = account.mailbox()
            query = mailbox.new_query()
            
            # Filter by date
            cutoff_date = datetime.now() - timedelta(days=days)
            query.on_attribute('receivedDateTime').greater_equal(cutoff_date)
            
            # Get messages
            messages = mailbox.get_messages(query=query, limit=1000)
            
            emails = []
            for msg in messages:
                email_data = {
                    'id': msg.object_id,
                    'subject': msg.subject,
                    'sender': msg.sender.address if msg.sender else 'Unknown',
                    'recipients': [r.address for r in msg.to_recipients] if msg.to_recipients else [],
                    'date_received': msg.received,
                    'message_id': msg.message_id,
                    'has_attachments': msg.has_attachments,
                    'body': msg.body
                }
                emails.append(email_data)
            
            logger.info(f"Retrieved {len(emails)} emails from O365 from last {days} days")
            return emails
            
        except Exception as e:
            logger.error(f"Error retrieving emails from O365: {e}")
            return []
    
    def _get_emails_from_exchange(self, days: int = 7) -> List[Dict]:
        """
        Get emails from Exchange for the last N days.
        
        Args:
            days: Number of days to look back
            
        Returns:
            List of email data dictionaries
        """
        account = self._get_exchange_account()
        if not account:
            logger.warning("Could not authenticate with Exchange")
            return []
        
        try:
            # Filter by date
            cutoff_date = datetime.now() - timedelta(days=days)
            
            # Get messages from inbox
            messages = account.inbox.filter(
                received__gte=cutoff_date
            ).order_by('-datetime_received')
            
            emails = []
            for msg in messages:
                email_data = {
                    'id': msg.id,
                    'subject': msg.subject,
                    'sender': msg.sender.email_address if msg.sender else 'Unknown',
                    'recipients': [r.email_address for r in msg.to_recipients] if msg.to_recipients else [],
                    'date_received': msg.datetime_received,
                    'message_id': msg.message_id,
                    'has_attachments': msg.has_attachments,
                    'body': msg.body
                }
                emails.append(email_data)
            
            logger.info(f"Retrieved {len(emails)} emails from Exchange from last {days} days")
            return emails
            
        except Exception as e:
            logger.error(f"Error retrieving emails from Exchange: {e}")
            return []
    
    def sync_emails(self, days: int = 7) -> Dict[str, int]:
        """
        Main method to sync emails from Office 365/Exchange and save them as .msg files.
        
        Args:
            days: Number of days to look back for emails
            
        Returns:
            Dictionary with sync statistics
        """
        logger.info(f"Starting email sync for last {days} days")
        
        # Get existing files
        existing_files = self._get_existing_files(days)
        
        # Get emails from both sources
        o365_emails = self._get_emails_from_o365(days)
        exchange_emails = self._get_emails_from_exchange(days)
        
        # Combine and deduplicate emails
        all_emails = o365_emails + exchange_emails
        unique_emails = {email['message_id']: email for email in all_emails if email.get('message_id')}.values()
        
        logger.info(f"Found {len(unique_emails)} unique emails to process")
        
        # Track sync results
        synced_count = 0
        skipped_count = 0
        error_count = 0
        
        # Process each email
        for email_data in unique_emails:
            try:
                # Check if email already exists
                identifier = self._get_email_identifier(email_data)
                if identifier in existing_files:
                    logger.debug(f"Email already exists: {identifier}")
                    skipped_count += 1
                    continue
                
                # Download and save email
                account = self._get_o365_account() or self._get_exchange_account()
                if account:
                    result = self._download_email_as_msg(email_data, account)
                    if result:
                        synced_count += 1
                        logger.info(f"Successfully synced email: {identifier}")
                    else:
                        error_count += 1
                        logger.error(f"Failed to sync email: {identifier}")
                else:
                    error_count += 1
                    logger.error("No valid account available for syncing")
                    
            except Exception as e:
                error_count += 1
                logger.error(f"Error processing email {email_data.get('subject', 'Unknown')}: {e}")
        
        # Log summary
        logger.info(f"Email sync completed. Synced: {synced_count}, Skipped: {skipped_count}, Errors: {error_count}")
        
        return {
            'synced': synced_count,
            'skipped': skipped_count,
            'errors': error_count,
            'total_processed': len(unique_emails)
        }
    
    def get_sync_status(self) -> Dict[str, any]:
        """
        Get the current sync status and statistics.
        
        Returns:
            Dictionary with sync status information
        """
        try:
            # Count total .msg files
            total_files = len(list(self.outlook_folder_path.glob("*.msg")))
            
            # Get recent files (last 7 days)
            recent_files = self._get_existing_files(7)
            
            # Check credentials status
            o365_configured = all([self.client_id, self.client_secret, self.tenant_id])
            exchange_configured = all([self.exchange_email, self.exchange_password])
            
            # Test connections
            o365_working = self._get_o365_account() is not None
            exchange_working = self._get_exchange_account() is not None
            
            return {
                'status': 'healthy' if (o365_working or exchange_working) else 'unhealthy',
                'total_msg_files': total_files,
                'recent_msg_files': len(recent_files),
                'outlook_folder_path': str(self.outlook_folder_path),
                'o365_configured': o365_configured,
                'o365_working': o365_working,
                'exchange_configured': exchange_configured,
                'exchange_working': exchange_working,
                'last_sync_check': datetime.now().isoformat()
            }
            
        except Exception as e:
            logger.error(f"Error getting sync status: {e}")
            return {
                'status': 'error',
                'error': str(e),
                'last_sync_check': datetime.now().isoformat()
            }
