import os
import logging
import base64
from datetime import datetime, timedelta
from pathlib import Path
from typing import List, Dict, Optional, Set
import json
import requests
import msal

logger = logging.getLogger(__name__)


class Office365EmailSync:
    """
    Class to synchronize emails from Office 365 and save them as .msg files
    for indexing by Qdrant and use with Llama.
    """
    
    # Configuration
    tenant_id = "your_tenant_id"
    client_id = "your_client_id"
    client_secret = "your_client_secret"
    shared_mailbox = "user@example.com"

    def __init__(self, outlook_folder_path: Optional[str] = None):
        """
        Initialize the Office 365 email sync.
        
        Args:
            tenant_id: Azure AD tenant ID
            client_id: Azure AD client ID
            client_secret: Azure AD client secret
            shared_mailbox: Shared mailbox email address
            outlook_folder_path: Path to the Outlook folder for storing .msg files
        """
        self.tenant_id = os.environ.get('OFFICE365_TENANT_ID')
        self.client_id = os.environ.get('OFFICE365_CLIENT_ID')
        self.client_secret = os.environ.get('OFFICE365_CLIENT_SECRET')
        self.shared_mailbox = os.environ.get('OFFICE365_EMAIL')
        self.authority = f"https://login.microsoftonline.com/{self.tenant_id}"
        self.scopes = ["https://graph.microsoft.com/.default"]
        
        # Initialize the MSAL client
        self.app = msal.ConfidentialClientApplication(
            client_id=self.client_id,
            client_credential=self.client_secret,
            authority=self.authority,
            token_cache=None  # Disable token cache to ensure fresh token
        )
        
        self.outlook_folder_path = Path(outlook_folder_path or os.environ.get('DATA_DIR', ''))
        self.outlook_folder_path.mkdir(parents=True, exist_ok=True)
        
        if not self._validate_credentials():
            raise ValueError("Office 365 credentials not properly configured")
    
    def _validate_credentials(self) -> bool:
        """Validate that Office 365 credentials are available."""
        o365_configured = all([self.client_id, self.client_secret, self.tenant_id])
        
        if not o365_configured:
            logger.error("Office 365 credentials are not configured")
            return False
        
        return True
    
    def get_access_token(self):
        """Acquire access token using client credentials flow"""
        # Always get a fresh token
        result = self.app.acquire_token_for_client(self.scopes)
        
        if "access_token" not in result:
            error_msg = f"Token acquisition failed: {result.get('error_description', 'Unknown error')}"
            logging.error(error_msg)
            raise Exception(error_msg)
        
        logging.info("Successfully acquired access token")
        return result["access_token"]
    
    def get_full_message(self, mailbox, message_id: str, access_token):
        """
        Fetch the complete message including full body content
        
        Args:
            message_id: The ID of the message to fetch
            
        Returns:
            dict: Complete message data including full body
        """
        try:
            # Specify select parameter to get full body content
            endpoint = f"https://graph.microsoft.com/v1.0/users/{mailbox}/messages/{message_id}"
            params = {
                '$select': 'subject,body,from,receivedDateTime,hasAttachments,id'
            }
            headers = {
                'Authorization': f'Bearer {access_token}',
                'Content-Type': 'application/json'
            }
            
            response = requests.get(endpoint, headers=headers, params=params)
            response.raise_for_status()
            
            message_data = response.json()
            
            # Extract the full body content
            return {
                'subject': message_data.get('subject', ''),
                'body': message_data.get('body', {}).get('content', ''),  # Get full body content
                'body_type': message_data.get('body', {}).get('contentType', ''),  # html or text
                'from_email': message_data.get('from', {}).get('emailAddress', {}).get('address', ''),
                'received_time': message_data.get('receivedDateTime', ''),
                'has_attachments': message_data.get('hasAttachments', False),
                'message_id': message_data.get('id', '')
            }
            
        except requests.exceptions.HTTPError as http_err:
            logging.error(f"HTTP error occurred: {http_err}")
            raise
        except Exception as e:
            logging.error(f"Error fetching full message: {str(e)}")
            raise
    
    def read_emails(self, access_token, mailbox, folder="inbox", top=10, days_back=0, to_address=None, subject=None):
        """
        Read emails from shared mailbox
        
        Args:
            mailbox (str): Shared mailbox email address
            folder (str): Folder to read from (default: inbox)
            top (int): Number of emails to retrieve
            days_back (int): Number of days back to search
        """
        try:
            # Calculate date filter
            date_filter = datetime.now() - timedelta(days=days_back)
            date_filter_str = date_filter.strftime("%Y-%m-%dT%H:%M:%SZ")
            
            # Use /beta endpoint for shared mailbox access
            endpoint = f"https://graph.microsoft.com/v1.0/users/{mailbox}/mailFolders/{folder}/messages"
            
            headers = {
                'Authorization': f'Bearer {access_token}',
                'Content-Type': 'application/json',
                'ConsistencyLevel': 'eventual'
            }
            
            # Build params safely
            addr = to_address.replace("'", "''")  # escape single quotes
            params = {
                "$filter": f"receivedDateTime ge {date_filter_str} and startswith(subject,'IOC Daily Summary')",
                "$select": "id,subject,receivedDateTime,from,bodyPreview,hasAttachments",
                "$orderby": "receivedDateTime desc",
                "$top": top
            }
            
            logging.info(f"Requesting emails from endpoint: {endpoint}")
            response = requests.get(endpoint, headers=headers, params=params)
            
            if response.status_code != 200:
                logging.error(f"Request failed with status {response.status_code} and body: {response.text}")
                response.raise_for_status()
            
            emails = response.json().get('value', [])
            logging.info(f"Successfully retrieved {len(emails)} emails")
            
            return [{
                'id': email.get('id'),
                'subject': email.get('subject'),
                'from': email.get('from', {}).get('emailAddress', {}).get('name'),
                'received': email.get('receivedDateTime'),
                'preview': email.get('bodyPreview'),
                'has_attachments': email.get('hasAttachments')
            } for email in emails]
            
        except requests.exceptions.HTTPError as http_err:
            logging.error(f"HTTP error occurred: {http_err}")
            if hasattr(http_err.response, 'text'):
                logging.error(f"Response content: {http_err.response.text}")
            raise
        except Exception as e:
            logging.error(f"An error occurred: {str(e)}")
            raise

    def _get_email_identifier(self, email_data: Dict) -> str:
        """Generate a unique identifier for an email."""
        # Use message ID if available, otherwise create one from subject and date
        if email_data.get('message_id'):
            return email_data['message_id']
        
        # Create identifier from subject, sender, and date
        subject = email_data.get('subject', 'No Subject').replace(' ', '_')[:50]
        sender = email_data.get('from', 'Unknown').replace(' ', '_')
        date_str = email_data.get('received', '2000-01-01T00:00:00Z')
        
        # Sanitize the identifier to be Windows filename safe
        # Replace colons and other invalid characters
        identifier = f"{date_str}_{sender}_{subject}"
        # Replace invalid Windows filename characters
        invalid_chars = ['<', '>', ':', '"', '|', '?', '*', '\\', '/']
        for char in invalid_chars:
            identifier = identifier.replace(char, '_')
        
        return identifier
    
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
    
    def _download_email_as_msg(self, email_data: Dict, access_token: str) -> Optional[str]:
        """
        Download an email and save it as a .msg file.
        
        Args:
            email_data: Dictionary containing email information
            access_token: Access token for Microsoft Graph API
            
        Returns:
            Path to saved .msg file if successful, None otherwise
        """
        try:
            email_id = email_data.get('id')
            if not email_id:
                logger.warning("Email ID not found, skipping")
                return None
            
            # Get the full email message using Microsoft Graph API
            full_message = self.get_full_message(self.shared_mailbox, email_id, access_token)
            if not full_message:
                logger.warning(f"Could not retrieve message with ID: {email_id}")
                return None
            
            # Generate filename
            identifier = self._get_email_identifier(email_data)
            filename = f"{identifier}.msg"
            file_path = self.outlook_folder_path / filename
            
            # Save the email content as a simple text file (since we don't have .msg conversion)
            # You might want to implement proper .msg conversion here
            with open(file_path, 'w', encoding='utf-8') as f:
                f.write(f"Subject: {full_message.get('subject', '')}\n")
                f.write(f"From: {full_message.get('from_email', '')}\n")
                f.write(f"Received: {full_message.get('received_time', '')}\n")
                f.write(f"Body Type: {full_message.get('body_type', '')}\n")
                f.write(f"Has Attachments: {full_message.get('has_attachments', False)}\n")
                f.write(f"Message ID: {full_message.get('message_id', '')}\n")
                f.write("\n" + "="*50 + "\n\n")
                f.write(full_message.get('body', ''))
            
            logger.info(f"Saved email as .msg file: {file_path}")
            return str(file_path)
            
        except Exception as e:
            logger.error(f"Error downloading email as .msg: {e}")
            return None
    
    def _get_emails_from_o365(self, days: int = 7) -> List[Dict]:
        """
        Get emails from Office 365 for the last N days.
        Only returns emails with subjects like "IOC Daily Summary" that were sent FROM the current user.
        
        Args:
            days: Number of days to look back
            
        Returns:
            List of email data dictionaries
        """
        try:
            access_token = self.get_access_token()
            
            # Read emails from sent items folder (emails sent by the user)
            emails = self.read_emails(
                access_token,
                mailbox=self.shared_mailbox,
                top=800,  # Only fetch what you need
                days_back=730,  # ~2 years
                folder='sentitems',
                to_address='ioc.dailysummary@tap.pt',
                subject='IOC Daily Summary'
            )
            
            # Filter emails to only include those with "IOC Daily Summary" in the subject
            filtered_emails = []
            for email in emails:
                subject = email.get('subject', '')
                if subject and 'IOC Daily Summary' in subject:
                    filtered_emails.append(email)
            
            logger.info(f"Retrieved {len(emails)} emails from O365 from last {days} days")
            logger.info(f"Filtered to {len(filtered_emails)} emails with 'IOC Daily Summary' in subject")
            return filtered_emails
            
        except Exception as e:
            logger.error(f"Error retrieving emails from O365: {e}")
            return []
    
    def sync_emails(self, days: int = 7) -> Dict[str, int]:
        """
        Main method to sync emails from Office 365 and save them as .msg files.
        Only syncs emails with subjects like "IOC Daily Summary" that were sent FROM the current user.
        
        Args:
            days: Number of days to look back for emails
            
        Returns:
            Dictionary with sync statistics
        """
        logger.info(f"Starting email sync for last {days} days")
        
        # Get existing files
        existing_files = self._get_existing_files(days)
        
        # Get emails from Office 365
        o365_emails = self._get_emails_from_o365(days)
        
        logger.info(f"Found {len(o365_emails)} emails to process")
        
        # Track sync results
        synced_count = 0
        skipped_count = 0
        error_count = 0
        
        # Process each email
        for email_data in o365_emails:
            try:
                # Check if email already exists
                identifier = self._get_email_identifier(email_data)
                if identifier in existing_files:
                    logger.debug(f"Email already exists: {identifier}")
                    skipped_count += 1
                    continue
                
                # Download and save email
                access_token = self.get_access_token()
                if access_token:
                    result = self._download_email_as_msg(email_data, access_token)
                    if result:
                        synced_count += 1
                        logger.info(f"Successfully synced email: {identifier}")
                    else:
                        error_count += 1
                        logger.error(f"Failed to sync email: {identifier}")
                else:
                    error_count += 1
                    logger.error("No valid access token available for syncing")
                    
            except Exception as e:
                error_count += 1
                logger.error(f"Error processing email {email_data.get('subject', 'Unknown')}: {e}")
        
        # Log summary
        logger.info(f"Email sync completed. Synced: {synced_count}, Skipped: {skipped_count}, Errors: {error_count}")
        
        return {
            'synced': synced_count,
            'skipped': skipped_count,
            'errors': error_count,
            'total_processed': len(o365_emails)
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
            
            # Test connection
            try:
                access_token = self.get_access_token()
                o365_working = access_token is not None
            except:
                o365_working = False
            
            return {
                'status': 'healthy' if o365_working else 'unhealthy',
                'total_msg_files': total_files,
                'recent_msg_files': len(recent_files),
                'outlook_folder_path': str(self.outlook_folder_path),
                'o365_configured': o365_configured,
                'o365_working': o365_working,
                'last_sync_check': datetime.now().isoformat()
            }
            
        except Exception as e:
            logger.error(f"Error getting sync status: {e}")
            return {
                'status': 'error',
                'error': str(e),
                'last_sync_check': datetime.now().isoformat()
            }
