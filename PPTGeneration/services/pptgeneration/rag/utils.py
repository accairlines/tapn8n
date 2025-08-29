import logging
from typing import Dict, Any, Optional

logger = logging.getLogger(__name__)


def log_processing(level: str, message: str, details: Optional[Dict[str, Any]] = None):
    """Log processing activity to database and console."""
    try:
        # Log to console
        log_func = getattr(logger, level.lower(), logger.info)
        if details:
            log_func(f"{message} - Details: {details}")
        else:
            log_func(message)
            
    except Exception as e:
        # Fallback to console logging only
        logger.error(f"Failed to log to database: {e}")
        logger.log(getattr(logging, level.upper(), logging.INFO), message)


def format_file_size(size_bytes: int) -> str:
    """Format file size in human readable format."""
    if size_bytes == 0:
        return "0B"
    
    size_names = ["B", "KB", "MB", "GB"]
    i = 0
    while size_bytes >= 1024 and i < len(size_names) - 1:
        size_bytes /= 1024.0
        i += 1
    
    return f"{size_bytes:.1f}{size_names[i]}"


def sanitize_filename(filename: str) -> str:
    """Sanitize filename for safe file operations."""
    import re
    # Remove or replace unsafe characters
    filename = re.sub(r'[<>:"/\\|?*]', '_', filename)
    # Remove leading/trailing spaces and dots
    filename = filename.strip(' .')
    # Limit length
    if len(filename) > 200:
        filename = filename[:200]
    return filename


def validate_email_content(content: str) -> bool:
    """Validate email content for processing."""
    if not content or not content.strip():
        return False
    
    # Check minimum content length
    if len(content.strip()) < 10:
        return False
    
    # Check for common spam indicators (basic)
    spam_indicators = [
        'buy now', 'click here', 'limited time', 'act now',
        'free money', 'lottery', 'winner', 'urgent'
    ]
    
    content_lower = content.lower()
    spam_score = sum(1 for indicator in spam_indicators if indicator in content_lower)
    
    # If too many spam indicators, mark as invalid
    if spam_score > 3:
        return False
    
    return True


def extract_key_phrases(text: str, max_phrases: int = 5) -> list:
    """Extract key phrases from text (basic implementation)."""
    import re
    
    # Remove common stop words
    stop_words = {
        'a', 'o', 'e', 'de', 'da', 'do', 'em', 'um', 'uma', 'para', 'por',
        'com', 'sem', 'na', 'no', 'se', 'que', 'este', 'esta', 'isto',
        'the', 'a', 'an', 'and', 'or', 'but', 'in', 'on', 'at', 'to', 'for',
        'of', 'with', 'by', 'is', 'are', 'was', 'were', 'be', 'been', 'being'
    }
    
    # Clean text and split into words
    words = re.findall(r'\b\w+\b', text.lower())
    
    # Filter out stop words and short words
    filtered_words = [word for word in words if word not in stop_words and len(word) > 3]
    
    # Count word frequency
    word_freq = {}
    for word in filtered_words:
        word_freq[word] = word_freq.get(word, 0) + 1
    
    # Sort by frequency and return top phrases
    sorted_words = sorted(word_freq.items(), key=lambda x: x[1], reverse=True)
    return [word for word, freq in sorted_words[:max_phrases]]


def calculate_processing_metrics(start_time, end_time, items_processed: int) -> Dict[str, Any]:
    """Calculate processing performance metrics."""
    processing_time = end_time - start_time
    
    metrics = {
        'processing_time_seconds': processing_time,
        'items_processed': items_processed,
        'items_per_second': items_processed / processing_time if processing_time > 0 else 0,
        'start_time': start_time.isoformat(),
        'end_time': end_time.isoformat()
    }
    
    return metrics


def get_system_info() -> Dict[str, Any]:
    """Get system information for monitoring."""
    import platform
    import psutil
    
    try:
        system_info = {
            'platform': platform.system(),
            'platform_version': platform.version(),
            'python_version': platform.python_version(),
            'cpu_count': psutil.cpu_count(),
            'memory_total_gb': round(psutil.virtual_memory().total / (1024**3), 2),
            'memory_available_gb': round(psutil.virtual_memory().available / (1024**3), 2),
            'disk_usage_percent': psutil.disk_usage('/').percent
        }
        return system_info
    except ImportError:
        # psutil not available
        return {
            'platform': platform.system(),
            'platform_version': platform.version(),
            'python_version': platform.python_version(),
            'note': 'psutil not available for detailed system info'
        }
    except Exception as e:
        return {
            'error': f'Failed to get system info: {str(e)}'
        }
