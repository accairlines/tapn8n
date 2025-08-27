from django.db import models
from django.utils import timezone


class EmailDocument(models.Model):
    """Model to track processed email documents."""
    
    file_path = models.CharField(max_length=500, unique=True)
    subject = models.CharField(max_length=500, blank=True)
    sender = models.CharField(max_length=200, blank=True)
    recipient = models.CharField(max_length=200, blank=True)
    date_received = models.DateTimeField(null=True, blank=True)
    msg_id = models.CharField(max_length=200, blank=True)
    content_length = models.IntegerField(default=0)
    chunk_count = models.IntegerField(default=0)
    processed_at = models.DateTimeField(default=timezone.now)
    last_indexed = models.DateTimeField(default=timezone.now)
    
    class Meta:
        ordering = ['-processed_at']
    
    def __str__(self):
        return f"{self.subject} - {self.sender} ({self.date_received})"


class ProcessingLog(models.Model):
    """Model to log processing activities."""
    
    LEVEL_CHOICES = [
        ('INFO', 'Info'),
        ('WARNING', 'Warning'),
        ('ERROR', 'Error'),
    ]
    
    timestamp = models.DateTimeField(default=timezone.now)
    level = models.CharField(max_length=10, choices=LEVEL_CHOICES, default='INFO')
    message = models.TextField()
    details = models.JSONField(default=dict, blank=True)
    
    class Meta:
        ordering = ['-timestamp']
    
    def __str__(self):
        return f"{self.timestamp} - {self.level}: {self.message}"
