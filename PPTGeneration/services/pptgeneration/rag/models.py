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
    image_count = models.IntegerField(default=0)  # Number of images in email
    processed_at = models.DateTimeField(default=timezone.now)
    last_indexed = models.DateTimeField(default=timezone.now)
    
    class Meta:
        ordering = ['-processed_at']
    
    def __str__(self):
        return f"{self.subject} - {self.sender} ({self.date_received})"


class EmailImage(models.Model):
    """Model to track processed email images."""
    
    email_document = models.ForeignKey(EmailDocument, on_delete=models.CASCADE, related_name='images')
    mime_type = models.CharField(max_length=100)
    size_bytes = models.IntegerField()
    caption = models.TextField(blank=True)
    ocr_text = models.TextField(blank=True)
    entities = models.JSONField(default=list)
    embedding_vector = models.JSONField(default=list)  # Store as JSON for flexibility
    processed_at = models.DateTimeField(default=timezone.now)
    
    class Meta:
        ordering = ['-processed_at']
    
    def __str__(self):
        return f"Image from {self.email_document.subject} ({self.mime_type})"

