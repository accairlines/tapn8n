from django.contrib import admin
from .models import EmailDocument


@admin.register(EmailDocument)
class EmailDocumentAdmin(admin.ModelAdmin):
    list_display = ['subject', 'sender', 'date_received', 'processed_at', 'chunk_count']
    list_filter = ['processed_at', 'date_received', 'sender']
    search_fields = ['subject', 'sender', 'recipient', 'msg_id']
    readonly_fields = ['processed_at', 'last_indexed']
    
    def has_add_permission(self, request):
        return False  # Documents are added automatically during processing

