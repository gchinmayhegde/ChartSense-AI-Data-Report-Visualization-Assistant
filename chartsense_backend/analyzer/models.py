import os
import json
from django.db import models
from django.core.validators import FileExtensionValidator


def upload_to_media(instance, filename):
    """Generate upload path for files"""
    return os.path.join('uploads', filename)


class UploadedFile(models.Model):
    """Model to store uploaded PDF files"""
    ANALYSIS_STATUS_CHOICES = [
        ('pending', 'Pending'),
        ('processing', 'Processing'),
        ('completed', 'Completed'),
        ('failed', 'Failed'),
    ]
    
    file = models.FileField(
        upload_to=upload_to_media,
        validators=[FileExtensionValidator(allowed_extensions=['pdf'])],
        help_text="Only PDF files are allowed"
    )
    original_filename = models.CharField(max_length=255)
    uploaded_at = models.DateTimeField(auto_now_add=True)
    file_size = models.PositiveIntegerField(help_text="File size in bytes")
    analysis_status = models.CharField(
        max_length=20, 
        choices=ANALYSIS_STATUS_CHOICES, 
        default='pending',
        help_text="Current analysis status"
    )
    
    class Meta:
        ordering = ['-uploaded_at']
    
    def __str__(self):
        return f"{self.original_filename} - {self.uploaded_at.strftime('%Y-%m-%d %H:%M')}"
    
    def delete(self, *args, **kwargs):
        """Delete file from storage when model instance is deleted"""
        if self.file:
            if os.path.isfile(self.file.path):
                os.remove(self.file.path)
        super().delete(*args, **kwargs)


class AnalysisResult(models.Model):
    """Model to store analysis results for uploaded files"""
    STATUS_CHOICES = [
        ('pending', 'Pending'),
        ('processing', 'Processing'),
        ('completed', 'Completed'),
        ('failed', 'Failed'),
    ]
    
    uploaded_file = models.OneToOneField(
        UploadedFile, 
        on_delete=models.CASCADE,
        related_name='analysis_result'
    )
    status = models.CharField(max_length=20, choices=STATUS_CHOICES, default='pending')
    total_pages = models.PositiveIntegerField(default=0)
    pages_processed = models.PositiveIntegerField(default=0)
    tables_found = models.PositiveIntegerField(default=0)
    error_message = models.TextField(blank=True, null=True)
    processing_time = models.FloatField(
        null=True, 
        blank=True,
        help_text="Processing time in seconds"
    )
    created_at = models.DateTimeField(auto_now_add=True)
    updated_at = models.DateTimeField(auto_now=True)
    
    class Meta:
        ordering = ['-created_at']
    
    def __str__(self):
        return f"Analysis for {self.uploaded_file.original_filename} - {self.status}"


class ExtractedTable(models.Model):
    """Model to store individual extracted tables"""
    analysis_result = models.ForeignKey(
        AnalysisResult, 
        on_delete=models.CASCADE,
        related_name='extracted_tables'
    )
    page_number = models.PositiveIntegerField()
    table_index = models.PositiveIntegerField(help_text="Index of table on the page")
    bounding_box = models.JSONField(
        help_text="Coordinates of the table bounding box"
    )
    table_data = models.JSONField(
        help_text="Structured table data as JSON"
    )
    confidence_score = models.FloatField(
        null=True, 
        blank=True,
        help_text="Confidence score from the detection model"
    )
    extracted_at = models.DateTimeField(auto_now_add=True)
    
    class Meta:
        ordering = ['page_number', 'table_index']
        unique_together = ['analysis_result', 'page_number', 'table_index']
    
    def __str__(self):
        return f"Table {self.table_index} from page {self.page_number} - {self.analysis_result.uploaded_file.original_filename}"
    
    @property
    def table_summary(self):
        """Return a summary of the table structure"""
        if not self.table_data:
            return "No data"
        
        try:
            rows = len(self.table_data.get('rows', []))
            cols = len(self.table_data.get('columns', []))
            return f"{rows} rows × {cols} columns"
        except:
            return "Structure unknown"