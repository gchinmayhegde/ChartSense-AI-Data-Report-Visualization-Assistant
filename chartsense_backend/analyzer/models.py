import os
from django.db import models
from django.core.validators import FileExtensionValidator


def upload_to_media(instance, filename):
    """Generate upload path for files"""
    return os.path.join('uploads', filename)


class UploadedFile(models.Model):
    """Model to store uploaded PDF files"""
    file = models.FileField(
        upload_to=upload_to_media,
        validators=[FileExtensionValidator(allowed_extensions=['pdf'])],
        help_text="Only PDF files are allowed"
    )
    original_filename = models.CharField(max_length=255)
    uploaded_at = models.DateTimeField(auto_now_add=True)
    file_size = models.PositiveIntegerField(help_text="File size in bytes")
    
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