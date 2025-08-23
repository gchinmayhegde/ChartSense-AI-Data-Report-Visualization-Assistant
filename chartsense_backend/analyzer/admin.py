
# Register your models here.
from django.contrib import admin
from .models import UploadedFile


@admin.register(UploadedFile)
class UploadedFileAdmin(admin.ModelAdmin):
    list_display = ['original_filename', 'file_size', 'uploaded_at']
    list_filter = ['uploaded_at']
    search_fields = ['original_filename']
    readonly_fields = ['uploaded_at', 'file_size']
    
    def get_readonly_fields(self, request, obj=None):
        if obj:  # Editing existing object
            return self.readonly_fields + ['file', 'original_filename']
        return self.readonly_fields