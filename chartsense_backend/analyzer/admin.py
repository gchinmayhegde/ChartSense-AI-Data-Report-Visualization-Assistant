from django.contrib import admin
from .models import UploadedFile, AnalysisResult, ExtractedTable


@admin.register(UploadedFile)
class UploadedFileAdmin(admin.ModelAdmin):
    list_display = ['original_filename', 'file_size', 'uploaded_at', 'analysis_status']
    list_filter = ['uploaded_at', 'analysis_status']
    search_fields = ['original_filename']
    readonly_fields = ['uploaded_at', 'file_size']
    
    def get_readonly_fields(self, request, obj=None):
        if obj:  # Editing existing object
            return self.readonly_fields + ['file', 'original_filename']
        return self.readonly_fields


@admin.register(AnalysisResult)
class AnalysisResultAdmin(admin.ModelAdmin):
    list_display = ['uploaded_file', 'status', 'total_pages', 'tables_found', 'created_at']
    list_filter = ['status', 'created_at']
    search_fields = ['uploaded_file__original_filename']
    readonly_fields = ['created_at', 'updated_at']


@admin.register(ExtractedTable)
class ExtractedTableAdmin(admin.ModelAdmin):
    list_display = ['analysis_result', 'page_number', 'table_index', 'confidence_score']
    list_filter = ['page_number']
    search_fields = ['analysis_result__uploaded_file__original_filename']
    readonly_fields = ['extracted_at']