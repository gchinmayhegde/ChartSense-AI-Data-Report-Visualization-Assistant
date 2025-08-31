from django.urls import path
from . import views

app_name = 'analyzer'

urlpatterns = [
    path('upload/', views.upload_file, name='upload_file'),
    path('upload/status/', views.upload_status, name='upload_status'),
    
    # Phase 2 endpoints (new)
    path('analyze/', views.analyze_file, name='analyze_file'),
    path('analysis/<int:analysis_id>/', views.get_analysis_result, name='get_analysis_result'),
    path('analysis/<int:analysis_id>/tables/', views.get_extracted_tables, name='get_extracted_tables'),
    path('files/', views.list_uploaded_files, name='list_uploaded_files'),
    path('files/<int:file_id>/', views.get_file_details, name='get_file_details'),
]