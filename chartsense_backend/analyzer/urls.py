from django.urls import path
from . import views

app_name = 'analyzer'

urlpatterns = [
    path('upload/', views.upload_file, name='upload_file'),
    path('upload/status/', views.upload_status, name='upload_status'),
]