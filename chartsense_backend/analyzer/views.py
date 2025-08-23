from django.shortcuts import render

# Create your views here.
import os
import logging
from rest_framework import status
from rest_framework.decorators import api_view, parser_classes
from rest_framework.parsers import MultiPartParser, FormParser
from rest_framework.response import Response
from django.conf import settings
from django.core.files.storage import default_storage
from django.core.files.base import ContentFile

from .models import UploadedFile
from .serializers import FileUploadSerializer, FileUploadResponseSerializer

# Configure logging
logger = logging.getLogger(__name__)


@api_view(['POST'])
@parser_classes([MultiPartParser, FormParser])
def upload_file(request):
    """
    Upload a PDF file to the server.
    
    Expected request format:
    - Method: POST
    - Content-Type: multipart/form-data
    - Body: file (PDF file)
    
    Returns:
    - 200: Success response with file info
    - 400: Bad request (validation errors)
    - 500: Internal server error
    """
    try:
        # Check if file was provided
        if 'file' not in request.FILES:
            return Response(
                {
                    "status": "error",
                    "message": "No file provided. Please include a 'file' field in your request."
                },
                status=status.HTTP_400_BAD_REQUEST
            )
        
        uploaded_file = request.FILES['file']
        
        # Basic file validation
        if not uploaded_file.name:
            return Response(
                {
                    "status": "error",
                    "message": "File name is required."
                },
                status=status.HTTP_400_BAD_REQUEST
            )
        
        # Check if file is PDF
        if not uploaded_file.name.lower().endswith('.pdf'):
            return Response(
                {
                    "status": "error",
                    "message": "Only PDF files are allowed."
                },
                status=status.HTTP_400_BAD_REQUEST
            )
        
        # Check file size (10MB limit)
        if uploaded_file.size > 10 * 1024 * 1024:
            return Response(
                {
                    "status": "error",
                    "message": "File size cannot exceed 10MB."
                },
                status=status.HTTP_400_BAD_REQUEST
            )
        
        # Use serializer for validation and saving
        serializer = FileUploadSerializer(data={'file': uploaded_file})
        
        if serializer.is_valid():
            # Save the file
            saved_file = serializer.save()
            
            logger.info(f"File uploaded successfully: {saved_file.original_filename}")
            
            # Return success response
            response_data = {
                "status": "uploaded",
                "filename": saved_file.original_filename,
                "file_id": saved_file.id,
                "message": f"File '{saved_file.original_filename}' uploaded successfully."
            }
            
            return Response(response_data, status=status.HTTP_200_OK)
        
        else:
            # Return validation errors
            return Response(
                {
                    "status": "error",
                    "message": "File validation failed.",
                    "errors": serializer.errors
                },
                status=status.HTTP_400_BAD_REQUEST
            )
    
    except Exception as e:
        logger.error(f"File upload error: {str(e)}")
        return Response(
            {
                "status": "error",
                "message": f"File upload failed: {str(e)}"
            },
            status=status.HTTP_500_INTERNAL_SERVER_ERROR
        )


@api_view(['GET'])
def upload_status(request):
    """
    Check the upload endpoint status.
    """
    return Response(
        {
            "status": "active",
            "endpoint": "/api/upload/",
            "methods": ["POST"],
            "supported_formats": ["PDF"],
            "max_file_size": "10MB"
        },
        status=status.HTTP_200_OK
    )