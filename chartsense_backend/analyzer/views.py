import os
import time
import logging
from rest_framework import status
from rest_framework.decorators import api_view, parser_classes
from rest_framework.parsers import MultiPartParser, FormParser
from rest_framework.response import Response
from django.conf import settings
from django.shortcuts import get_object_or_404
from django.db import transaction

from .models import UploadedFile, AnalysisResult, ExtractedTable
from .serializers import (
    FileUploadSerializer, FileUploadResponseSerializer,
    AnalyzeRequestSerializer, AnalyzeResponseSerializer,
    AnalysisResultSerializer, ExtractedTableSerializer
)
from .utils.table_extractor import TableExtractor

# Configure logging
logger = logging.getLogger(__name__)


# ===== PHASE 1 ENDPOINTS (EXISTING) =====

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


# ===== PHASE 2 ENDPOINTS (NEW) =====

@api_view(['POST'])
def analyze_file(request):
    """
    Analyze an uploaded PDF file to extract tables.
    
    Expected request format:
    - Method: POST
    - Content-Type: application/json
    - Body: {"file_id": <int>}
    
    Returns:
    - 200: Analysis started successfully
    - 400: Bad request (validation errors)
    - 404: File not found
    - 500: Internal server error
    """
    try:
        # Validate request data
        serializer = AnalyzeRequestSerializer(data=request.data)
        if not serializer.is_valid():
            return Response(
                {
                    "status": "error",
                    "message": "Invalid request data.",
                    "errors": serializer.errors
                },
                status=status.HTTP_400_BAD_REQUEST
            )
        
        file_id = serializer.validated_data['file_id']
        uploaded_file = get_object_or_404(UploadedFile, id=file_id)
        
        # Check Hugging Face API key
        if not settings.HUGGINGFACE_API_KEY:
            return Response(
                {
                    "status": "error",
                    "message": "Hugging Face API key not configured. Please set HUGGINGFACE_API_KEY in settings."
                },
                status=status.HTTP_500_INTERNAL_SERVER_ERROR
            )
        
        # Create or get existing analysis result
        analysis_result, created = AnalysisResult.objects.get_or_create(
            uploaded_file=uploaded_file,
            defaults={'status': 'pending'}
        )
        
        if not created and analysis_result.status in ['processing', 'completed']:
            return Response(
                {
                    "status": "info",
                    "message": f"File analysis is already {analysis_result.status}.",
                    "analysis_id": analysis_result.id,
                    "file_id": file_id
                },
                status=status.HTTP_200_OK
            )
        
        # Update file and analysis status
        uploaded_file.analysis_status = 'processing'
        uploaded_file.save()
        
        analysis_result.status = 'processing'
        analysis_result.save()
        
        # Initialize table extractor
        extractor = TableExtractor()
        
        try:
            # Process the PDF file
            start_time = time.time()
            
            # Extract tables from the PDF
            tables_data = extractor.extract_tables_from_pdf(uploaded_file.file.path)
            
            processing_time = time.time() - start_time
            
            # Save results to database
            with transaction.atomic():
                # Update analysis result
                analysis_result.status = 'completed'
                analysis_result.total_pages = tables_data.get('total_pages', 0)
                analysis_result.pages_processed = tables_data.get('pages_processed', 0)
                analysis_result.tables_found = len(tables_data.get('tables', []))
                analysis_result.processing_time = processing_time
                analysis_result.save()
                
                # Save extracted tables
                for table_info in tables_data.get('tables', []):
                    ExtractedTable.objects.create(
                        analysis_result=analysis_result,
                        page_number=table_info['page_number'],
                        table_index=table_info['table_index'],
                        bounding_box=table_info['bounding_box'],
                        table_data=table_info['table_data'],
                        confidence_score=table_info.get('confidence_score')
                    )
                
                # Update file status
                uploaded_file.analysis_status = 'completed'
                uploaded_file.save()
            
            logger.info(f"Analysis completed for file {uploaded_file.original_filename}. Found {analysis_result.tables_found} tables in {processing_time:.2f}s")
            
            return Response(
                {
                    "status": "completed",
                    "message": f"Analysis completed successfully. Found {analysis_result.tables_found} tables.",
                    "analysis_id": analysis_result.id,
                    "file_id": file_id,
                    "tables_found": analysis_result.tables_found,
                    "processing_time": f"{processing_time:.2f}s"
                },
                status=status.HTTP_200_OK
            )
        
        except Exception as e:
            # Handle analysis errors
            logger.error(f"Analysis error for file {uploaded_file.original_filename}: {str(e)}")
            
            analysis_result.status = 'failed'
            analysis_result.error_message = str(e)
            analysis_result.save()
            
            uploaded_file.analysis_status = 'failed'
            uploaded_file.save()
            
            return Response(
                {
                    "status": "error",
                    "message": f"Analysis failed: {str(e)}",
                    "analysis_id": analysis_result.id,
                    "file_id": file_id
                },
                status=status.HTTP_500_INTERNAL_SERVER_ERROR
            )
    
    except Exception as e:
        logger.error(f"Analyze file endpoint error: {str(e)}")
        return Response(
            {
                "status": "error",
                "message": f"Request processing failed: {str(e)}"
            },
            status=status.HTTP_500_INTERNAL_SERVER_ERROR
        )


@api_view(['GET'])
def get_analysis_result(request, analysis_id):
    """
    Get analysis result by ID.
    
    Returns:
    - 200: Analysis result data
    - 404: Analysis not found
    - 500: Internal server error
    """
    try:
        analysis_result = get_object_or_404(AnalysisResult, id=analysis_id)
        serializer = AnalysisResultSerializer(analysis_result)
        return Response(serializer.data, status=status.HTTP_200_OK)
    
    except Exception as e:
        logger.error(f"Get analysis result error: {str(e)}")
        return Response(
            {
                "status": "error",
                "message": f"Failed to retrieve analysis result: {str(e)}"
            },
            status=status.HTTP_500_INTERNAL_SERVER_ERROR
        )


@api_view(['GET'])
def get_extracted_tables(request, analysis_id):
    """
    Get extracted tables for a specific analysis.
    
    Returns:
    - 200: List of extracted tables
    - 404: Analysis not found
    - 500: Internal server error
    """
    try:
        analysis_result = get_object_or_404(AnalysisResult, id=analysis_id)
        tables = ExtractedTable.objects.filter(analysis_result=analysis_result)
        serializer = ExtractedTableSerializer(tables, many=True)
        
        return Response(
            {
                "analysis_id": analysis_id,
                "filename": analysis_result.uploaded_file.original_filename,
                "tables_count": len(serializer.data),
                "tables": serializer.data
            },
            status=status.HTTP_200_OK
        )
    
    except Exception as e:
        logger.error(f"Get extracted tables error: {str(e)}")
        return Response(
            {
                "status": "error",
                "message": f"Failed to retrieve extracted tables: {str(e)}"
            },
            status=status.HTTP_500_INTERNAL_SERVER_ERROR
        )


@api_view(['GET'])
def list_uploaded_files(request):
    """
    List all uploaded files with their analysis status.
    
    Returns:
    - 200: List of uploaded files
    - 500: Internal server error
    """
    try:
        files = UploadedFile.objects.all()
        files_data = []
        
        for file in files:
            file_data = {
                "id": file.id,
                "filename": file.original_filename,
                "file_size": file.file_size,
                "uploaded_at": file.uploaded_at,
                "analysis_status": file.analysis_status,
                "analysis_result": None
            }
            
            # Add analysis result if available
            if hasattr(file, 'analysis_result'):
                analysis = file.analysis_result
                file_data["analysis_result"] = {
                    "id": analysis.id,
                    "status": analysis.status,
                    "tables_found": analysis.tables_found,
                    "processing_time": analysis.processing_time,
                    "created_at": analysis.created_at
                }
            
            files_data.append(file_data)
        
        return Response(
            {
                "files_count": len(files_data),
                "files": files_data
            },
            status=status.HTTP_200_OK
        )
    
    except Exception as e:
        logger.error(f"List uploaded files error: {str(e)}")
        return Response(
            {
                "status": "error",
                "message": f"Failed to retrieve files: {str(e)}"
            },
            status=status.HTTP_500_INTERNAL_SERVER_ERROR
        )


@api_view(['GET'])
def get_file_details(request, file_id):
    """
    Get detailed information about a specific uploaded file.
    
    Returns:
    - 200: File details with analysis results
    - 404: File not found
    - 500: Internal server error
    """
    try:
        uploaded_file = get_object_or_404(UploadedFile, id=file_id)
        
        file_data = {
            "id": uploaded_file.id,
            "filename": uploaded_file.original_filename,
            "file_size": uploaded_file.file_size,
            "uploaded_at": uploaded_file.uploaded_at,
            "analysis_status": uploaded_file.analysis_status,
            "analysis_result": None
        }
        
        # Add detailed analysis result if available
        if hasattr(uploaded_file, 'analysis_result'):
            serializer = AnalysisResultSerializer(uploaded_file.analysis_result)
            file_data["analysis_result"] = serializer.data
        
        return Response(file_data, status=status.HTTP_200_OK)
    
    except Exception as e:
        logger.error(f"Get file details error: {str(e)}")
        return Response(
            {
                "status": "error",
                "message": f"Failed to retrieve file details: {str(e)}"
            },
            status=status.HTTP_500_INTERNAL_SERVER_ERROR
        )