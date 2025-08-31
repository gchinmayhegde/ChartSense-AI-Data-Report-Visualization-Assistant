import os
import tempfile
from django.test import TestCase
from django.core.files.uploadedfile import SimpleUploadedFile
from django.urls import reverse
from rest_framework.test import APITestCase
from rest_framework import status
from unittest.mock import patch, MagicMock

from .models import UploadedFile, AnalysisResult, ExtractedTable


class UploadedFileModelTest(TestCase):
    """Test cases for UploadedFile model"""
    
    def setUp(self):
        self.test_file = SimpleUploadedFile(
            "test.pdf",
            b"fake pdf content",
            content_type="application/pdf"
        )
    
    def test_create_uploaded_file(self):
        """Test creating an uploaded file"""
        uploaded_file = UploadedFile.objects.create(
            file=self.test_file,
            original_filename="test.pdf",
            file_size=1000
        )
        
        self.assertEqual(uploaded_file.original_filename, "test.pdf")
        self.assertEqual(uploaded_file.file_size, 1000)
        self.assertEqual(uploaded_file.analysis_status, 'pending')
    
    def test_string_representation(self):
        """Test string representation of uploaded file"""
        uploaded_file = UploadedFile.objects.create(
            file=self.test_file,
            original_filename="test.pdf",
            file_size=1000
        )
        
        self.assertIn("test.pdf", str(uploaded_file))


class FileUploadAPITest(APITestCase):
    """Test cases for file upload API"""
    
    def test_upload_pdf_success(self):
        """Test successful PDF upload"""
        test_file = SimpleUploadedFile(
            "test.pdf",
            b"fake pdf content",
            content_type="application/pdf"
        )
        
        url = reverse('analyzer:upload_file')
        response = self.client.post(url, {'file': test_file}, format='multipart')
        
        self.assertEqual(response.status_code, status.HTTP_200_OK)
        self.assertEqual(response.data['status'], 'uploaded')
        self.assertEqual(response.data['filename'], 'test.pdf')
        
        # Check if file was saved to database
        self.assertEqual(UploadedFile.objects.count(), 1)
    
    def test_upload_non_pdf_file(self):
        """Test upload of non-PDF file"""
        test_file = SimpleUploadedFile(
            "test.txt",
            b"text content",
            content_type="text/plain"
        )
        
        url = reverse('analyzer:upload_file')
        response = self.client.post(url, {'file': test_file}, format='multipart')
        
        self.assertEqual(response.status_code, status.HTTP_400_BAD_REQUEST)
        self.assertIn('Only PDF files are allowed', response.data['message'])
    
    def test_upload_no_file(self):
        """Test upload request without file"""
        url = reverse('analyzer:upload_file')
        response = self.client.post(url, {}, format='multipart')
        
        self.assertEqual(response.status_code, status.HTTP_400_BAD_REQUEST)
        self.assertIn('No file provided', response.data['message'])
    
    def test_upload_status_endpoint(self):
        """Test upload status endpoint"""
        url = reverse('analyzer:upload_status')
        response = self.client.get(url)
        
        self.assertEqual(response.status_code, status.HTTP_200_OK)
        self.assertEqual(response.data['status'], 'active')


class AnalysisAPITest(APITestCase):
    """Test cases for analysis API"""
    
    def setUp(self):
        self.test_file = SimpleUploadedFile(
            "test.pdf",
            b"fake pdf content",
            content_type="application/pdf"
        )
        self.uploaded_file = UploadedFile.objects.create(
            file=self.test_file,
            original_filename="test.pdf",
            file_size=1000
        )
    
    @patch('analyzer.views.TableExtractor')
    @patch('analyzer.views.settings.HUGGINGFACE_API_KEY', 'test-key')
    def test_analyze_file_success(self, mock_extractor_class):
        """Test successful file analysis"""
        # Mock the table extractor
        mock_extractor = MagicMock()
        mock_extractor.extract_tables_from_pdf.return_value = {
            'total_pages': 2,
            'pages_processed': 2,
            'tables': [
                {
                    'page_number': 1,
                    'table_index': 0,
                    'bounding_box': [0, 0, 100, 100],
                    'table_data': {'rows': [['A', 'B'], ['1', '2']], 'columns': ['Col1', 'Col2']},
                    'confidence_score': 0.95
                }
            ]
        }
        mock_extractor_class.return_value = mock_extractor
        
        url = reverse('analyzer:analyze_file')
        response = self.client.post(url, {'file_id': self.uploaded_file.id}, format='json')
        
        self.assertEqual(response.status_code, status.HTTP_200_OK)
        self.assertEqual(response.data['status'], 'completed')
        self.assertEqual(response.data['tables_found'], 1)
        
        # Check if analysis result was created
        self.assertTrue(hasattr(self.uploaded_file, 'analysis_result'))
        
        # Check if extracted table was created
        self.assertEqual(ExtractedTable.objects.count(), 1)
    
    def test_analyze_file_invalid_id(self):
        """Test analysis with invalid file ID"""
        url = reverse('analyzer:analyze_file')
        response = self.client.post(url, {'file_id': 999}, format='json')
        
        self.assertEqual(response.status_code, status.HTTP_404_NOT_FOUND)
    
    @patch('analyzer.views.settings.HUGGINGFACE_API_KEY', '')
    def test_analyze_file_no_api_key(self):
        """Test analysis without Hugging Face API key"""
        url = reverse('analyzer:analyze_file')
        response = self.client.post(url, {'file_id': self.uploaded_file.id}, format='json')
        
        self.assertEqual(response.status_code, status.HTTP_500_INTERNAL_SERVER_ERROR)
        self.assertIn('API key not configured', response.data['message'])