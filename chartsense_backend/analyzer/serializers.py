from rest_framework import serializers
from .models import UploadedFile, AnalysisResult, ExtractedTable


class FileUploadSerializer(serializers.ModelSerializer):
    """Serializer for file upload"""
    file = serializers.FileField()
    
    class Meta:
        model = UploadedFile
        fields = ['file', 'original_filename', 'uploaded_at', 'file_size', 'analysis_status']
        read_only_fields = ['original_filename', 'uploaded_at', 'file_size', 'analysis_status']
    
    def validate_file(self, value):
        """Validate uploaded file"""
        # Check file extension
        if not value.name.lower().endswith('.pdf'):
            raise serializers.ValidationError("Only PDF files are allowed.")
        
        # Check file size (10MB limit)
        if value.size > 10 * 1024 * 1024:
            raise serializers.ValidationError("File size cannot exceed 10MB.")
        
        return value
    
    def create(self, validated_data):
        """Create new uploaded file instance"""
        file = validated_data['file']
        validated_data['original_filename'] = file.name
        validated_data['file_size'] = file.size
        return super().create(validated_data)


class FileUploadResponseSerializer(serializers.Serializer):
    """Serializer for upload response"""
    status = serializers.CharField()
    filename = serializers.CharField()
    file_id = serializers.IntegerField(required=False)
    message = serializers.CharField(required=False)


class ExtractedTableSerializer(serializers.ModelSerializer):
    """Serializer for extracted tables"""
    table_summary = serializers.ReadOnlyField()
    
    class Meta:
        model = ExtractedTable
        fields = [
            'id', 'page_number', 'table_index', 'bounding_box', 
            'table_data', 'confidence_score', 'extracted_at', 'table_summary'
        ]


class AnalysisResultSerializer(serializers.ModelSerializer):
    """Serializer for analysis results"""
    extracted_tables = ExtractedTableSerializer(many=True, read_only=True)
    filename = serializers.CharField(source='uploaded_file.original_filename', read_only=True)
    
    class Meta:
        model = AnalysisResult
        fields = [
            'id', 'filename', 'status', 'total_pages', 'pages_processed',
            'tables_found', 'error_message', 'processing_time', 
            'created_at', 'updated_at', 'extracted_tables'
        ]


class AnalyzeRequestSerializer(serializers.Serializer):
    """Serializer for analysis request"""
    file_id = serializers.IntegerField(help_text="ID of the uploaded file to analyze")
    
    def validate_file_id(self, value):
        """Validate that the file exists and hasn't been analyzed yet"""
        try:
            uploaded_file = UploadedFile.objects.get(id=value)
            # Check if analysis already exists
            if hasattr(uploaded_file, 'analysis_result'):
                analysis = uploaded_file.analysis_result
                if analysis.status in ['processing', 'completed']:
                    raise serializers.ValidationError(
                        f"File is already being analyzed or has been analyzed. Current status: {analysis.status}"
                    )
        except UploadedFile.DoesNotExist:
            raise serializers.ValidationError("File with this ID does not exist.")
        
        return value


class AnalyzeResponseSerializer(serializers.Serializer):
    """Serializer for analysis response"""
    status = serializers.CharField()
    message = serializers.CharField()
    analysis_id = serializers.IntegerField(required=False)
    file_id = serializers.IntegerField()
    estimated_time = serializers.CharField(required=False)