from rest_framework import serializers
from .models import UploadedFile


class FileUploadSerializer(serializers.ModelSerializer):
    """Serializer for file upload"""
    file = serializers.FileField()
    
    class Meta:
        model = UploadedFile
        fields = ['file', 'original_filename', 'uploaded_at', 'file_size']
        read_only_fields = ['original_filename', 'uploaded_at', 'file_size']
    
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