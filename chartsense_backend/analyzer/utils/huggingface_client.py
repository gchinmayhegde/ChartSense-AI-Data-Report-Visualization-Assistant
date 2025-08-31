import time
import requests
import logging
from typing import Dict, Any, Optional
from django.conf import settings

logger = logging.getLogger(__name__)


class HuggingFaceClient:
    """Client for interacting with Hugging Face API"""
    
    def __init__(self):
        self.api_key = settings.HUGGINGFACE_API_KEY
        self.base_url = "https://api-inference.huggingface.co/models"
        self.headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json"
        }
        self.request_delay = settings.HUGGINGFACE_REQUEST_DELAY
    
    def query_model(self, model_name: str, inputs: Dict[str, Any], max_retries: int = 3) -> Optional[Dict]:
        """
        Query a Hugging Face model with retry logic
        
        Args:
            model_name: Name of the model to query
            inputs: Input data for the model
            max_retries: Maximum number of retry attempts
            
        Returns:
            Response data from the API or None if failed
        """
        url = f"{self.base_url}/{model_name}"
        
        for attempt in range(max_retries):
            try:
                logger.info(f"Querying {model_name} (attempt {attempt + 1}/{max_retries})")
                
                response = requests.post(
                    url,
                    headers=self.headers,
                    json=inputs,
                    timeout=60
                )
                
                if response.status_code == 200:
                    logger.info(f"Successfully queried {model_name}")
                    # Add delay to respect rate limits
                    time.sleep(self.request_delay)
                    return response.json()
                
                elif response.status_code == 503:
                    # Model is loading, wait and retry
                    wait_time = 20 * (attempt + 1)  # Exponential backoff
                    logger.warning(f"Model {model_name} is loading, waiting {wait_time}s before retry")
                    time.sleep(wait_time)
                    continue
                
                elif response.status_code == 429:
                    # Rate limit exceeded
                    wait_time = 30 * (attempt + 1)
                    logger.warning(f"Rate limit exceeded, waiting {wait_time}s before retry")
                    time.sleep(wait_time)
                    continue
                
                else:
                    logger.error(f"API request failed with status {response.status_code}: {response.text}")
                    if attempt == max_retries - 1:
                        raise Exception(f"API request failed: {response.status_code} - {response.text}")
                    time.sleep(5 * (attempt + 1))
            
            except requests.exceptions.Timeout:
                logger.warning(f"Request to {model_name} timed out (attempt {attempt + 1}/{max_retries})")
                if attempt == max_retries - 1:
                    raise Exception(f"Request timed out after {max_retries} attempts")
                time.sleep(5 * (attempt + 1))
            
            except requests.exceptions.RequestException as e:
                logger.error(f"Request error for {model_name}: {str(e)}")
                if attempt == max_retries - 1:
                    raise Exception(f"Request failed: {str(e)}")
                time.sleep(5 * (attempt + 1))
        
        return None
    
    def detect_tables(self, image_data: bytes) -> Optional[Dict]:
        """
        Detect tables in an image using the table detection model
        
        Args:
            image_data: Image data as bytes
            
        Returns:
            Detection results or None if failed
        """
        model_name = settings.HUGGINGFACE_TABLE_DETECTION_MODEL
        
        # Convert image data to base64 string
        import base64
        image_b64 = base64.b64encode(image_data).decode('utf-8')
        
        inputs = {
            "inputs": image_b64
        }
        
        return self.query_model(model_name, inputs)
    
    def recognize_table_structure(self, image_data: bytes) -> Optional[Dict]:
        """
        Recognize table structure in an image
        
        Args:
            image_data: Image data as bytes
            
        Returns:
            Structure recognition results or None if failed
        """
        model_name = settings.HUGGINGFACE_TABLE_STRUCTURE_MODEL
        
        # Convert image data to base64 string
        import base64
        image_b64 = base64.b64encode(image_data).decode('utf-8')
        
        inputs = {
            "inputs": image_b64
        }
        
        return self.query_model(model_name, inputs)