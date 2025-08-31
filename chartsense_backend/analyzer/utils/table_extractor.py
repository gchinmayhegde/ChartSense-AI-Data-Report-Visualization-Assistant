import io
import json
import logging
from typing import List, Dict, Any
from PIL import Image
import numpy as np

from .pdf_processor import PDFProcessor
from .huggingface_client import HuggingFaceClient

logger = logging.getLogger(__name__)


class TableExtractor:
    """Main class for extracting tables from PDF files using Hugging Face models"""
    
    def __init__(self):
        self.pdf_processor = PDFProcessor()
        self.hf_client = HuggingFaceClient()
        self.confidence_threshold = 0.7  # Minimum confidence score for table detection
    
    def extract_tables_from_pdf(self, pdf_path: str) -> Dict[str, Any]:
        """
        Extract tables from a PDF file
        
        Args:
            pdf_path: Path to the PDF file
            
        Returns:
            Dictionary containing extraction results
        """
        logger.info(f"Starting table extraction for PDF: {pdf_path}")
        
        # Convert PDF to images
        page_images = self.pdf_processor.pdf_to_images(pdf_path)
        
        if not page_images:
            raise Exception("No pages could be processed from the PDF")
        
        total_pages = len(page_images)
        pages_processed = 0
        all_tables = []
        
        logger.info(f"Processing {total_pages} pages for table extraction")
        
        for page_num, image_bytes in page_images:
            try:
                logger.info(f"Processing page {page_num}")
                
                # Detect tables on this page
                page_tables = self._extract_tables_from_page(page_num, image_bytes)
                all_tables.extend(page_tables)
                
                pages_processed += 1
                logger.info(f"Found {len(page_tables)} tables on page {page_num}")
                
            except Exception as e:
                logger.error(f"Error processing page {page_num}: {str(e)}")
                continue
        
        result = {
            'total_pages': total_pages,
            'pages_processed': pages_processed,
            'tables': all_tables,
            'pdf_path': pdf_path
        }
        
        logger.info(f"Table extraction completed. Found {len(all_tables)} tables total.")
        return result
    
    def _extract_tables_from_page(self, page_num: int, image_bytes: bytes) -> List[Dict[str, Any]]:
        """
        Extract tables from a single page image
        
        Args:
            page_num: Page number
            image_bytes: Page image as bytes
            
        Returns:
            List of extracted tables from this page
        """
        page_tables = []
        
        try:
            # Step 1: Detect tables in the image
            detection_results = self.hf_client.detect_tables(image_bytes)
            
            if not detection_results:
                logger.warning(f"No detection results for page {page_num}")
                return page_tables
            
            # Process detection results
            tables_detected = self._process_detection_results(detection_results)
            
            if not tables_detected:
                logger.info(f"No tables detected on page {page_num}")
                return page_tables
            
            logger.info(f"Detected {len(tables_detected)} potential tables on page {page_num}")
            
            # Step 2: For each detected table, extract structure and content
            for table_index, table_detection in enumerate(tables_detected):
                try:
                    # Crop the table region from the original image
                    table_image_bytes = self._crop_table_region(
                        image_bytes, 
                        table_detection['bounding_box']
                    )
                    
                    # Step 3: Recognize table structure
                    structure_results = self.hf_client.recognize_table_structure(table_image_bytes)
                    
                    if structure_results:
                        # Process structure results into structured data
                        table_data = self._process_structure_results(structure_results)
                        
                        table_info = {
                            'page_number': page_num,
                            'table_index': table_index,
                            'bounding_box': table_detection['bounding_box'],
                            'confidence_score': table_detection.get('confidence_score'),
                            'table_data': table_data
                        }
                        
                        page_tables.append(table_info)
                        logger.info(f"Successfully extracted table {table_index} from page {page_num}")
                    else:
                        logger.warning(f"Failed to recognize structure for table {table_index} on page {page_num}")
                
                except Exception as e:
                    logger.error(f"Error processing table {table_index} on page {page_num}: {str(e)}")
                    continue
        
        except Exception as e:
            logger.error(f"Error extracting tables from page {page_num}: {str(e)}")
        
        return page_tables
    
    def _process_detection_results(self, detection_results: Dict[str, Any]) -> List[Dict[str, Any]]:
        """
        Process table detection results from Hugging Face API
        
        Args:
            detection_results: Raw detection results from API
            
        Returns:
            List of processed table detections
        """
        tables = []
        
        try:
            # Handle different possible response formats from Hugging Face
            if isinstance(detection_results, list):
                detections = detection_results
            elif isinstance(detection_results, dict):
                detections = detection_results.get('predictions', detection_results.get('detections', [detection_results]))
            else:
                logger.warning(f"Unexpected detection results format: {type(detection_results)}")
                return tables
            
            for detection in detections:
                # Extract bounding box and confidence score
                bbox = detection.get('box', detection.get('bounding_box', {}))
                score = detection.get('score', detection.get('confidence', 0.0))
                label = detection.get('label', detection.get('class', ''))
                
                # Filter by confidence threshold and ensure it's a table
                if score >= self.confidence_threshold and 'table' in label.lower():
                    # Normalize bounding box format
                    if isinstance(bbox, dict):
                        # Convert from dict format {x1, y1, x2, y2} or {xmin, ymin, xmax, ymax}
                        x1 = bbox.get('x1', bbox.get('xmin', 0))
                        y1 = bbox.get('y1', bbox.get('ymin', 0))
                        x2 = bbox.get('x2', bbox.get('xmax', 100))
                        y2 = bbox.get('y2', bbox.get('ymax', 100))
                        normalized_bbox = [x1, y1, x2, y2]
                    elif isinstance(bbox, list) and len(bbox) >= 4:
                        normalized_bbox = bbox[:4]
                    else:
                        logger.warning(f"Invalid bounding box format: {bbox}")
                        continue
                    
                    tables.append({
                        'bounding_box': normalized_bbox,
                        'confidence_score': score,
                        'label': label
                    })
        
        except Exception as e:
            logger.error(f"Error processing detection results: {str(e)}")
        
        return tables
    
    def _crop_table_region(self, image_bytes: bytes, bounding_box: List[float]) -> bytes:
        """
        Crop table region from image based on bounding box
        
        Args:
            image_bytes: Original image as bytes
            bounding_box: Bounding box coordinates [x1, y1, x2, y2]
            
        Returns:
            Cropped image as bytes
        """
        try:
            # Open image with PIL
            image = Image.open(io.BytesIO(image_bytes))
            
            # Ensure bounding box coordinates are within image bounds
            x1, y1, x2, y2 = bounding_box
            img_width, img_height = image.size
            
            # Normalize and clamp coordinates
            x1 = max(0, min(int(x1), img_width))
            y1 = max(0, min(int(y1), img_height))
            x2 = max(x1, min(int(x2), img_width))
            y2 = max(y1, min(int(y2), img_height))
            
            # Crop the image
            cropped_image = image.crop((x1, y1, x2, y2))
            
            # Convert back to bytes
            img_byte_array = io.BytesIO()
            cropped_image.save(img_byte_array, format='PNG')
            return img_byte_array.getvalue()
        
        except Exception as e:
            logger.error(f"Error cropping table region: {str(e)}")
            # Return original image if cropping fails
            return image_bytes
    
    def _process_structure_results(self, structure_results: Dict[str, Any]) -> Dict[str, Any]:
        """
        Process table structure recognition results
        
        Args:
            structure_results: Raw structure results from API
            
        Returns:
            Structured table data
        """
        try:
            # Handle different possible response formats
            if isinstance(structure_results, dict):
                # Look for structured table data
                if 'table' in structure_results:
                    return structure_results['table']
                elif 'data' in structure_results:
                    return structure_results['data']
                elif 'rows' in structure_results or 'columns' in structure_results:
                    return structure_results
            
            # If we have a list of results, take the first one
            elif isinstance(structure_results, list) and structure_results:
                return self._process_structure_results(structure_results[0])
            
            # Fallback: create a basic structure with the raw results
            return {
                'raw_results': structure_results,
                'rows': [],
                'columns': [],
                'cells': [],
                'note': 'Table structure could not be fully parsed, raw results included'
            }
        
        except Exception as e:
            logger.error(f"Error processing structure results: {str(e)}")
            return {
                'error': str(e),
                'raw_results': structure_results,
                'rows': [],
                'columns': [],
                'cells': []
            }