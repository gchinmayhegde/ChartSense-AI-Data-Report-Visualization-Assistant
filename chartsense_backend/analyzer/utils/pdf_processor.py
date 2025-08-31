import io
import logging
from typing import List, Tuple
from PIL import Image
import fitz  # PyMuPDF

logger = logging.getLogger(__name__)


class PDFProcessor:
    """Handles PDF processing and page-to-image conversion"""
    
    def __init__(self):
        self.dpi = 150  # DPI for image conversion
        self.image_format = 'PNG'
    
    def pdf_to_images(self, pdf_path: str) -> List[Tuple[int, bytes]]:
        """
        Convert PDF pages to images
        
        Args:
            pdf_path: Path to the PDF file
            
        Returns:
            List of tuples (page_number, image_bytes)
        """
        images = []
        
        try:
            # Open PDF document
            doc = fitz.open(pdf_path)
            total_pages = doc.page_count
            
            logger.info(f"Processing PDF with {total_pages} pages")
            
            for page_num in range(total_pages):
                try:
                    # Get the page
                    page = doc[page_num]
                    
                    # Create a transformation matrix for the desired DPI
                    mat = fitz.Matrix(self.dpi/72, self.dpi/72)
                    
                    # Render page as image
                    pix = page.get_pixmap(matrix=mat)
                    
                    # Convert to PIL Image
                    img_data = pix.tobytes(self.image_format.lower())
                    
                    # Convert to bytes for API
                    image_bytes = self._pil_image_to_bytes(Image.open(io.BytesIO(img_data)))
                    
                    images.append((page_num + 1, image_bytes))
                    logger.info(f"Converted page {page_num + 1} to image")
                
                except Exception as e:
                    logger.error(f"Error processing page {page_num + 1}: {str(e)}")
                    continue
            
            doc.close()
            logger.info(f"Successfully converted {len(images)} pages to images")
            return images
        
        except Exception as e:
            logger.error(f"Error processing PDF {pdf_path}: {str(e)}")
            raise Exception(f"Failed to process PDF: {str(e)}")
    
    def _pil_image_to_bytes(self, pil_image: Image.Image) -> bytes:
        """
        Convert PIL Image to bytes
        
        Args:
            pil_image: PIL Image object
            
        Returns:
            Image as bytes
        """
        img_byte_array = io.BytesIO()
        
        # Ensure image is in RGB mode for consistent processing
        if pil_image.mode != 'RGB':
            pil_image = pil_image.convert('RGB')
        
        pil_image.save(img_byte_array, format=self.image_format, quality=85, optimize=True)
        return img_byte_array.getvalue()
    
    def get_pdf_info(self, pdf_path: str) -> dict:
        """
        Get basic information about the PDF
        
        Args:
            pdf_path: Path to the PDF file
            
        Returns:
            Dictionary with PDF information
        """
        try:
            doc = fitz.open(pdf_path)
            info = {
                'page_count': doc.page_count,
                'title': doc.metadata.get('title', ''),
                'author': doc.metadata.get('author', ''),
                'subject': doc.metadata.get('subject', ''),
                'creator': doc.metadata.get('creator', ''),
                'producer': doc.metadata.get('producer', ''),
                'creation_date': doc.metadata.get('creationDate', ''),
                'modification_date': doc.metadata.get('modDate', ''),
            }
            doc.close()
            return info
        
        except Exception as e:
            logger.error(f"Error getting PDF info for {pdf_path}: {str(e)}")
            return {'page_count': 0, 'error': str(e)}