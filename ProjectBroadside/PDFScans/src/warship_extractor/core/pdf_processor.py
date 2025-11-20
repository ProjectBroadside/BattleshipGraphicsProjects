"""
PDF processing with high-resolution conversion and memory optimization.

Handles PDF page extraction, conversion to images, and memory-efficient
processing of large documents using PyMuPDF.
"""

import io
import logging
from pathlib import Path
from typing import Generator, List, Optional, Tuple, Union

import fitz  # PyMuPDF
from PIL import Image

from ..config.settings import settings

logger = logging.getLogger(__name__)


class PDFProcessor:
    """
    Handles PDF processing with high-resolution conversion and memory optimization.
    """
    
    def __init__(self, dpi: Optional[int] = None):
        """
        Initialize the PDF processor.
        
        Args:
            dpi: DPI for image conversion (defaults to settings)
        """
        self.dpi = dpi or settings.default_dpi
        self.zoom_factor = self.dpi / 72.0  # PDF default is 72 DPI
        
        logger.info(f"PDFProcessor initialized with DPI: {self.dpi}")
    
    def open_document(self, pdf_path: Union[str, Path]) -> fitz.Document:
        """
        Open a PDF document.
        
        Args:
            pdf_path: Path to the PDF file
            
        Returns:
            Opened PDF document
            
        Raises:
            FileNotFoundError: If PDF file doesn't exist
            RuntimeError: If PDF cannot be opened
        """
        pdf_path = Path(pdf_path)
        
        if not pdf_path.exists():
            raise FileNotFoundError(f"PDF file not found: {pdf_path}")
        
        try:
            doc = fitz.open(str(pdf_path))
            logger.info(f"Opened PDF: {pdf_path.name} ({len(doc)} pages)")
            return doc
        except Exception as e:
            logger.error(f"Failed to open PDF {pdf_path}: {str(e)}")
            raise RuntimeError(f"Cannot open PDF: {str(e)}") from e
    
    def get_page_count(self, pdf_path: Union[str, Path]) -> int:
        """
        Get the number of pages in a PDF without keeping it open.
        
        Args:
            pdf_path: Path to the PDF file
            
        Returns:
            Number of pages
        """
        with self.open_document(pdf_path) as doc:
            return len(doc)
    
    def convert_page_to_image(
        self, 
        page: fitz.Page,
        dpi: Optional[int] = None,
        alpha: bool = False
    ) -> Tuple[Image.Image, dict]:
        """
        Convert a PDF page to a PIL Image.
        
        Args:
            page: PyMuPDF page object
            dpi: DPI for conversion (uses instance default if None)
            alpha: Include alpha channel
            
        Returns:
            Tuple of (PIL Image, metadata dict)
        """
        target_dpi = dpi or self.dpi
        zoom = target_dpi / 72.0
        matrix = fitz.Matrix(zoom, zoom)
        
        try:
            # Get page dimensions
            rect = page.rect
            metadata = {
                "page_number": page.number + 1,
                "width": rect.width,
                "height": rect.height,
                "dpi": target_dpi,
                "zoom_factor": zoom
            }
            
            # Convert to pixmap
            pixmap = page.get_pixmap(matrix=matrix, alpha=alpha)
            
            # Convert to PIL Image
            img_data = pixmap.pil_tobytes(format="PNG")
            image = Image.open(io.BytesIO(img_data))
            
            # Clean up pixmap to free memory
            pixmap = None
            
            logger.debug(f"Converted page {metadata['page_number']} to {image.size} image")
            
            return image, metadata
            
        except Exception as e:
            logger.error(f"Failed to convert page {page.number + 1}: {str(e)}")
            raise RuntimeError(f"Page conversion failed: {str(e)}") from e
    
    def process_pages(
        self,
        pdf_path: Union[str, Path],
        page_numbers: Optional[List[int]] = None,
        dpi: Optional[int] = None
    ) -> Generator[Tuple[Image.Image, dict], None, None]:
        """
        Process PDF pages and yield images with metadata.
        
        Args:
            pdf_path: Path to the PDF file
            page_numbers: Specific page numbers to process (1-indexed), None for all
            dpi: DPI for conversion
            
        Yields:
            Tuples of (PIL Image, metadata dict)
        """
        doc = self.open_document(pdf_path)
        
        try:
            # Determine which pages to process
            if page_numbers is None:
                pages_to_process = range(len(doc))
            else:
                # Convert 1-indexed to 0-indexed and validate
                pages_to_process = []
                for page_num in page_numbers:
                    if 1 <= page_num <= len(doc):
                        pages_to_process.append(page_num - 1)
                    else:
                        logger.warning(f"Invalid page number {page_num}, skipping")
            
            logger.info(f"Processing {len(pages_to_process)} pages from {Path(pdf_path).name}")
            
            for page_idx in pages_to_process:
                try:
                    page = doc[page_idx]
                    image, metadata = self.convert_page_to_image(page, dpi)
                    
                    # Add PDF metadata
                    metadata.update({
                        "pdf_path": str(pdf_path),
                        "pdf_name": Path(pdf_path).stem,
                        "total_pages": len(doc)
                    })
                    
                    yield image, metadata
                    
                except Exception as e:
                    logger.error(f"Error processing page {page_idx + 1}: {str(e)}")
                    continue
                    
        finally:
            doc.close()
    
    def extract_page_images(
        self,
        pdf_path: Union[str, Path],
        output_dir: Optional[Union[str, Path]] = None,
        page_numbers: Optional[List[int]] = None,
        dpi: Optional[int] = None,
        format: str = "PNG"
    ) -> List[Tuple[Path, dict]]:
        """
        Extract pages as image files.
        
        Args:
            pdf_path: Path to the PDF file
            output_dir: Output directory (defaults to settings output directory)
            page_numbers: Specific pages to extract (1-indexed)
            dpi: DPI for conversion
            format: Image format (PNG, JPEG, etc.)
            
        Returns:
            List of (output_path, metadata) tuples
        """
        if output_dir is None:
            output_dir = settings.get_output_path("extracted_pages")
        else:
            output_dir = Path(output_dir)
        
        output_dir.mkdir(parents=True, exist_ok=True)
        
        pdf_name = Path(pdf_path).stem
        results = []
        
        for image, metadata in self.process_pages(pdf_path, page_numbers, dpi):
            page_num = metadata["page_number"]
            
            # Generate output filename
            output_filename = f"{pdf_name}_page_{page_num:03d}.{format.lower()}"
            output_path = output_dir / output_filename
            
            try:
                # Save image with appropriate DPI
                save_kwargs = {}
                if format.upper() == "PNG":
                    save_kwargs["dpi"] = (metadata["dpi"], metadata["dpi"])
                elif format.upper() == "JPEG":
                    save_kwargs["dpi"] = (metadata["dpi"], metadata["dpi"])
                    save_kwargs["quality"] = 95
                
                image.save(output_path, format=format, **save_kwargs)
                
                metadata["output_path"] = str(output_path)
                metadata["file_size"] = output_path.stat().st_size
                
                results.append((output_path, metadata))
                
                logger.debug(f"Saved page {page_num} to {output_path}")
                
            except Exception as e:
                logger.error(f"Failed to save page {page_num}: {str(e)}")
                continue
        
        logger.info(f"Extracted {len(results)} pages from {pdf_name}")
        return results
    
    def get_page_info(self, pdf_path: Union[str, Path]) -> List[dict]:
        """
        Get information about all pages in a PDF.
        
        Args:
            pdf_path: Path to the PDF file
            
        Returns:
            List of page information dictionaries
        """
        doc = self.open_document(pdf_path)
        page_info = []
        
        try:
            for page_num, page in enumerate(doc):
                rect = page.rect
                info = {
                    "page_number": page_num + 1,
                    "width": rect.width,
                    "height": rect.height,
                    "rotation": page.rotation,
                    "has_images": len(page.get_images()) > 0,
                    "has_text": bool(page.get_text().strip()),
                    "image_count": len(page.get_images())
                }
                page_info.append(info)
        
        finally:
            doc.close()
        
        return page_info
    
    def estimate_memory_usage(
        self, 
        pdf_path: Union[str, Path],
        dpi: Optional[int] = None
    ) -> dict:
        """
        Estimate memory usage for processing a PDF.
        
        Args:
            pdf_path: Path to the PDF file
            dpi: DPI for conversion estimation
            
        Returns:
            Memory usage estimation dictionary
        """
        target_dpi = dpi or self.dpi
        page_info = self.get_page_info(pdf_path)
        
        total_pixels = 0
        max_page_pixels = 0
        
        for info in page_info:
            # Estimate pixel count at target DPI
            zoom = target_dpi / 72.0
            width_pixels = int(info["width"] * zoom)
            height_pixels = int(info["height"] * zoom)
            page_pixels = width_pixels * height_pixels
            
            total_pixels += page_pixels
            max_page_pixels = max(max_page_pixels, page_pixels)
        
        # Estimate memory usage (4 bytes per pixel for RGBA)
        bytes_per_pixel = 4
        max_page_memory = max_page_pixels * bytes_per_pixel
        total_memory = total_pixels * bytes_per_pixel
        
        return {
            "total_pages": len(page_info),
            "estimated_total_memory_mb": total_memory / (1024 * 1024),
            "estimated_max_page_memory_mb": max_page_memory / (1024 * 1024),
            "dpi": target_dpi,
            "total_pixels": total_pixels,
            "max_page_pixels": max_page_pixels
        }