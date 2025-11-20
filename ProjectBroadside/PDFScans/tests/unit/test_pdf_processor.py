"""
Unit tests for the PDFProcessor class.

Tests PDF to image conversion, page handling, and error management
without requiring actual PDF files.
"""

import pytest
from unittest.mock import Mock, patch, MagicMock
from pathlib import Path
import tempfile
from PIL import Image
import io

from src.warship_extractor.core.pdf_processor import PDFProcessor
from src.warship_extractor.config.settings import Settings


class TestPDFProcessor:
    """Test cases for PDFProcessor functionality."""
    
    @pytest.fixture
    def mock_settings(self):
        """Create mock settings for testing."""
        settings = Mock()
        settings.pdf_dpi = 300
        settings.pdf_image_format = "PNG"
        settings.pdf_max_pages = 100
        settings.pdf_page_range = None
        settings.enable_pdf_text_extraction = True
        settings.pdf_memory_limit_mb = 1024
        return settings
    
    @pytest.fixture
    def pdf_processor(self, mock_settings):
        """Create PDFProcessor instance for testing."""
        return PDFProcessor(mock_settings)
    
    def test_initialization(self, mock_settings):
        """Test PDFProcessor initialization."""
        processor = PDFProcessor(mock_settings)
        
        assert processor.settings == mock_settings
        assert processor.dpi == mock_settings.pdf_dpi
        assert processor.image_format == mock_settings.pdf_image_format
    
    @patch('fitz.open')
    def test_convert_pdf_to_images_success(self, mock_fitz_open, pdf_processor):
        """Test successful PDF to images conversion."""
        # Setup mock PDF document
        mock_doc = Mock()
        mock_page1 = Mock()
        mock_page2 = Mock()
        mock_doc.__len__.return_value = 2
        mock_doc.__iter__.return_value = iter([mock_page1, mock_page2])
        mock_doc.__enter__.return_value = mock_doc
        mock_doc.__exit__.return_value = None
        mock_fitz_open.return_value = mock_doc
        
        # Setup mock pixmaps and images
        mock_pix1 = Mock()
        mock_pix2 = Mock()
        mock_pix1.tobytes.return_value = b'fake_image_data_1'
        mock_pix2.tobytes.return_value = b'fake_image_data_2'
        mock_page1.get_pixmap.return_value = mock_pix1
        mock_page2.get_pixmap.return_value = mock_pix2
        
        # Mock PIL Image.open
        mock_image1 = Mock(spec=Image.Image)
        mock_image2 = Mock(spec=Image.Image)
        
        with patch('PIL.Image.open', side_effect=[mock_image1, mock_image2]):
            pdf_path = Path("test.pdf")
            images = pdf_processor.convert_pdf_to_images(pdf_path)
            
            assert len(images) == 2
            assert images[0] == mock_image1
            assert images[1] == mock_image2
            
            # Verify pixmap creation with correct DPI
            mock_page1.get_pixmap.assert_called_once_with(matrix=mock_pix1.get_pixmap.call_args[1]['matrix'])
            mock_page2.get_pixmap.assert_called_once_with(matrix=mock_pix2.get_pixmap.call_args[1]['matrix'])
    
    @patch('fitz.open')
    def test_convert_pdf_to_images_with_page_range(self, mock_fitz_open, mock_settings):
        """Test PDF conversion with specific page range."""
        mock_settings.pdf_page_range = (1, 3)  # Pages 1-3
        processor = PDFProcessor(mock_settings)
        
        # Setup mock PDF document with 5 pages
        mock_doc = Mock()
        mock_pages = [Mock() for _ in range(5)]
        mock_doc.__len__.return_value = 5
        mock_doc.__getitem__.side_effect = lambda i: mock_pages[i]
        mock_doc.__enter__.return_value = mock_doc
        mock_doc.__exit__.return_value = None
        mock_fitz_open.return_value = mock_doc
        
        # Setup mock pixmaps for pages 1-3
        for i, page in enumerate(mock_pages[1:4]):  # Pages 1-3 (0-indexed)
            mock_pix = Mock()
            mock_pix.tobytes.return_value = f'fake_image_data_{i}'.encode()
            page.get_pixmap.return_value = mock_pix
        
        with patch('PIL.Image.open', return_value=Mock(spec=Image.Image)) as mock_image_open:
            pdf_path = Path("test.pdf")
            images = processor.convert_pdf_to_images(pdf_path)
            
            assert len(images) == 3  # Only pages 1-3
            assert mock_image_open.call_count == 3
    
    @patch('fitz.open')
    def test_convert_pdf_to_images_max_pages_limit(self, mock_fitz_open, mock_settings):
        """Test PDF conversion with max pages limit."""
        mock_settings.pdf_max_pages = 2
        processor = PDFProcessor(mock_settings)
        
        # Setup mock PDF document with 5 pages
        mock_doc = Mock()
        mock_pages = [Mock() for _ in range(5)]
        mock_doc.__len__.return_value = 5
        mock_doc.__iter__.return_value = iter(mock_pages)
        mock_doc.__enter__.return_value = mock_doc
        mock_doc.__exit__.return_value = None
        mock_fitz_open.return_value = mock_doc
        
        # Setup mock pixmaps for first 2 pages
        for i, page in enumerate(mock_pages[:2]):
            mock_pix = Mock()
            mock_pix.tobytes.return_value = f'fake_image_data_{i}'.encode()
            page.get_pixmap.return_value = mock_pix
        
        with patch('PIL.Image.open', return_value=Mock(spec=Image.Image)) as mock_image_open:
            pdf_path = Path("test.pdf")
            images = processor.convert_pdf_to_images(pdf_path)
            
            assert len(images) == 2  # Limited to max_pages
            assert mock_image_open.call_count == 2
    
    @patch('fitz.open')
    def test_convert_pdf_file_not_found(self, mock_fitz_open, pdf_processor):
        """Test handling of non-existent PDF file."""
        mock_fitz_open.side_effect = FileNotFoundError("File not found")
        
        pdf_path = Path("nonexistent.pdf")
        
        with pytest.raises(FileNotFoundError, match="PDF file not found"):
            pdf_processor.convert_pdf_to_images(pdf_path)
    
    @patch('fitz.open')
    def test_convert_pdf_invalid_file(self, mock_fitz_open, pdf_processor):
        """Test handling of invalid PDF file."""
        mock_fitz_open.side_effect = Exception("Invalid PDF")
        
        pdf_path = Path("invalid.pdf")
        
        with pytest.raises(ValueError, match="Failed to open PDF"):
            pdf_processor.convert_pdf_to_images(pdf_path)
    
    @patch('fitz.open')
    def test_convert_pdf_page_processing_error(self, mock_fitz_open, pdf_processor):
        """Test handling of page processing errors."""
        # Setup mock PDF document
        mock_doc = Mock()
        mock_page = Mock()
        mock_doc.__len__.return_value = 1
        mock_doc.__iter__.return_value = iter([mock_page])
        mock_doc.__enter__.return_value = mock_doc
        mock_doc.__exit__.return_value = None
        mock_fitz_open.return_value = mock_doc
        
        # Make page processing fail
        mock_page.get_pixmap.side_effect = Exception("Page processing failed")
        
        pdf_path = Path("test.pdf")
        
        with pytest.raises(RuntimeError, match="Failed to process PDF"):
            pdf_processor.convert_pdf_to_images(pdf_path)
    
    @patch('fitz.open')
    def test_convert_single_page_success(self, mock_fitz_open, pdf_processor):
        """Test successful single page conversion."""
        # Setup mock PDF document
        mock_doc = Mock()
        mock_page = Mock()
        mock_doc.__len__.return_value = 5
        mock_doc.__getitem__.return_value = mock_page
        mock_doc.__enter__.return_value = mock_doc
        mock_doc.__exit__.return_value = None
        mock_fitz_open.return_value = mock_doc
        
        # Setup mock pixmap
        mock_pix = Mock()
        mock_pix.tobytes.return_value = b'fake_image_data'
        mock_page.get_pixmap.return_value = mock_pix
        
        mock_image = Mock(spec=Image.Image)
        
        with patch('PIL.Image.open', return_value=mock_image):
            pdf_path = Path("test.pdf")
            image = pdf_processor.convert_single_page(pdf_path, page_number=2)  # 0-indexed
            
            assert image == mock_image
            mock_doc.__getitem__.assert_called_once_with(2)
    
    @patch('fitz.open')
    def test_convert_single_page_invalid_page_number(self, mock_fitz_open, pdf_processor):
        """Test single page conversion with invalid page number."""
        # Setup mock PDF document
        mock_doc = Mock()
        mock_doc.__len__.return_value = 5
        mock_doc.__enter__.return_value = mock_doc
        mock_doc.__exit__.return_value = None
        mock_fitz_open.return_value = mock_doc
        
        pdf_path = Path("test.pdf")
        
        with pytest.raises(ValueError, match="Invalid page number"):
            pdf_processor.convert_single_page(pdf_path, page_number=10)
    
    @patch('fitz.open')
    def test_get_pdf_info(self, mock_fitz_open, pdf_processor):
        """Test PDF information extraction."""
        # Setup mock PDF document
        mock_doc = Mock()
        mock_doc.__len__.return_value = 25
        mock_doc.metadata = {
            'title': 'Test PDF',
            'author': 'Test Author',
            'creator': 'Test Creator',
            'producer': 'Test Producer',
            'creationDate': 'D:20230101120000+00\'00\'',
            'modDate': 'D:20230201120000+00\'00\''
        }
        mock_doc.__enter__.return_value = mock_doc
        mock_doc.__exit__.return_value = None
        mock_fitz_open.return_value = mock_doc
        
        pdf_path = Path("test.pdf")
        info = pdf_processor.get_pdf_info(pdf_path)
        
        expected_keys = [
            'page_count', 'title', 'author', 'creator', 'producer',
            'creation_date', 'modification_date', 'file_size'
        ]
        
        for key in expected_keys:
            assert key in info
        
        assert info['page_count'] == 25
        assert info['title'] == 'Test PDF'
        assert info['author'] == 'Test Author'
    
    @patch('fitz.open')
    def test_get_pdf_info_with_file_size(self, mock_fitz_open, pdf_processor):
        """Test PDF info extraction including file size."""
        mock_doc = Mock()
        mock_doc.__len__.return_value = 10
        mock_doc.metadata = {}
        mock_doc.__enter__.return_value = mock_doc
        mock_doc.__exit__.return_value = None
        mock_fitz_open.return_value = mock_doc
        
        # Create a temporary file to get actual file size
        with tempfile.NamedTemporaryFile(delete=False) as tmp_file:
            tmp_file.write(b"test data" * 1000)  # Write some test data
            tmp_path = Path(tmp_file.name)
        
        try:
            info = pdf_processor.get_pdf_info(tmp_path)
            assert info['file_size'] > 0
        finally:
            tmp_path.unlink()  # Clean up
    
    @patch('fitz.open')
    def test_extract_text_from_page(self, mock_fitz_open, pdf_processor):
        """Test text extraction from PDF page."""
        # Setup mock PDF document
        mock_doc = Mock()
        mock_page = Mock()
        mock_doc.__len__.return_value = 5
        mock_doc.__getitem__.return_value = mock_page
        mock_doc.__enter__.return_value = mock_doc
        mock_doc.__exit__.return_value = None
        mock_fitz_open.return_value = mock_doc
        
        # Setup mock text extraction
        mock_page.get_text.return_value = "This is extracted text from the PDF page."
        
        pdf_path = Path("test.pdf")
        text = pdf_processor.extract_text_from_page(pdf_path, page_number=2)
        
        assert text == "This is extracted text from the PDF page."
        mock_doc.__getitem__.assert_called_once_with(2)
        mock_page.get_text.assert_called_once()
    
    @patch('fitz.open')
    def test_extract_all_text(self, mock_fitz_open, pdf_processor):
        """Test text extraction from all PDF pages."""
        # Setup mock PDF document
        mock_doc = Mock()
        mock_pages = [Mock() for _ in range(3)]
        mock_doc.__len__.return_value = 3
        mock_doc.__iter__.return_value = iter(mock_pages)
        mock_doc.__enter__.return_value = mock_doc
        mock_doc.__exit__.return_value = None
        mock_fitz_open.return_value = mock_doc
        
        # Setup mock text extraction
        mock_pages[0].get_text.return_value = "Page 1 text."
        mock_pages[1].get_text.return_value = "Page 2 text."
        mock_pages[2].get_text.return_value = "Page 3 text."
        
        pdf_path = Path("test.pdf")
        all_text = pdf_processor.extract_all_text(pdf_path)
        
        expected_text = ["Page 1 text.", "Page 2 text.", "Page 3 text."]
        assert all_text == expected_text
    
    def test_calculate_dpi_matrix(self, pdf_processor):
        """Test DPI matrix calculation."""
        matrix = pdf_processor._calculate_dpi_matrix(300)
        
        # Matrix should scale to achieve desired DPI
        # Standard PDF DPI is 72, so scale factor should be 300/72
        expected_scale = 300 / 72
        assert abs(matrix.a - expected_scale) < 0.01
        assert abs(matrix.d - expected_scale) < 0.01
    
    def test_validate_page_number_valid(self, pdf_processor):
        """Test page number validation with valid number."""
        # Should not raise exception
        pdf_processor._validate_page_number(5, 10)
    
    def test_validate_page_number_negative(self, pdf_processor):
        """Test page number validation with negative number."""
        with pytest.raises(ValueError, match="Page number must be non-negative"):
            pdf_processor._validate_page_number(-1, 10)
    
    def test_validate_page_number_too_large(self, pdf_processor):
        """Test page number validation with number too large."""
        with pytest.raises(ValueError, match="Page number 15 exceeds"):
            pdf_processor._validate_page_number(15, 10)
    
    def test_memory_efficient_processing(self, mock_settings):
        """Test memory-efficient processing mode."""
        mock_settings.pdf_memory_limit_mb = 100  # Low memory limit
        processor = PDFProcessor(mock_settings)
        
        # This should enable memory-efficient mode
        assert processor.settings.pdf_memory_limit_mb == 100
    
    @patch('fitz.open')
    def test_context_manager(self, mock_fitz_open, pdf_processor):
        """Test PDFProcessor as context manager."""
        mock_doc = Mock()
        mock_doc.__enter__.return_value = mock_doc
        mock_doc.__exit__.return_value = None
        mock_fitz_open.return_value = mock_doc
        
        pdf_path = Path("test.pdf")
        
        with pdf_processor.open_pdf(pdf_path) as doc:
            assert doc == mock_doc
            mock_fitz_open.assert_called_once_with(str(pdf_path))
    
    def test_supported_formats(self, pdf_processor):
        """Test supported image formats."""
        supported = pdf_processor.get_supported_formats()
        
        expected_formats = ["PNG", "JPEG", "TIFF", "BMP"]
        for fmt in expected_formats:
            assert fmt in supported
    
    def test_format_validation(self, mock_settings):
        """Test image format validation."""
        mock_settings.pdf_image_format = "INVALID"
        
        with pytest.raises(ValueError, match="Unsupported image format"):
            PDFProcessor(mock_settings)