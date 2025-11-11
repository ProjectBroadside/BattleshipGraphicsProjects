"""
Test configuration and fixtures for the warship extractor test suite.

This module provides common fixtures, test data, and configuration
used across all test modules.
"""

import json
import tempfile
from pathlib import Path
from typing import Dict, Any, List
from unittest.mock import Mock, MagicMock

import pytest
import numpy as np
from PIL import Image

# Test configuration
TEST_DATA_DIR = Path(__file__).parent / "data"
TEST_OUTPUT_DIR = Path(__file__).parent / "output"


@pytest.fixture(scope="session")
def test_data_dir():
    """Get the test data directory."""
    TEST_DATA_DIR.mkdir(exist_ok=True)
    return TEST_DATA_DIR


@pytest.fixture(scope="session")
def test_output_dir():
    """Get the test output directory."""
    TEST_OUTPUT_DIR.mkdir(exist_ok=True)
    return TEST_OUTPUT_DIR


@pytest.fixture
def temp_dir():
    """Create a temporary directory for tests."""
    with tempfile.TemporaryDirectory() as tmpdir:
        yield Path(tmpdir)


@pytest.fixture
def sample_image():
    """Create a sample test image."""
    # Create a simple RGB image
    image = Image.new('RGB', (800, 600), color='white')
    return image


@pytest.fixture
def sample_image_array():
    """Create a sample test image as numpy array."""
    return np.random.randint(0, 255, (600, 800, 3), dtype=np.uint8)


@pytest.fixture
def sample_detection():
    """Sample detection result."""
    return {
        'bbox': [100, 150, 300, 250],
        'label': 'warship',
        'confidence': 0.85,
        'page': 1
    }


@pytest.fixture
def sample_detections():
    """Multiple sample detection results."""
    return [
        {
            'bbox': [100, 150, 300, 250],
            'label': 'warship',
            'confidence': 0.85,
            'page': 1
        },
        {
            'bbox': [400, 200, 600, 350],
            'label': 'ship',
            'confidence': 0.72,
            'page': 1
        },
        {
            'bbox': [150, 400, 350, 550],
            'label': 'vessel',
            'confidence': 0.91,
            'page': 2
        }
    ]


@pytest.fixture
def overlapping_detections():
    """Sample overlapping detections for NMS testing."""
    return [
        {
            'bbox': [100, 100, 200, 200],
            'label': 'warship',
            'confidence': 0.9
        },
        {
            'bbox': [110, 110, 210, 210],
            'label': 'warship', 
            'confidence': 0.8
        },
        {
            'bbox': [300, 300, 400, 400],
            'label': 'ship',
            'confidence': 0.7
        }
    ]


@pytest.fixture
def sample_extraction_results():
    """Sample complete extraction results."""
    return {
        'detections': [
            {
                'bbox': [100, 150, 300, 250],
                'label': 'warship',
                'confidence': 0.85,
                'page': 1,
                'cropped_image': None
            }
        ],
        'statistics': {
            'total_detections': 1,
            'pages_processed': 1,
            'processing_time': 45.2,
            'avg_confidence': 0.85,
            'detections_per_page': {'1': 1}
        },
        'metadata': {
            'pdf_path': '/test/sample.pdf',
            'extraction_date': '2024-01-01T10:00:00',
            'model_name': 'microsoft/Florence-2-large',
            'settings': {
                'confidence_threshold': 0.3,
                'pdf_dpi': 300
            }
        }
    }


@pytest.fixture
def mock_florence_model():
    """Mock Florence-2 model for testing."""
    mock_model = Mock()
    mock_processor = Mock()
    
    # Mock model output
    mock_model.generate.return_value = [
        'warship<loc_100><loc_150><loc_300><loc_250>'
    ]
    
    # Mock processor
    mock_processor.post_process_generation.return_value = {
        '<OD>': {
            'bboxes': [[100, 150, 300, 250]],
            'labels': ['warship']
        }
    }
    
    return {
        'model': mock_model,
        'processor': mock_processor
    }


@pytest.fixture
def mock_pdf_pages():
    """Mock PDF pages for testing."""
    # Create sample images representing PDF pages
    pages = []
    for i in range(3):
        image = Image.new('RGB', (1200, 800), color='white')
        pages.append(image)
    return pages


@pytest.fixture
def test_settings():
    """Test configuration settings."""
    return {
        'model_name': 'microsoft/Florence-2-base',
        'confidence_threshold': 0.3,
        'pdf_dpi': 200,  # Lower for faster tests
        'batch_size': 1,
        'enable_gpu': False,  # Disable GPU for consistent testing
        'enable_file_logging': False,
        'log_level': 'WARNING'  # Reduce log noise in tests
    }


@pytest.fixture
def mock_settings(test_settings):
    """Mock settings object with test configuration."""
    from unittest.mock import Mock
    
    settings = Mock()
    for key, value in test_settings.items():
        setattr(settings, key, value)
    
    # Add path methods
    settings.get_cache_path.return_value = Path("/tmp/test_cache")
    settings.get_output_path.return_value = Path("/tmp/test_output")
    settings.get_log_path.return_value = Path("/tmp/test.log")
    
    return settings


class MockPDFDocument:
    """Mock PDF document for testing."""
    
    def __init__(self, page_count=3):
        self.page_count = page_count
        self._pages = [MockPDFPage(i) for i in range(page_count)]
    
    def __len__(self):
        return self.page_count
    
    def __getitem__(self, index):
        return self._pages[index]
    
    def load_page(self, page_num):
        return self._pages[page_num]
    
    def close(self):
        pass


class MockPDFPage:
    """Mock PDF page for testing."""
    
    def __init__(self, page_num):
        self.page_num = page_num
        self.rect = Mock()
        self.rect.width = 1200
        self.rect.height = 800
    
    def get_pixmap(self, matrix=None, dpi=None):
        # Return mock pixmap
        mock_pixmap = Mock()
        mock_pixmap.pil_tobytes.return_value = b'fake_image_data'
        mock_pixmap.width = 1200
        mock_pixmap.height = 800
        return mock_pixmap


@pytest.fixture
def mock_pdf_document():
    """Mock PDF document fixture."""
    return MockPDFDocument(page_count=3)


@pytest.fixture
def sample_pdf_metadata():
    """Sample PDF metadata."""
    return {
        'title': 'Jane\'s Fighting Ships 1900',
        'author': 'Jane\'s Publishing',
        'subject': 'Naval vessels and warships',
        'creator': 'Historical Archive',
        'producer': 'PDF Generator',
        'creation_date': '2024-01-01',
        'modification_date': '2024-01-01',
        'page_count': 150
    }


def create_test_pdf(output_path: Path, page_count: int = 3) -> Path:
    """
    Create a simple test PDF file.
    
    Args:
        output_path: Path where to save the PDF
        page_count: Number of pages to create
        
    Returns:
        Path to the created PDF
    """
    try:
        from reportlab.pdfgen import canvas
        from reportlab.lib.pagesizes import letter
        
        c = canvas.Canvas(str(output_path), pagesize=letter)
        
        for i in range(page_count):
            c.drawString(100, 750, f"Test PDF - Page {i+1}")
            c.drawString(100, 700, "This is a test PDF for warship extraction testing")
            
            # Draw some simple shapes to simulate content
            c.rect(200, 400, 300, 150, stroke=1, fill=0)
            c.drawString(220, 470, "Simulated warship illustration")
            
            c.showPage()
        
        c.save()
        return output_path
        
    except ImportError:
        # If reportlab not available, create empty file
        output_path.touch()
        return output_path


@pytest.fixture
def test_pdf(temp_dir):
    """Create a test PDF file."""
    pdf_path = temp_dir / "test.pdf"
    return create_test_pdf(pdf_path, page_count=3)


@pytest.fixture(autouse=True)
def setup_test_logging():
    """Set up logging for tests."""
    import logging
    
    # Reduce logging noise during tests
    logging.getLogger('warship_extractor').setLevel(logging.WARNING)
    logging.getLogger('transformers').setLevel(logging.ERROR)
    logging.getLogger('torch').setLevel(logging.ERROR)


@pytest.fixture
def performance_timer():
    """Timer for performance testing."""
    import time
    
    class Timer:
        def __init__(self):
            self.start_time = None
            self.end_time = None
        
        def start(self):
            self.start_time = time.time()
        
        def stop(self):
            self.end_time = time.time()
            return self.elapsed
        
        @property
        def elapsed(self):
            if self.start_time and self.end_time:
                return self.end_time - self.start_time
            return None
    
    return Timer()


# Test data generators
def generate_random_bbox(image_width=800, image_height=600):
    """Generate a random bounding box."""
    x1 = np.random.randint(0, image_width - 100)
    y1 = np.random.randint(0, image_height - 100)
    x2 = np.random.randint(x1 + 50, min(x1 + 200, image_width))
    y2 = np.random.randint(y1 + 50, min(y1 + 150, image_height))
    return [x1, y1, x2, y2]


def generate_test_detections(count=10, labels=None):
    """Generate test detection data."""
    if labels is None:
        labels = ['warship', 'ship', 'vessel', 'boat']
    
    detections = []
    for i in range(count):
        detection = {
            'bbox': generate_random_bbox(),
            'label': np.random.choice(labels),
            'confidence': np.random.uniform(0.3, 0.95),
            'page': np.random.randint(1, 4)
        }
        detections.append(detection)
    
    return detections


# Test markers
pytest.mark.unit = pytest.mark.unit
pytest.mark.integration = pytest.mark.integration
pytest.mark.performance = pytest.mark.performance
pytest.mark.slow = pytest.mark.slow