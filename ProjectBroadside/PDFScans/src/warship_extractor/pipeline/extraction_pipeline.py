"""
Main extraction pipeline that orchestrates all components for warship detection.

This module provides the high-level pipeline that coordinates PDF processing,
image conversion, warship detection, post-processing, and output generation.
"""

import json
import logging
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

from PIL import Image

from ..config.settings import settings
from ..core.model_manager import ModelManager
from ..core.pdf_processor import PDFProcessor
from ..detection.detector import WarshipDetector, Detection
from ..detection.prompt_strategies import WarshipPromptStrategy
from ..processing.nms_filter import NMSFilter
from ..processing.image_processor import ImageProcessor

logger = logging.getLogger(__name__)


class ExtractionResult:
    """Represents the result of processing a single PDF page."""
    
    def __init__(
        self,
        page_number: int,
        detections: List[Detection],
        output_paths: List[Path],
        metadata: dict,
        processing_time: float
    ):
        """
        Initialize extraction result.
        
        Args:
            page_number: Page number (1-indexed)
            detections: List of Detection objects
            output_paths: List of output file paths
            metadata: Page metadata
            processing_time: Processing time in seconds
        """
        self.page_number = page_number
        self.detections = detections
        self.output_paths = output_paths
        self.metadata = metadata
        self.processing_time = processing_time
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert result to dictionary."""
        return {
            "page_number": self.page_number,
            "detections": [d.to_dict() for d in self.detections],
            "output_paths": [str(p) for p in self.output_paths],
            "metadata": self.metadata,
            "processing_time": self.processing_time
        }


class PipelineStats:
    """Tracks pipeline performance statistics."""
    
    def __init__(self):
        """Initialize pipeline statistics."""
        self.reset()
    
    def reset(self):
        """Reset all statistics."""
        self.total_pages = 0
        self.successful_pages = 0
        self.total_detections = 0
        self.total_processing_time = 0.0
        self.error_count = 0
        self.errors = []
        self.start_time = None
        self.end_time = None
    
    def start_pipeline(self):
        """Mark pipeline start."""
        self.start_time = time.time()
    
    def end_pipeline(self):
        """Mark pipeline end."""
        self.end_time = time.time()
    
    def add_page_result(self, result: ExtractionResult):
        """Add a page result to statistics."""
        self.successful_pages += 1
        self.total_detections += len(result.detections)
        self.total_processing_time += result.processing_time
    
    def add_error(self, page_number: int, error: str):
        """Add an error to statistics."""
        self.error_count += 1
        self.errors.append({"page": page_number, "error": error})
    
    def get_summary(self) -> Dict[str, Any]:
        """Get pipeline statistics summary."""
        total_time = (self.end_time or time.time()) - (self.start_time or 0)
        
        return {
            "total_pages": self.total_pages,
            "successful_pages": self.successful_pages,
            "failed_pages": self.error_count,
            "success_rate": self.successful_pages / max(1, self.total_pages),
            "total_detections": self.total_detections,
            "avg_detections_per_page": self.total_detections / max(1, self.successful_pages),
            "total_pipeline_time": total_time,
            "total_processing_time": self.total_processing_time,
            "avg_processing_time_per_page": self.total_processing_time / max(1, self.successful_pages),
            "error_count": self.error_count,
            "errors": self.errors
        }


class ExtractionPipeline:
    """
    Main pipeline for extracting warship illustrations from PDFs.
    """
    
    def __init__(
        self,
        model_manager: Optional[ModelManager] = None,
        pdf_processor: Optional[PDFProcessor] = None,
        detector: Optional[WarshipDetector] = None,
        nms_filter: Optional[NMSFilter] = None,
        image_processor: Optional[ImageProcessor] = None,
        prompt_strategy: Optional[WarshipPromptStrategy] = None
    ):
        """
        Initialize the extraction pipeline.
        
        Args:
            model_manager: Model manager instance (creates new if None)
            pdf_processor: PDF processor instance (creates new if None)
            detector: Warship detector instance (creates new if None)
            nms_filter: NMS filter instance (creates new if None)
            image_processor: Image processor instance (creates new if None)
            prompt_strategy: Prompt strategy instance (creates new if None)
        """
        self.prompt_strategy = prompt_strategy or WarshipPromptStrategy()
        self.model_manager = model_manager or ModelManager()
        self.pdf_processor = pdf_processor or PDFProcessor()
        self.detector = detector or WarshipDetector(
            model_manager=self.model_manager,
            prompt_strategy=self.prompt_strategy
        )
        self.nms_filter = nms_filter or NMSFilter()
        self.image_processor = image_processor or ImageProcessor()
        
        self.stats = PipelineStats()
        
        logger.info("ExtractionPipeline initialized with all components")
    
    def process_pdf(
        self,
        pdf_path: Union[str, Path],
        output_dir: Optional[Union[str, Path]] = None,
        page_numbers: Optional[List[int]] = None,
        dpi: Optional[int] = None,
        strategy: str = "optimized",
        save_visualizations: Optional[bool] = None,
        save_metadata: Optional[bool] = None
    ) -> List[ExtractionResult]:
        """
        Process a complete PDF file and extract warship illustrations.
        
        Args:
            pdf_path: Path to the PDF file
            output_dir: Output directory (uses settings if None)
            page_numbers: Specific pages to process (1-indexed), None for all
            dpi: DPI for image conversion (uses settings if None)
            strategy: Detection strategy ("optimized", "primary", "technical", etc.)
            save_visualizations: Whether to save annotated images
            save_metadata: Whether to save detection metadata
            
        Returns:
            List of ExtractionResult objects
        """
        pdf_path = Path(pdf_path)
        
        if not pdf_path.exists():
            raise FileNotFoundError(f"PDF file not found: {pdf_path}")
        
        if output_dir is None:
            output_dir = settings.get_output_path(pdf_path.stem)
        else:
            output_dir = Path(output_dir)
        
        output_dir.mkdir(parents=True, exist_ok=True)
        
        save_visualizations = save_visualizations if save_visualizations is not None else settings.save_visualizations
        save_metadata = save_metadata if save_metadata is not None else settings.save_metadata
        
        logger.info(f"Starting PDF processing: {pdf_path.name}")
        logger.info(f"Output directory: {output_dir}")
        
        self.stats.reset()
        self.stats.start_pipeline()
        
        # Get page count for progress tracking
        total_pages = self.pdf_processor.get_page_count(pdf_path)
        if page_numbers is not None:
            pages_to_process = [p for p in page_numbers if 1 <= p <= total_pages]
            self.stats.total_pages = len(pages_to_process)
        else:
            self.stats.total_pages = total_pages
        
        logger.info(f"Processing {self.stats.total_pages} pages from {total_pages} total pages")
        
        # Ensure model is loaded
        if not self.model_manager.is_loaded():
            logger.info("Loading Florence-2 model...")
            self.model_manager.load_model()
        
        results = []
        
        # Process pages
        for image, page_metadata in self.pdf_processor.process_pages(pdf_path, page_numbers, dpi):
            page_number = page_metadata["page_number"]
            
            try:
                logger.info(f"Processing page {page_number}/{total_pages}")
                
                result = self._process_single_page(
                    image=image,
                    page_metadata=page_metadata,
                    output_dir=output_dir,
                    strategy=strategy,
                    save_visualizations=save_visualizations
                )
                
                results.append(result)
                self.stats.add_page_result(result)
                
                logger.info(f"Page {page_number} completed: {len(result.detections)} detections in {result.processing_time:.2f}s")
                
            except Exception as e:
                error_msg = f"Failed to process page {page_number}: {str(e)}"
                logger.error(error_msg)
                self.stats.add_error(page_number, str(e))
                continue
        
        self.stats.end_pipeline()
        
        # Save comprehensive metadata
        if save_metadata:
            self._save_pipeline_metadata(results, output_dir, pdf_path)
        
        # Log final statistics
        summary = self.stats.get_summary()
        logger.info(f"Pipeline completed: {summary['successful_pages']}/{summary['total_pages']} pages processed")
        logger.info(f"Total detections: {summary['total_detections']}")
        logger.info(f"Success rate: {summary['success_rate']:.1%}")
        logger.info(f"Total time: {summary['total_pipeline_time']:.2f}s")
        
        return results
    
    def _process_single_page(
        self,
        image: Image.Image,
        page_metadata: dict,
        output_dir: Path,
        strategy: str,
        save_visualizations: bool
    ) -> ExtractionResult:
        """Process a single PDF page."""
        start_time = time.time()
        page_number = page_metadata["page_number"]
        
        # Run detection
        raw_detections = self.detector.detect_warships(
            image=image,
            strategy=strategy
        )
        
        logger.debug(f"Page {page_number}: Found {len(raw_detections)} raw detections")
        
        # Apply post-processing filters
        filtered_detections = self.nms_filter.apply_comprehensive_filtering(raw_detections)
        
        logger.debug(f"Page {page_number}: {len(filtered_detections)} detections after filtering")
        
        # Save extracted regions
        output_paths = []
        if filtered_detections:
            page_output_dir = output_dir / "extracted_images"
            filename_prefix = f"{page_metadata['pdf_name']}_page_{page_number:03d}"
            
            saved_regions = self.image_processor.process_detection_batch(
                image=image,
                detections=filtered_detections,
                output_dir=page_output_dir,
                filename_prefix=filename_prefix
            )
            
            output_paths = [path for path, _ in saved_regions]
        
        # Save visualization if requested
        if save_visualizations:
            viz_dir = output_dir / "visualizations"
            viz_path = viz_dir / f"{page_metadata['pdf_name']}_page_{page_number:03d}_annotated.png"
            
            saved_viz = self.image_processor.create_detection_summary_image(
                image=image,
                detections=filtered_detections,
                output_path=viz_path
            )
            
            if saved_viz:
                output_paths.append(saved_viz)
        
        processing_time = time.time() - start_time
        
        # Create result
        result = ExtractionResult(
            page_number=page_number,
            detections=filtered_detections,
            output_paths=output_paths,
            metadata=page_metadata,
            processing_time=processing_time
        )
        
        return result
    
    def _save_pipeline_metadata(
        self,
        results: List[ExtractionResult],
        output_dir: Path,
        pdf_path: Path
    ) -> None:
        """Save comprehensive pipeline metadata."""
        metadata_dir = output_dir / "metadata"
        metadata_dir.mkdir(exist_ok=True)
        
        # Save individual page results
        for result in results:
            page_metadata_path = metadata_dir / f"page_{result.page_number:03d}_detections.json"
            with open(page_metadata_path, 'w') as f:
                json.dump(result.to_dict(), f, indent=2)
        
        # Save pipeline summary
        summary_data = {
            "pdf_info": {
                "name": pdf_path.name,
                "path": str(pdf_path),
                "total_pages": len(results)
            },
            "processing_summary": self.stats.get_summary(),
            "detection_summary": {
                "total_detections": sum(len(r.detections) for r in results),
                "pages_with_detections": sum(1 for r in results if r.detections),
                "avg_detections_per_page": sum(len(r.detections) for r in results) / max(1, len(results))
            },
            "settings_used": {
                "dpi": self.pdf_processor.dpi,
                "iou_threshold": self.nms_filter.iou_threshold,
                "min_detection_area": settings.min_detection_area,
                "max_detection_area": settings.max_detection_area,
                "enhance_images": self.image_processor.enhance_images,
                "padding_pixels": self.image_processor.padding
            }
        }
        
        summary_path = metadata_dir / "pipeline_summary.json"
        with open(summary_path, 'w') as f:
            json.dump(summary_data, f, indent=2)
        
        logger.info(f"Saved pipeline metadata to {metadata_dir}")
    
    def process_multiple_pdfs(
        self,
        pdf_paths: List[Union[str, Path]],
        output_base_dir: Optional[Union[str, Path]] = None,
        **kwargs
    ) -> Dict[str, List[ExtractionResult]]:
        """
        Process multiple PDF files.
        
        Args:
            pdf_paths: List of PDF file paths
            output_base_dir: Base output directory
            **kwargs: Arguments passed to process_pdf
            
        Returns:
            Dictionary mapping PDF names to their results
        """
        if output_base_dir is None:
            output_base_dir = settings.get_output_path("batch_processing")
        else:
            output_base_dir = Path(output_base_dir)
        
        output_base_dir.mkdir(parents=True, exist_ok=True)
        
        all_results = {}
        
        logger.info(f"Starting batch processing of {len(pdf_paths)} PDFs")
        
        for i, pdf_path in enumerate(pdf_paths):
            pdf_path = Path(pdf_path)
            
            if not pdf_path.exists():
                logger.error(f"PDF not found: {pdf_path}")
                continue
            
            logger.info(f"Processing PDF {i+1}/{len(pdf_paths)}: {pdf_path.name}")
            
            try:
                # Create PDF-specific output directory
                pdf_output_dir = output_base_dir / pdf_path.stem
                
                # Process the PDF
                results = self.process_pdf(
                    pdf_path=pdf_path,
                    output_dir=pdf_output_dir,
                    **kwargs
                )
                
                all_results[pdf_path.name] = results
                
                # Clear memory between PDFs
                self.model_manager.optimize_memory()
                
            except Exception as e:
                logger.error(f"Failed to process PDF {pdf_path.name}: {str(e)}")
                continue
        
        logger.info(f"Batch processing completed: {len(all_results)}/{len(pdf_paths)} PDFs processed successfully")
        
        return all_results
    
    def estimate_processing_time(
        self,
        pdf_path: Union[str, Path],
        page_numbers: Optional[List[int]] = None
    ) -> Dict[str, Any]:
        """
        Estimate processing time for a PDF.
        
        Args:
            pdf_path: Path to the PDF file
            page_numbers: Specific pages to estimate for
            
        Returns:
            Dictionary with time estimates
        """
        pdf_path = Path(pdf_path)
        
        # Get page information
        page_info = self.pdf_processor.get_page_info(pdf_path)
        
        if page_numbers is not None:
            pages_to_estimate = [info for info in page_info if info["page_number"] in page_numbers]
        else:
            pages_to_estimate = page_info
        
        # Rough time estimates (these would be calibrated based on actual performance)
        base_time_per_page = 15.0  # seconds
        detection_time_per_page = 10.0  # seconds for detection
        processing_time_per_page = 2.0  # seconds for post-processing
        
        total_estimated_time = len(pages_to_estimate) * (
            base_time_per_page + detection_time_per_page + processing_time_per_page
        )
        
        return {
            "total_pages": len(page_info),
            "pages_to_process": len(pages_to_estimate),
            "estimated_time_seconds": total_estimated_time,
            "estimated_time_minutes": total_estimated_time / 60,
            "estimated_time_per_page": total_estimated_time / max(1, len(pages_to_estimate)),
            "memory_estimate": self.pdf_processor.estimate_memory_usage(pdf_path)
        }
    
    def get_pipeline_status(self) -> Dict[str, Any]:
        """Get current pipeline status and statistics."""
        return {
            "model_loaded": self.model_manager.is_loaded(),
            "memory_usage": self.model_manager.get_memory_usage(),
            "current_stats": self.stats.get_summary(),
            "detector_stats": self.detector.get_performance_stats()
        }
    
    def cleanup(self) -> None:
        """Clean up resources and optimize memory."""
        logger.info("Cleaning up pipeline resources")
        
        if self.model_manager:
            self.model_manager.optimize_memory()
        
        if self.detector:
            self.detector.optimize_memory()
    
    def __enter__(self):
        """Context manager entry."""
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit."""
        self.cleanup()