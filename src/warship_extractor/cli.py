"""
Command-line interface for the warship extractor system.

This module provides a comprehensive CLI for running warship extraction
with various options and configurations.
"""

import argparse
import sys
import time
from pathlib import Path
from typing import Optional

from .config.settings import settings
from .pipeline.extraction_pipeline import ExtractionPipeline
from .utils.logger import setup_logging, log_system_info, main_logger
from .utils.visualization import create_summary_report


def create_parser() -> argparse.ArgumentParser:
    """
    Create the command-line argument parser.
    
    Returns:
        Configured argument parser
    """
    parser = argparse.ArgumentParser(
        description="Extract warship illustrations from historical PDF documents using Florence-2",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Basic extraction
  warship-extractor extract input.pdf

  # Extract with custom output directory
  warship-extractor extract input.pdf --output-dir /path/to/output

  # Extract with specific confidence threshold
  warship-extractor extract input.pdf --confidence-threshold 0.8

  # Extract specific pages only
  warship-extractor extract input.pdf --pages 1-10,15,20-25

  # Batch process multiple PDFs
  warship-extractor batch /path/to/pdfs --pattern "*.pdf"

  # Generate report from existing results
  warship-extractor report /path/to/results.json --output-dir /path/to/report
        """
    )
    
    # Add version
    parser.add_argument(
        '--version',
        action='version',
        version='%(prog)s 1.0.0'
    )
    
    # Global options
    parser.add_argument(
        '--log-level',
        choices=['DEBUG', 'INFO', 'WARNING', 'ERROR'],
        default='INFO',
        help='Set logging level (default: INFO)'
    )
    
    parser.add_argument(
        '--log-file',
        type=Path,
        help='Path to log file (default: auto-generated)'
    )
    
    parser.add_argument(
        '--config',
        type=Path,
        help='Path to configuration file'
    )
    
    parser.add_argument(
        '--no-color',
        action='store_true',
        help='Disable colored output'
    )
    
    # Create subparsers
    subparsers = parser.add_subparsers(dest='command', help='Available commands')
    
    # Extract command
    extract_parser = subparsers.add_parser(
        'extract',
        help='Extract warships from a single PDF'
    )
    extract_parser.add_argument(
        'input_file',
        type=Path,
        help='Path to input PDF file'
    )
    extract_parser.add_argument(
        '--output-dir',
        type=Path,
        help='Output directory for results (default: auto-generated)'
    )
    extract_parser.add_argument(
        '--confidence-threshold',
        type=float,
        default=0.3,
        help='Minimum confidence threshold for detections (default: 0.3)'
    )
    extract_parser.add_argument(
        '--pages',
        type=str,
        help='Pages to process (e.g., "1-10,15,20-25", default: all)'
    )
    extract_parser.add_argument(
        '--max-pages',
        type=int,
        help='Maximum number of pages to process'
    )
    extract_parser.add_argument(
        '--dpi',
        type=int,
        default=300,
        help='DPI for PDF rendering (default: 300)'
    )
    extract_parser.add_argument(
        '--no-nms',
        action='store_true',
        help='Disable Non-Maximum Suppression filtering'
    )
    extract_parser.add_argument(
        '--no-enhancement',
        action='store_true',
        help='Disable image enhancement'
    )
    extract_parser.add_argument(
        '--save-debug',
        action='store_true',
        help='Save debug images and intermediate results'
    )
    extract_parser.add_argument(
        '--generate-report',
        action='store_true',
        help='Generate HTML summary report'
    )
    
    # Batch command
    batch_parser = subparsers.add_parser(
        'batch',
        help='Process multiple PDF files'
    )
    batch_parser.add_argument(
        'input_dir',
        type=Path,
        help='Directory containing PDF files'
    )
    batch_parser.add_argument(
        '--pattern',
        type=str,
        default='*.pdf',
        help='File pattern to match (default: *.pdf)'
    )
    batch_parser.add_argument(
        '--output-dir',
        type=Path,
        help='Base output directory for all results'
    )
    batch_parser.add_argument(
        '--confidence-threshold',
        type=float,
        default=0.3,
        help='Minimum confidence threshold for detections (default: 0.3)'
    )
    batch_parser.add_argument(
        '--max-files',
        type=int,
        help='Maximum number of files to process'
    )
    batch_parser.add_argument(
        '--parallel',
        action='store_true',
        help='Process files in parallel (experimental)'
    )
    batch_parser.add_argument(
        '--generate-reports',
        action='store_true',
        help='Generate HTML reports for each file'
    )
    
    # Report command
    report_parser = subparsers.add_parser(
        'report',
        help='Generate report from existing results'
    )
    report_parser.add_argument(
        'results_file',
        type=Path,
        help='Path to results JSON file'
    )
    report_parser.add_argument(
        '--output-dir',
        type=Path,
        help='Output directory for report'
    )
    report_parser.add_argument(
        '--template',
        type=str,
        choices=['standard', 'detailed', 'compact'],
        default='standard',
        help='Report template to use (default: standard)'
    )
    
    # Info command
    info_parser = subparsers.add_parser(
        'info',
        help='Show system information and configuration'
    )
    info_parser.add_argument(
        '--show-config',
        action='store_true',
        help='Show current configuration'
    )
    info_parser.add_argument(
        '--show-model',
        action='store_true',
        help='Show model information'
    )
    
    return parser


def parse_page_ranges(page_spec: str) -> list[int]:
    """
    Parse page specification string into list of page numbers.
    
    Args:
        page_spec: Page specification (e.g., "1-10,15,20-25")
        
    Returns:
        List of page numbers
    """
    pages = []
    
    for part in page_spec.split(','):
        part = part.strip()
        
        if '-' in part:
            start, end = part.split('-', 1)
            start, end = int(start.strip()), int(end.strip())
            pages.extend(range(start, end + 1))
        else:
            pages.append(int(part))
    
    return sorted(list(set(pages)))


def extract_command(args) -> int:
    """
    Execute the extract command.
    
    Args:
        args: Parsed command-line arguments
        
    Returns:
        Exit code (0 for success, 1 for error)
    """
    try:
        # Validate input file
        if not args.input_file.exists():
            main_logger.error(f"Input file not found: {args.input_file}")
            return 1
        
        if not args.input_file.suffix.lower() == '.pdf':
            main_logger.error(f"Input file must be a PDF: {args.input_file}")
            return 1
        
        # Set up output directory
        if args.output_dir:
            output_dir = args.output_dir
        else:
            output_dir = Path(f"warship_extraction_{args.input_file.stem}_{int(time.time())}")
        
        output_dir.mkdir(parents=True, exist_ok=True)
        
        # Parse page ranges if specified
        pages_to_process = None
        if args.pages:
            try:
                pages_to_process = parse_page_ranges(args.pages)
                main_logger.info(f"Processing pages: {pages_to_process}")
            except ValueError as e:
                main_logger.error(f"Invalid page specification: {args.pages} - {e}")
                return 1
        
        # Update settings based on arguments
        extraction_settings = {
            'confidence_threshold': args.confidence_threshold,
            'pdf_dpi': args.dpi,
            'enable_nms': not args.no_nms,
            'enable_enhancement': not args.no_enhancement,
            'save_debug_images': args.save_debug,
            'max_pages': args.max_pages
        }
        
        main_logger.info(f"Starting extraction of {args.input_file}")
        main_logger.info(f"Output directory: {output_dir}")
        main_logger.info(f"Settings: {extraction_settings}")
        
        # Create and run pipeline
        pipeline = ExtractionPipeline(**extraction_settings)
        
        results = pipeline.process_pdf(
            pdf_path=args.input_file,
            output_dir=output_dir,
            pages=pages_to_process
        )
        
        # Generate report if requested
        if args.generate_report:
            main_logger.info("Generating summary report...")
            report_path = create_summary_report(results, output_dir)
            main_logger.info(f"Report generated: {report_path}")
        
        # Log final statistics
        stats = results.get('statistics', {})
        main_logger.info(f"Extraction completed successfully!")
        main_logger.info(f"Total detections: {stats.get('total_detections', 0)}")
        main_logger.info(f"Processing time: {stats.get('processing_time', 0):.1f}s")
        main_logger.info(f"Results saved to: {output_dir}")
        
        return 0
        
    except Exception as e:
        main_logger.error(f"Extraction failed: {e}", exc_info=True)
        return 1


def batch_command(args) -> int:
    """
    Execute the batch command.
    
    Args:
        args: Parsed command-line arguments
        
    Returns:
        Exit code (0 for success, 1 for error)
    """
    try:
        # Validate input directory
        if not args.input_dir.exists():
            main_logger.error(f"Input directory not found: {args.input_dir}")
            return 1
        
        # Find PDF files
        pdf_files = list(args.input_dir.glob(args.pattern))
        
        if not pdf_files:
            main_logger.error(f"No PDF files found in {args.input_dir} matching {args.pattern}")
            return 1
        
        # Limit number of files if specified
        if args.max_files:
            pdf_files = pdf_files[:args.max_files]
        
        main_logger.info(f"Found {len(pdf_files)} PDF files to process")
        
        # Set up base output directory
        if args.output_dir:
            base_output_dir = args.output_dir
        else:
            base_output_dir = Path(f"warship_extraction_batch_{int(time.time())}")
        
        base_output_dir.mkdir(parents=True, exist_ok=True)
        
        # Process each file
        successful = 0
        failed = 0
        
        for i, pdf_file in enumerate(pdf_files, 1):
            main_logger.info(f"Processing file {i}/{len(pdf_files)}: {pdf_file.name}")
            
            try:
                # Create subdirectory for this file
                file_output_dir = base_output_dir / pdf_file.stem
                file_output_dir.mkdir(exist_ok=True)
                
                # Create pipeline instance
                pipeline = ExtractionPipeline(
                    confidence_threshold=args.confidence_threshold
                )
                
                # Process the file
                results = pipeline.process_pdf(
                    pdf_path=pdf_file,
                    output_dir=file_output_dir
                )
                
                # Generate report if requested
                if args.generate_reports:
                    create_summary_report(results, file_output_dir)
                
                successful += 1
                stats = results.get('statistics', {})
                main_logger.info(f"✓ {pdf_file.name}: {stats.get('total_detections', 0)} detections")
                
            except Exception as e:
                failed += 1
                main_logger.error(f"✗ {pdf_file.name}: {e}")
                continue
        
        # Final summary
        main_logger.info(f"Batch processing completed!")
        main_logger.info(f"Successful: {successful}, Failed: {failed}")
        main_logger.info(f"Results saved to: {base_output_dir}")
        
        return 0 if failed == 0 else 1
        
    except Exception as e:
        main_logger.error(f"Batch processing failed: {e}", exc_info=True)
        return 1


def report_command(args) -> int:
    """
    Execute the report command.
    
    Args:
        args: Parsed command-line arguments
        
    Returns:
        Exit code (0 for success, 1 for error)
    """
    try:
        # Validate results file
        if not args.results_file.exists():
            main_logger.error(f"Results file not found: {args.results_file}")
            return 1
        
        # Load results
        import json
        with open(args.results_file, 'r') as f:
            results = json.load(f)
        
        # Set up output directory
        if args.output_dir:
            output_dir = args.output_dir
        else:
            output_dir = args.results_file.parent / "report"
        
        output_dir.mkdir(parents=True, exist_ok=True)
        
        # Generate report
        main_logger.info(f"Generating report from {args.results_file}")
        report_path = create_summary_report(results, output_dir)
        main_logger.info(f"Report generated: {report_path}")
        
        return 0
        
    except Exception as e:
        main_logger.error(f"Report generation failed: {e}", exc_info=True)
        return 1


def info_command(args) -> int:
    """
    Execute the info command.
    
    Args:
        args: Parsed command-line arguments
        
    Returns:
        Exit code (0 for success, 1 for error)
    """
    try:
        main_logger.info("=== Warship Extractor Information ===")
        
        # Show system information
        log_system_info(main_logger)
        
        # Show configuration if requested
        if args.show_config:
            main_logger.info("\n=== Configuration ===")
            main_logger.info(f"Model Name: {settings.model_name}")
            main_logger.info(f"Cache Directory: {settings.cache_dir}")
            main_logger.info(f"Output Directory: {settings.output_dir}")
            main_logger.info(f"Log Level: {settings.log_level}")
            main_logger.info(f"Enable GPU: {settings.enable_gpu}")
            main_logger.info(f"Batch Size: {settings.batch_size}")
            main_logger.info(f"PDF DPI: {settings.pdf_dpi}")
            main_logger.info(f"Confidence Threshold: {settings.confidence_threshold}")
        
        # Show model information if requested
        if args.show_model:
            from .core.model_manager import ModelManager
            
            main_logger.info("\n=== Model Information ===")
            model_manager = ModelManager()
            
            main_logger.info(f"Model: {model_manager.model_name}")
            main_logger.info(f"Device: {model_manager.device}")
            main_logger.info(f"Cache Path: {model_manager.cache_dir}")
            
            # Check if model is already downloaded
            if model_manager.is_model_cached():
                main_logger.info("Model Status: ✓ Cached locally")
            else:
                main_logger.info("Model Status: ✗ Not cached (will download on first use)")
        
        return 0
        
    except Exception as e:
        main_logger.error(f"Info command failed: {e}", exc_info=True)
        return 1


def main() -> int:
    """
    Main entry point for the CLI.
    
    Returns:
        Exit code (0 for success, 1 for error)
    """
    try:
        # Create parser and parse arguments
        parser = create_parser()
        args = parser.parse_args()
        
        # Show help if no command specified
        if not args.command:
            parser.print_help()
            return 1
        
        # Set up logging
        setup_logging(
            log_level=args.log_level,
            log_file=args.log_file,
            enable_colors=not args.no_color
        )
        
        main_logger.info(f"Warship Extractor CLI - Command: {args.command}")
        
        # Execute command
        if args.command == 'extract':
            return extract_command(args)
        elif args.command == 'batch':
            return batch_command(args)
        elif args.command == 'report':
            return report_command(args)
        elif args.command == 'info':
            return info_command(args)
        else:
            main_logger.error(f"Unknown command: {args.command}")
            return 1
            
    except KeyboardInterrupt:
        main_logger.info("Operation cancelled by user")
        return 1
    except Exception as e:
        if 'main_logger' in locals():
            main_logger.error(f"Unexpected error: {e}", exc_info=True)
        else:
            print(f"Fatal error: {e}", file=sys.stderr)
        return 1


if __name__ == '__main__':
    sys.exit(main())