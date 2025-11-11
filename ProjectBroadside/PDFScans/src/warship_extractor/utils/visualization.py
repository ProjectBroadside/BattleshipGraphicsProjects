"""
Visualization utilities for the warship extractor system.

This module provides tools for visualizing detection results, bounding boxes,
and creating summary reports with images and statistics.
"""

import json
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import cv2
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import numpy as np
from PIL import Image, ImageDraw, ImageFont

from ..config.settings import settings
from .logger import get_logger

logger = get_logger(__name__)


def draw_bounding_boxes(
    image: Union[np.ndarray, Image.Image, str, Path],
    detections: List[Dict[str, Any]],
    output_path: Optional[Path] = None,
    show_labels: bool = True,
    show_confidence: bool = True,
    font_size: int = 12,
    line_thickness: int = 2,
    colors: Optional[Dict[str, Tuple[int, int, int]]] = None
) -> Union[np.ndarray, Image.Image]:
    """
    Draw bounding boxes on an image with detection results.
    
    Args:
        image: Input image (array, PIL Image, or path)
        detections: List of detection dictionaries with bbox, label, confidence
        output_path: Optional path to save the annotated image
        show_labels: Whether to show labels on boxes
        show_confidence: Whether to show confidence scores
        font_size: Font size for labels
        line_thickness: Thickness of bounding box lines
        colors: Custom colors for different labels
        
    Returns:
        Annotated image
    """
    # Load image if path provided
    if isinstance(image, (str, Path)):
        image = Image.open(image)
    elif isinstance(image, np.ndarray):
        image = Image.fromarray(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
    
    # Create a copy for annotation
    annotated = image.copy()
    draw = ImageDraw.Draw(annotated)
    
    # Default colors for different detection types
    default_colors = {
        'warship': (255, 0, 0),      # Red
        'ship': (0, 255, 0),         # Green
        'vessel': (0, 0, 255),       # Blue
        'boat': (255, 255, 0),       # Yellow
        'default': (255, 165, 0)     # Orange
    }
    colors = colors or default_colors
    
    # Try to load a font
    try:
        font = ImageFont.truetype("arial.ttf", font_size)
    except (OSError, IOError):
        try:
            font = ImageFont.load_default()
        except:
            font = None
    
    for detection in detections:
        bbox = detection.get('bbox', [])
        label = detection.get('label', 'unknown')
        confidence = detection.get('confidence', 0.0)
        
        if len(bbox) != 4:
            continue
        
        x1, y1, x2, y2 = bbox
        
        # Determine color
        color = colors.get(label.lower(), colors.get('default', (255, 165, 0)))
        
        # Draw bounding box
        draw.rectangle([x1, y1, x2, y2], outline=color, width=line_thickness)
        
        # Prepare label text
        if show_labels or show_confidence:
            text_parts = []
            if show_labels:
                text_parts.append(label)
            if show_confidence:
                text_parts.append(f"{confidence:.2f}")
            text = " ".join(text_parts)
            
            # Draw label background
            if font:
                bbox_text = draw.textbbox((x1, y1), text, font=font)
                text_width = bbox_text[2] - bbox_text[0]
                text_height = bbox_text[3] - bbox_text[1]
            else:
                text_width = len(text) * 6  # Approximate
                text_height = 12
            
            # Background rectangle
            draw.rectangle(
                [x1, y1 - text_height - 4, x1 + text_width + 4, y1],
                fill=color
            )
            
            # Text
            draw.text(
                (x1 + 2, y1 - text_height - 2),
                text,
                fill=(255, 255, 255),
                font=font
            )
    
    # Save if output path provided
    if output_path:
        output_path.parent.mkdir(parents=True, exist_ok=True)
        annotated.save(output_path)
        logger.info(f"Annotated image saved to {output_path}")
    
    return annotated


def create_detection_grid(
    detections: List[Dict[str, Any]],
    images: List[Union[np.ndarray, Image.Image]],
    output_path: Optional[Path] = None,
    grid_size: Tuple[int, int] = (3, 3),
    figsize: Tuple[int, int] = (15, 15),
    show_metadata: bool = True
) -> None:
    """
    Create a grid visualization of detected warships.
    
    Args:
        detections: List of detection results
        images: List of cropped detection images
        output_path: Path to save the grid image
        grid_size: (rows, cols) for the grid
        figsize: Figure size in inches
        show_metadata: Whether to show detection metadata
    """
    rows, cols = grid_size
    fig, axes = plt.subplots(rows, cols, figsize=figsize)
    
    # Flatten axes for easier indexing
    if rows == 1 and cols == 1:
        axes = [axes]
    elif rows == 1 or cols == 1:
        axes = axes.flatten()
    else:
        axes = axes.flatten()
    
    for i in range(rows * cols):
        ax = axes[i]
        
        if i < len(images) and i < len(detections):
            # Display image
            if isinstance(images[i], np.ndarray):
                ax.imshow(cv2.cvtColor(images[i], cv2.COLOR_BGR2RGB))
            else:
                ax.imshow(images[i])
            
            # Add metadata if requested
            if show_metadata:
                detection = detections[i]
                title_parts = []
                
                if 'label' in detection:
                    title_parts.append(detection['label'])
                if 'confidence' in detection:
                    title_parts.append(f"({detection['confidence']:.2f})")
                
                title = " ".join(title_parts) if title_parts else f"Detection {i+1}"
                ax.set_title(title, fontsize=10)
        else:
            # Empty subplot
            ax.axis('off')
        
        ax.set_xticks([])
        ax.set_yticks([])
    
    plt.tight_layout()
    
    if output_path:
        output_path.parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        logger.info(f"Detection grid saved to {output_path}")
    
    plt.show()


def create_summary_report(
    extraction_results: Dict[str, Any],
    output_dir: Path,
    include_images: bool = True,
    max_images_per_page: int = 6
) -> Path:
    """
    Create a comprehensive HTML summary report of extraction results.
    
    Args:
        extraction_results: Results from extraction pipeline
        output_dir: Directory to save report and assets
        include_images: Whether to include detection images
        max_images_per_page: Maximum number of images per page
        
    Returns:
        Path to the generated HTML report
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Create assets directory for images
    assets_dir = output_dir / "assets"
    assets_dir.mkdir(exist_ok=True)
    
    # Extract key information
    stats = extraction_results.get('statistics', {})
    detections = extraction_results.get('detections', [])
    metadata = extraction_results.get('metadata', {})
    
    # Generate HTML content
    html_content = f"""
    <!DOCTYPE html>
    <html lang="en">
    <head>
        <meta charset="UTF-8">
        <meta name="viewport" content="width=device-width, initial-scale=1.0">
        <title>Warship Extraction Report</title>
        <style>
            body {{
                font-family: Arial, sans-serif;
                margin: 0;
                padding: 20px;
                background-color: #f5f5f5;
            }}
            .container {{
                max-width: 1200px;
                margin: 0 auto;
                background-color: white;
                padding: 30px;
                border-radius: 10px;
                box-shadow: 0 0 10px rgba(0,0,0,0.1);
            }}
            .header {{
                text-align: center;
                border-bottom: 2px solid #333;
                padding-bottom: 20px;
                margin-bottom: 30px;
            }}
            .stats-grid {{
                display: grid;
                grid-template-columns: repeat(auto-fit, minmax(200px, 1fr));
                gap: 20px;
                margin-bottom: 30px;
            }}
            .stat-card {{
                background-color: #f8f9fa;
                padding: 20px;
                border-radius: 8px;
                text-align: center;
                border-left: 4px solid #007bff;
            }}
            .stat-number {{
                font-size: 2em;
                font-weight: bold;
                color: #007bff;
            }}
            .stat-label {{
                color: #666;
                margin-top: 5px;
            }}
            .image-grid {{
                display: grid;
                grid-template-columns: repeat(auto-fit, minmax(250px, 1fr));
                gap: 20px;
                margin-top: 20px;
            }}
            .image-card {{
                background-color: #f8f9fa;
                padding: 15px;
                border-radius: 8px;
                text-align: center;
            }}
            .image-card img {{
                max-width: 100%;
                height: auto;
                border-radius: 4px;
                margin-bottom: 10px;
            }}
            .metadata {{
                background-color: #f8f9fa;
                padding: 20px;
                border-radius: 8px;
                margin-top: 30px;
            }}
            .metadata pre {{
                background-color: white;
                padding: 15px;
                border-radius: 4px;
                overflow-x: auto;
            }}
        </style>
    </head>
    <body>
        <div class="container">
            <div class="header">
                <h1>Warship Extraction Report</h1>
                <p>Generated on {metadata.get('extraction_date', 'Unknown')}</p>
            </div>
            
            <div class="stats-grid">
                <div class="stat-card">
                    <div class="stat-number">{stats.get('total_detections', 0)}</div>
                    <div class="stat-label">Total Detections</div>
                </div>
                <div class="stat-card">
                    <div class="stat-number">{stats.get('pages_processed', 0)}</div>
                    <div class="stat-label">Pages Processed</div>
                </div>
                <div class="stat-card">
                    <div class="stat-number">{stats.get('processing_time', 0):.1f}s</div>
                    <div class="stat-label">Processing Time</div>
                </div>
                <div class="stat-card">
                    <div class="stat-number">{stats.get('avg_confidence', 0):.2f}</div>
                    <div class="stat-label">Avg Confidence</div>
                </div>
            </div>
    """
    
    # Add detection images if requested
    if include_images and detections:
        html_content += """
            <h2>Detected Warships</h2>
            <div class="image-grid">
        """
        
        for i, detection in enumerate(detections[:max_images_per_page]):
            if 'cropped_image' in detection:
                # Save cropped image
                image_filename = f"detection_{i:03d}.png"
                image_path = assets_dir / image_filename
                
                if isinstance(detection['cropped_image'], np.ndarray):
                    cv2.imwrite(str(image_path), detection['cropped_image'])
                else:
                    detection['cropped_image'].save(image_path)
                
                # Add to HTML
                html_content += f"""
                    <div class="image-card">
                        <img src="assets/{image_filename}" alt="Detection {i+1}">
                        <div><strong>{detection.get('label', 'Unknown')}</strong></div>
                        <div>Confidence: {detection.get('confidence', 0):.2f}</div>
                        <div>Page: {detection.get('page', 'Unknown')}</div>
                    </div>
                """
        
        html_content += """
            </div>
        """
    
    # Add metadata section
    html_content += f"""
            <div class="metadata">
                <h2>Processing Metadata</h2>
                <pre>{json.dumps(metadata, indent=2)}</pre>
            </div>
        </div>
    </body>
    </html>
    """
    
    # Save HTML report
    report_path = output_dir / "extraction_report.html"
    with open(report_path, 'w', encoding='utf-8') as f:
        f.write(html_content)
    
    logger.info(f"Summary report generated: {report_path}")
    return report_path


def plot_confidence_distribution(
    detections: List[Dict[str, Any]],
    output_path: Optional[Path] = None,
    bins: int = 20,
    figsize: Tuple[int, int] = (10, 6)
) -> None:
    """
    Plot the distribution of confidence scores.
    
    Args:
        detections: List of detection results
        output_path: Path to save the plot
        bins: Number of histogram bins
        figsize: Figure size in inches
    """
    confidences = [d.get('confidence', 0) for d in detections if 'confidence' in d]
    
    if not confidences:
        logger.warning("No confidence scores found in detections")
        return
    
    plt.figure(figsize=figsize)
    plt.hist(confidences, bins=bins, alpha=0.7, edgecolor='black')
    plt.xlabel('Confidence Score')
    plt.ylabel('Number of Detections')
    plt.title('Distribution of Detection Confidence Scores')
    plt.grid(True, alpha=0.3)
    
    # Add statistics
    mean_conf = np.mean(confidences)
    plt.axvline(mean_conf, color='red', linestyle='--', 
                label=f'Mean: {mean_conf:.3f}')
    plt.legend()
    
    if output_path:
        output_path.parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        logger.info(f"Confidence distribution plot saved to {output_path}")
    
    plt.show()


def create_page_overview(
    page_image: Union[np.ndarray, Image.Image],
    detections: List[Dict[str, Any]],
    page_number: int,
    output_path: Optional[Path] = None
) -> Union[np.ndarray, Image.Image]:
    """
    Create an overview image showing all detections on a PDF page.
    
    Args:
        page_image: Original page image
        detections: List of detections for this page
        page_number: Page number for labeling
        output_path: Optional path to save the overview
        
    Returns:
        Annotated page overview image
    """
    # Filter detections for this page
    page_detections = [d for d in detections if d.get('page') == page_number]
    
    # Draw bounding boxes
    overview = draw_bounding_boxes(
        page_image,
        page_detections,
        show_labels=True,
        show_confidence=True,
        line_thickness=3,
        font_size=14
    )
    
    # Add page title
    draw = ImageDraw.Draw(overview)
    try:
        font = ImageFont.truetype("arial.ttf", 24)
    except:
        font = None
    
    title = f"Page {page_number} - {len(page_detections)} detections"
    draw.text((10, 10), title, fill=(0, 0, 0), font=font)
    
    if output_path:
        output_path.parent.mkdir(parents=True, exist_ok=True)
        overview.save(output_path)
        logger.info(f"Page overview saved to {output_path}")
    
    return overview


def save_detection_metadata(
    detections: List[Dict[str, Any]],
    output_path: Path,
    format: str = 'json'
) -> None:
    """
    Save detection metadata in various formats.
    
    Args:
        detections: List of detection results
        output_path: Path to save metadata
        format: Output format ('json', 'csv', 'txt')
    """
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    if format.lower() == 'json':
        with open(output_path, 'w') as f:
            json.dump(detections, f, indent=2, default=str)
    
    elif format.lower() == 'csv':
        import pandas as pd
        
        # Flatten detection data for CSV
        rows = []
        for i, detection in enumerate(detections):
            row = {
                'detection_id': i,
                'label': detection.get('label', ''),
                'confidence': detection.get('confidence', 0),
                'page': detection.get('page', ''),
                'bbox_x1': detection.get('bbox', [0, 0, 0, 0])[0],
                'bbox_y1': detection.get('bbox', [0, 0, 0, 0])[1],
                'bbox_x2': detection.get('bbox', [0, 0, 0, 0])[2],
                'bbox_y2': detection.get('bbox', [0, 0, 0, 0])[3],
            }
            rows.append(row)
        
        df = pd.DataFrame(rows)
        df.to_csv(output_path, index=False)
    
    elif format.lower() == 'txt':
        with open(output_path, 'w') as f:
            f.write("Warship Detection Results\n")
            f.write("=" * 40 + "\n\n")
            
            for i, detection in enumerate(detections):
                f.write(f"Detection {i+1}:\n")
                f.write(f"  Label: {detection.get('label', 'Unknown')}\n")
                f.write(f"  Confidence: {detection.get('confidence', 0):.3f}\n")
                f.write(f"  Page: {detection.get('page', 'Unknown')}\n")
                if 'bbox' in detection:
                    bbox = detection['bbox']
                    f.write(f"  Bounding Box: [{bbox[0]}, {bbox[1]}, {bbox[2]}, {bbox[3]}]\n")
                f.write("\n")
    
    logger.info(f"Detection metadata saved to {output_path} ({format} format)")