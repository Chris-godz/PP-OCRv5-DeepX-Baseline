"""
Main entry point for OCR Image Viewer Application
Handles command line arguments and application startup
"""

import sys
import argparse

import dx_engine

from PySide6.QtWidgets import QApplication

from image_viewer import ImageViewerApp
from ocr_engine import create_ocr_workers


def main():
    """Main application entry point"""
    parser = argparse.ArgumentParser(description="OCR Image Viewer Application")
    parser.add_argument('--version', default='v4', choices=['v4', 'v5'], 
                       help='Model version to use (default: v4)')
    parser.add_argument('--workers', type=int, default=3,
                       help='Number of OCR worker instances (default: 3)')
    
    args = parser.parse_args()
    
    # Create OCR workers
    print(f"Initializing OCR workers with version {args.version}...")
    ocr_workers = create_ocr_workers(version=args.version, num_workers=args.workers)
    print(f"Created {len(ocr_workers)} OCR worker instances")
    
    # Start Qt application
    app = QApplication(sys.argv)
    viewer = ImageViewerApp(ocr_workers=ocr_workers)
    viewer.show()
    
    print("OCR Image Viewer started successfully")
    sys.exit(app.exec())


if __name__ == "__main__":
    main() 