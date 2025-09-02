"""
Main Image Viewer Application
Contains the main application class for OCR image processing
"""

import cv2
import numpy as np
import copy
import tqdm

from PySide6.QtWidgets import (
    QWidget, QHBoxLayout, QVBoxLayout, QLabel, QPushButton, 
    QFileDialog, QSizePolicy, QScrollArea, QTextEdit, QApplication
)
from PySide6.QtGui import QPixmap, QImage
from PySide6.QtCore import Qt, QSize

from collections import deque

from ui_components import PreviewGridWidget, ThumbnailListWidget, PerformanceInfoWidget
from engine.draw_utils import draw_with_poly_enhanced

# Global queue for preview images
preview_image_q = deque(maxlen=20)


class ImageViewerApp(QWidget):
    """Main application widget for OCR image processing and display"""
    
    def __init__(self, ocr_workers=None):
        super().__init__()
        self.setWindowTitle("PySide Image Viewer")
        
        # Set window to full screen size
        screen_size = QApplication.primaryScreen().geometry()
        self.setGeometry(100, 100, screen_size.width()/1.5, screen_size.height()/1.5)
        
        # Store OCR workers
        self.ocr_workers = ocr_workers if ocr_workers else []
        self.current_worker_index = 0  # Index for round-robin worker selection
        
        # Initialize UI first
        self.init_ui()
        
        # Store screen dimensions for reference
        self.max_width = screen_size.width()/2
        self.max_height = screen_size.height()/2
        self.last_result_image = None
        self.last_result_text = None

    def init_ui(self):
        """Initialize the user interface"""
        main_layout = QHBoxLayout()
        left_layout = QVBoxLayout()

        # Swap positions: Text first, then Preview
        info_title = QLabel("Inference Result Text")
        info_title.setStyleSheet("""
            QLabel {
                font-size: 11px;
                color: black;
                padding: 2px;
                border-radius: 2px;
                margin-bottom: 2px;
            }
        """)
        left_layout.addWidget(info_title)

        self.info_text = QTextEdit()
        self.info_text.setReadOnly(True)
        self.info_text.setTextInteractionFlags(
            Qt.TextInteractionFlag.TextSelectableByMouse |
            Qt.TextInteractionFlag.TextSelectableByKeyboard
        )

        text_scroll_area = QScrollArea()
        text_scroll_area.setWidgetResizable(True)
        text_scroll_area.setWidget(self.info_text)

        left_layout.addWidget(text_scroll_area, stretch=3)

        preview_title = QLabel("Processing Result Preview")
        preview_title.setStyleSheet("""
            QLabel {
                font-size: 11px;
                color: black;
                padding: 2px;
                border-radius: 2px;
                margin: 6px 0 3px 0;
            }
        """)
        left_layout.addWidget(preview_title)

        self.preview_grid = PreviewGridWidget()
        self.preview_grid.imageClicked.connect(self.handle_preview_click)

        scroll_area = QScrollArea()
        scroll_area.setWidgetResizable(True)
        scroll_area.setWidget(self.preview_grid)
        
        left_layout.addWidget(scroll_area, stretch=2)

        left_container = QWidget()
        left_container.setLayout(left_layout)

        main_layout.addWidget(left_container, stretch=1)

        center_layout = QVBoxLayout()

        main_image_title = QLabel("Main Image Viewer")
        main_image_title.setStyleSheet("""
            QLabel {
                font-size: 14px;
                color: black;
                padding: 2px;
                border-radius: 2px;
                margin-bottom: 3px;
            }
        """)

        center_layout.addWidget(main_image_title)
        
        self.image_label = QLabel("Please select an image")
        self.image_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.image_label.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding)
        self.image_label.setStyleSheet("border: 1px solid gray;")
        center_layout.addWidget(self.image_label, stretch=3)
        
        center_container = QWidget()
        center_container.setLayout(center_layout)
        main_layout.addWidget(center_container, stretch=3)

        right_layout = QVBoxLayout()

        # Add Performance Information widget before File Management
        self.performance_widget = PerformanceInfoWidget()
        right_layout.addWidget(self.performance_widget, stretch=1)

        file_title = QLabel("File Management")
        file_title.setStyleSheet("""
            QLabel {
                font-size: 11px;
                color: black;
                padding: 2px;
                border-radius: 2px;
                margin-bottom: 3px;
            }
        """)
        right_layout.addWidget(file_title)

        # Use ThumbnailListWidget with smaller thumbnail size for right layout
        self.file_list = ThumbnailListWidget(self.handle_file_list_click, thumbnail_size=50)
        right_layout.addWidget(self.file_list, stretch=5)

        upload_button = QPushButton("Upload Images")
        upload_button.clicked.connect(self.upload_images)
        right_layout.addWidget(upload_button)
        
        all_inference_button = QPushButton("Run All Inference")
        all_inference_button.clicked.connect(self.run_imagelist)
        right_layout.addWidget(all_inference_button)

        right_container = QWidget() 
        right_container.setLayout(right_layout)
        main_layout.addWidget(right_container, stretch=1)

        self.setLayout(main_layout)

    def get_next_worker(self):
        """
        Get next worker using round-robin selection
        """
        if not self.ocr_workers:
            return None
        
        worker = self.ocr_workers[self.current_worker_index]
        self.current_worker_index = (self.current_worker_index + 1) % len(self.ocr_workers)
        return worker

    def upload_images(self):
        """Open file dialog to select and upload images"""
        files, _ = QFileDialog.getOpenFileNames(
            self, "Select Image Files", "", "Images (*.png *.jpg *.jpeg *.bmp *.gif)"
        )
        for file_path in files:
            # Use add_thumbnail method for better visual representation
            self.file_list.add_thumbnail(file_path)

    def handle_file_list_click(self, file_path):
        """Handle file list item click - display selected image"""
        self.display_image(file_path)
    
    def handle_preview_click(self, index: int):
        """Handle preview image click - display cached result"""
        image, text, _ = preview_image_q[index]  # Extract as numpy array
        self.last_result_image = image           # Can be displayed directly or overwritten with inference results
        self.last_result_text = text
        self.update_display_from_cached()

    def ocr_run(self, image, file_path):
        """
        Run OCR inference on the given image
        
        Args:
            image: Input image for OCR processing
            file_path: Path to the image file
            
        Returns:
            tuple: (boxes, rotated_crops, rec_results)
        """
        worker = self.get_next_worker()
        if worker is None:
            print("No OCR workers available")
            return [], [], []
        
        try:
            boxes, rotated_crops, rec_results = worker(image)
            self.performance_widget.update_performance_data(
                det_npu=worker.detection_time_duration / worker.ocr_run_count,
                cls_npu=worker.classification_time_duration / worker.ocr_run_count,
                rec_npu=worker.recognition_time_duration / worker.ocr_run_count
            )
            return boxes, rotated_crops, rec_results
        except Exception as e:
            print(f"Error during OCR inference: {e}")
            return [], [], []
    
    def run_imagelist(self):
        """Run OCR inference on all loaded images"""
        if self.file_list.count() == 0:
            print("No image files available for inference.")
            return
        
        for i in tqdm.tqdm(range(self.file_list.count())):
            # Get file path from item data instead of text
            item = self.file_list.item(i)
            file_path = item.data(Qt.ItemDataRole.UserRole)
            image = cv2.imread(file_path)
            if image is None:
                self.image_label.setText(f"Failed to load image file: {file_path}")
                return
            boxes, rotated_crops, rec_results = self.ocr_run(image, file_path)
            self.last_result_image = self.ocr2image(image, boxes, boxes, rec_results)
            self.last_result_text = rec_results
            preview_image_q.append((copy.deepcopy(self.last_result_image), rec_results, file_path))
        
        self.preview_grid.set_images(preview_image_q)
        
    def update_display_from_cached(self):
        """Update display with cached result image and text"""
        if self.last_result_image is None:
            return
        
        rgb_image = self.last_result_image
        h, w, ch = rgb_image.shape
        bytes_per_line = ch * w
        qimage = QImage(rgb_image.data, w, h, bytes_per_line, QImage.Format.Format_RGB888)
        pixmap = QPixmap.fromImage(qimage)
        
        target_width = int(self.image_label.width() * 0.8)
        target_height = int(self.image_label.height() * 0.8)
        
        size = QSize(target_width, target_height)

        scaled = pixmap.scaled(
            size,
            Qt.AspectRatioMode.KeepAspectRatio,
            Qt.TransformationMode.SmoothTransformation
        )
        self.image_label.setPixmap(scaled)
        
        text_lines = ""
        for i, text in enumerate(self.last_result_text):
            if text[0][1] > 0.3:
                text_lines += f"{i+1}. {text[0][0]} : {text[0][1]:.2f}\n"
        self.info_text.setText(text_lines)
        
    def set_left_image(self, image_path):
        """Set image in the left panel (legacy method)"""
        image = cv2.imread(image_path)
        if image is None:
            return

        rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        h, w, ch = rgb.shape
        bytes_per_line = ch * w
        qimage = QImage(rgb.data, w, h, bytes_per_line, QImage.Format.Format_RGB888)
        pixmap = QPixmap.fromImage(qimage)

        scaled_pixmap = pixmap.scaled(
            self.left_image_label.size(),
            Qt.AspectRatioMode.KeepAspectRatio,
            Qt.TransformationMode.SmoothTransformation
        )
        self.left_image_label.setPixmap(scaled_pixmap)

    def ocr2image(self, org_image, boxes: list, rotated_crops: list, rec_results: list):
        """
        Convert OCR results to annotated image
        
        Args:
            org_image: Original input image
            boxes: Detection bounding boxes
            rotated_crops: Rotated crop regions
            rec_results: Recognition results
            
        Returns:
            numpy.ndarray: Annotated image with OCR results
        """
        from PIL import Image
        image = org_image[:, :, ::-1]
        ret_boxes = [line for line in boxes]
        txts = [line[0][0] for line in rec_results]
        scores = [line[0][1] for line in rec_results]
        bbox_text_poly_shape_quadruplets = []
        
        for i in range(len(ret_boxes)):
            bbox_text_poly_shape_quadruplets.append(
                ([np.array(ret_boxes[i]).flatten()], txts[i], image.shape, image.shape)
            )
        
        im_sample = draw_with_poly_enhanced(image, bbox_text_poly_shape_quadruplets)
        return np.array(im_sample)

    def display_image(self, file_path):
        """Display and process the selected image"""
        image = cv2.imread(file_path)
        if image is None:
            self.image_label.setText(f"Failed to load image file: {file_path}")
            return
        
        boxes, crops, rec_results = self.ocr_run(image, file_path)
        self.last_result_image = self.ocr2image(image, boxes, crops, rec_results)
        self.last_result_text = rec_results
        preview_image_q.append((copy.deepcopy(self.last_result_image), rec_results, file_path))
        self.update_display_from_cached()
        self.preview_grid.set_images(preview_image_q)
        
    def resizeEvent(self, event):
        """Handle window resize events"""
        if self.image_label.pixmap():
            if self.file_list.currentItem():
                # Get file path from item data instead of text
                item = self.file_list.currentItem()
                path = item.data(Qt.ItemDataRole.UserRole) if item else None
            self.update_display_from_cached()
            self.preview_grid.set_images(preview_image_q)
        super().resizeEvent(event) 