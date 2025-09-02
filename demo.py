import sys
import os

import argparse

from dx_engine import InferenceEngine as IE
from dx_engine import InferenceOption as IO

from PySide6.QtWidgets import (
    QApplication, QWidget, QHBoxLayout, QVBoxLayout,
    QLabel, QPushButton, QListWidget, QListWidgetItem, QFileDialog, QSizePolicy, QGridLayout, QScrollArea, QTextEdit, QTableWidget, QTableWidgetItem
)
from PySide6.QtGui import QPixmap, QIcon, QImage
from PySide6.QtCore import Qt, QSize, QEvent, QMimeData, Signal

import tqdm

import cv2
import numpy as np
import onnxruntime as ort
# from baidu import demo_pipeline as dpp

from engine.paddleocr import PaddleOcr
from engine.draw_utils import draw_with_poly_enhanced, draw_ocr

from collections import deque
import copy

# OCR result structure constants
# OCR results format: [[text, confidence_score], ...]
TEXT_INDEX = 0      # Index for recognized text in OCR result
SCORE_INDEX = 1     # Index for confidence score in OCR result
CONFIDENCE_THRESHOLD = 0.3  # Minimum confidence score to display

preview_image_q = deque(maxlen=20)

class ClickableLabel(QLabel):
    clicked = Signal(int)  # Send index when clicked

    def __init__(self, index, parent=None):
        super().__init__(parent)
        self.index = index

    def mousePressEvent(self, event):
        if event.button() == Qt.MouseButton.LeftButton:
            self.clicked.emit(self.index)

class PreviewGridWidget(QWidget):
    imageClicked = Signal(int)  # Signal to pass to external

    def __init__(self, parent=None):
        super().__init__(parent)
        self.grid_layout = QGridLayout()
        self.setLayout(self.grid_layout)
        self.image_labels = []

    def clear(self):
        for label in self.image_labels:
            self.grid_layout.removeWidget(label)
            label.deleteLater()
        self.image_labels.clear()

    def set_images(self, image_q:deque):
        self.clear()

        cols = 2  # Number of desired columns
        for i in range(len(image_q)):
            idx = len(image_q) - i - 1
            label = ClickableLabel(idx)
            label.setAlignment(Qt.AlignmentFlag.AlignCenter)
            label.setStyleSheet("border: 1px solid gray; background-color: white;")
            label.clicked.connect(self.imageClicked.emit)

            # Display if image is in numpy array format
            rgb = image_q[idx][0]
            h, w, ch = rgb.shape
            bytes_per_line = ch * w
            qimage = QImage(rgb.data, w, h, bytes_per_line, QImage.Format.Format_RGB888)
            pixmap = QPixmap.fromImage(qimage)

            # Scale to fit label size
            pixmap = pixmap.scaled(100, 100, Qt.AspectRatioMode.KeepAspectRatio, Qt.TransformationMode.SmoothTransformation)
            label.setPixmap(pixmap)

            row = i // cols
            col = i % cols
            self.grid_layout.addWidget(label, row, col)
            self.image_labels.append(label)


class ThumbnailListWidget(QListWidget):
    def __init__(self, image_click_callback, thumbnail_size=60):
        super().__init__()
        self.setViewMode(QListWidget.ViewMode.IconMode)
        # Set smaller fixed square size for thumbnails
        self.setIconSize(QSize(thumbnail_size, thumbnail_size))
        self.setResizeMode(QListWidget.ResizeMode.Adjust)
        self.setAcceptDrops(True)
        self.setDragEnabled(True)
        # Reduce spacing for more compact layout
        self.setSpacing(5)
        self.image_click_callback = image_click_callback
        self.itemClicked.connect(self.handle_item_click)
        self.thumbnail_size = thumbnail_size

    def dragEnterEvent(self, event):
        if event.mimeData().hasUrls():
            event.acceptProposedAction()
        else:
            event.ignore()

    def dragMoveEvent(self, event):
        event.acceptProposedAction()

    def dropEvent(self, event):
        if event.mimeData().hasUrls():
            for url in event.mimeData().urls():
                file_path = url.toLocalFile()
                if os.path.isfile(file_path) and self.is_image(file_path):
                    self.add_thumbnail(file_path)
        event.acceptProposedAction()

    def is_image(self, path):
        return path.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp', '.gif'))

    def truncate_filename(self, filename, max_length):
        """
        Truncate filename to fit within thumbnail width
        """
        if len(filename) <= max_length:
            return filename
        
        # Calculate available space for filename (considering padding and ellipsis)
        available_chars = max_length - 3  # Reserve 3 characters for "..."
        
        if available_chars <= 0:
            return "..."
        
        # Split filename and extension
        name, ext = os.path.splitext(filename)
        
        # If extension is too long, truncate it
        if len(ext) > available_chars // 2:
            ext = ext[:available_chars // 2]
        
        # Calculate remaining space for name
        name_chars = available_chars - len(ext)
        
        if name_chars <= 0:
            return "..." + ext
        
        # Truncate name and add ellipsis
        truncated_name = name[:name_chars] + "..."
        return truncated_name + ext

    def add_thumbnail(self, file_path):
        pixmap = QPixmap(file_path)
        if not pixmap.isNull():
            # Create square thumbnail with fixed size
            icon = QIcon(pixmap.scaled(
                self.thumbnail_size, 
                self.thumbnail_size, 
                Qt.AspectRatioMode.KeepAspectRatio, 
                Qt.TransformationMode.SmoothTransformation
            ))
            
            # Get filename and truncate it to fit thumbnail width
            filename = os.path.basename(file_path)
            # Estimate max characters that can fit in thumbnail width
            # Assuming average character width of 6-8 pixels and some padding
            max_chars = max(8, self.thumbnail_size // 8)
            truncated_filename = self.truncate_filename(filename, max_chars)
            
            # Show truncated filename
            item = QListWidgetItem(icon, truncated_filename)
            item.setData(Qt.ItemDataRole.UserRole, file_path)
            # Store full filename as tooltip for hover display
            item.setToolTip(filename)
            self.addItem(item)

    def handle_item_click(self, item):
        file_path = item.data(Qt.ItemDataRole.UserRole)
        self.image_click_callback(file_path)


class PerformanceInfoWidget(QWidget):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.init_ui()
        
    def init_ui(self):
        layout = QVBoxLayout()
        
        # Title
        title = QLabel("Performance Information (FPS)")
        title.setStyleSheet("""
            QLabel {
                font-size: 13px;
                color: black;
                padding: 2px;
                border-radius: 2px;
                margin-bottom: 2px;
            }
        """)
        layout.addWidget(title)
        
        # Performance table
        self.performance_table = QTableWidget()
        self.performance_table.setRowCount(4)  # 4 rows including header
        self.performance_table.setColumnCount(3)
        # self.performance_table.setMaximumHeight(100)  # Slightly increased for header
        self.performance_table.setMinimumHeight(80)   # Increased minimum height
        
        # Set table headers
        self.performance_table.setHorizontalHeaderLabels(["", "GPU", "NPU"])
        self.performance_table.setVerticalHeaderLabels(["", "det", "cls", "rec"])
        
        # Hide row and column headers for cleaner look
        self.performance_table.horizontalHeader().setVisible(False)
        self.performance_table.verticalHeader().setVisible(False)
        
        # Set row heights to minimize spacing
        self.performance_table.verticalHeader().setDefaultSectionSize(20)  # Compact row height
        
        # Set table style with minimal padding
        self.performance_table.setStyleSheet("""
            QTableWidget {
                font-family: 'Courier New', monospace;
                font-size: 12px;
                background-color: #f8f9fa;
                border: 1px solid #dee2e6;
                border-radius: 3px;
                gridline-color: #dee2e6;
                padding: 0px;
                margin: 0px;
            }
            QTableWidget::item {
                padding: 1px 2px;
                border: none;
                margin: 0px;
            }
            QHeaderView::section {
                background-color: #f8f9fa;
                border: none;
                padding: 1px;
                margin: 0px;
            }
        """)
        
        # Set column widths
        self.performance_table.setColumnWidth(0, 40)  # First column for labels
        self.performance_table.setColumnWidth(1, 60)  # GPU column
        self.performance_table.setColumnWidth(2, 200)  # NPU column
        
        # Set initial performance data
        self.update_performance_data()
        
        layout.addWidget(self.performance_table)

        self.setLayout(layout)
    
    def update_performance_data(self, det_gpu=10, det_npu=10.548, cls_gpu=95.3, cls_npu=0.19, rec_gpu=121, rec_npu=0.736, min_rec_npu=0.736):
        """
        Update performance information display
        """
        # Create header row
        header_row = [
            QTableWidgetItem(""),
            QTableWidgetItem("GPU"),
            QTableWidgetItem("NPU")
        ]
        
        # Create table items
        det_row = [
            QTableWidgetItem("det"),
            QTableWidgetItem(f"{det_gpu:.2f}"),
            QTableWidgetItem(f"{1000/(det_npu + 1e-10):.2f}")
        ]
        
        cls_row = [
            QTableWidgetItem("cls"),
            QTableWidgetItem(f"{cls_gpu:.2f}"),
            QTableWidgetItem(f"{1000/(cls_npu + 1e-10):.2f}")
        ]
        
        rec_row = [
            QTableWidgetItem("rec"),
            QTableWidgetItem(f"{rec_gpu:.2f}"),
            QTableWidgetItem(f"{1000/(rec_npu + 1e-10):.2f}(avg), {1000/(min_rec_npu + 1e-10):.2f}(max)")
        ]
        
        # Set items in table
        for col, item in enumerate(header_row):
            self.performance_table.setItem(0, col, item)
        for col, item in enumerate(det_row):
            self.performance_table.setItem(1, col, item)
        for col, item in enumerate(cls_row):
            self.performance_table.setItem(2, col, item)
        for col, item in enumerate(rec_row):
            self.performance_table.setItem(3, col, item)


class EnvironmentInfoWidget(QWidget):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.init_ui()
        
    def init_ui(self):
        layout = QVBoxLayout()
        
        # Environment table
        self.environment_table = QTableWidget()
        self.environment_table.setRowCount(3)  # 3 rows including header
        self.environment_table.setColumnCount(2)
        # self.accuracy_table.setMaximumHeight(100)  # Slightly increased for header
        self.environment_table.setMinimumHeight(80)   # Increased minimum height
        
        # Hide row and column headers for cleaner look
        self.environment_table.horizontalHeader().setVisible(False)
        self.environment_table.verticalHeader().setVisible(False)
        # Set row heights to minimize spacing
        self.environment_table.verticalHeader().setDefaultSectionSize(20)  # Compact row height
        
        # Set table style with minimal padding
        self.environment_table.setStyleSheet("""
            QTableWidget {
                font-family: 'Courier New', monospace;
                font-size: 12px;
                background-color: #f8f9fa;
                border: 1px solid #dee2e6;
                border-radius: 3px;
                gridline-color: #dee2e6;
                padding: 0px;
                margin: 0px;
            }
            QTableWidget::item {
                padding: 1px 2px;
                border: none;
                margin: 0px;
            }
            QTableWidget::item:first {
                background-color: #e9ecef;
                font-weight: bold;
                border-bottom: 1px solid #dee2e6;
                font-size: 13px;
            }
            QHeaderView::section {
                background-color: #f8f9fa;
                border: none;
                padding: 1px;
                margin: 0px;
            }
        """)
        
        # Set column widths
        self.environment_table.setColumnWidth(0, 100)
        self.environment_table.setColumnWidth(1, 240)
        
        # Set initial accuracy data
        self.set_environment_data()
        
        layout.addWidget(self.environment_table)
        self.setLayout(layout)

    def set_environment_data(self):
        """
        Update environment information display
        """
        # Create merged header row
        header_item = QTableWidgetItem("Environment Information")
        header_item.setTextAlignment(Qt.AlignmentFlag.AlignCenter)
        self.environment_table.setItem(0, 0, header_item)
        # Merge the first row across all columns
        self.environment_table.setSpan(0, 0, 1, 2)
        
        # Create table items
        gpuInfo_row = [
            QTableWidgetItem("GPU Device"),
            QTableWidgetItem("NVIDIA GeForce RTX 2080 Ti")
        ]
        
        npuInfo_row = [
            QTableWidgetItem("NPU Device"),
            QTableWidgetItem("DEEPX DX-M1")
        ]
        
        # Set items in table (starting from row 1 since row 0 is merged header)

        for col, item in enumerate(gpuInfo_row):
            self.environment_table.setItem(1, col, item)
        for col, item in enumerate(npuInfo_row):
            self.environment_table.setItem(2, col, item)


class AccuracyInfoWidget(QWidget):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.init_ui()
        
    def init_ui(self):
        layout = QVBoxLayout()
        
        # Title
        title = QLabel("Accuracy Information")
        title.setStyleSheet("""
            QLabel {
                font-size: 13px;
                color: black;
                padding: 0px;
                border-radius: 0px;
                margin-bottom: 0px;
            }
        """)
        layout.addWidget(title)
        
        # Performance table
        self.accuracy_table = QTableWidget()
        self.accuracy_table.setRowCount(4)  # 4 rows including header
        self.accuracy_table.setColumnCount(4)
        # self.accuracy_table.setMaximumHeight(100)  # Slightly increased for header
        self.accuracy_table.setMinimumHeight(80)   # Increased minimum height
        
        # Set table headers
        
        self.accuracy_table.setHorizontalHeaderLabels(["", "GPU", "DX-M1", "GAP"])
        self.accuracy_table.setVerticalHeaderLabels(["", "", "V4", "V5"])
        
        # Hide row and column headers for cleaner look
        self.accuracy_table.horizontalHeader().setVisible(False)
        self.accuracy_table.verticalHeader().setVisible(False)
        
        # Set row heights to minimize spacing
        self.accuracy_table.verticalHeader().setDefaultSectionSize(20)  # Compact row height
        
        # Set table style with minimal padding
        self.accuracy_table.setStyleSheet("""
            QTableWidget {
                font-family: 'Courier New', monospace;
                font-size: 12px;
                background-color: #f8f9fa;
                border: 1px solid #dee2e6;
                border-radius: 3px;
                gridline-color: #dee2e6;
                padding: 0px;
                margin: 0px;
            }
            QTableWidget::item {
                padding: 1px 2px;
                border: none;
                margin: 0px;
            }
            QTableWidget::item:first {
                background-color: #e9ecef;
                font-weight: bold;
                border-bottom: 1px solid #dee2e6;
                font-size: 13px;
            }
            QHeaderView::section {
                background-color: #f8f9fa;
                border: none;
                padding: 1px;
                margin: 0px;
            }
        """)
        
        # Set column widths
        self.accuracy_table.setColumnWidth(0, 40)  # First column for labels
        self.accuracy_table.setColumnWidth(1, 80)  # GPU column
        self.accuracy_table.setColumnWidth(2, 80)  # NPU column
        self.accuracy_table.setColumnWidth(3, 80)  # GAP column
        
        # Set initial accuracy data
        self.set_accuracy_data()
        
        layout.addWidget(self.accuracy_table)
        self.setLayout(layout)

    def set_accuracy_data(self):
        """
        Update accuracy information display
        """
        # Create merged header row
        header_item = QTableWidgetItem("Accuracy (Error Rate %)")
        header_item.setTextAlignment(Qt.AlignmentFlag.AlignCenter)
        self.accuracy_table.setItem(0, 0, header_item)
        # Merge the first row across all columns
        self.accuracy_table.setSpan(0, 0, 1, 4)
        
        # Create table items
        header_row = [
            QTableWidgetItem(""),
            QTableWidgetItem("GPU"),
            QTableWidgetItem("DX-M1"),
            QTableWidgetItem("GAP")
        ]
        
        v4_row = [
            QTableWidgetItem("V4"),
            QTableWidgetItem("6.70 %"),
            QTableWidgetItem("7.10 %"),
            QTableWidgetItem("-0.40 %")
        ]
        
        v5_row = [
            QTableWidgetItem("V5"),
            QTableWidgetItem("5.60 %"),
            QTableWidgetItem("6.00 %"),
            QTableWidgetItem("-0.40 %")
        ]
        
        # Set items in table (starting from row 1 since row 0 is merged header)
        for col, item in enumerate(header_row):
            self.accuracy_table.setItem(1, col, item)
        for col, item in enumerate(v4_row):
            self.accuracy_table.setItem(2, col, item)
        for col, item in enumerate(v5_row):
            self.accuracy_table.setItem(3, col, item)


class ImageViewerApp(QWidget):
    def __init__(self, ocr_workers=None, app_version=None):
        super().__init__()
        self.setWindowTitle(f"PySide Image Viewer {app_version}")
        
        # Set window to full screen size
        screen_size = QApplication.primaryScreen().geometry()
        self.setGeometry(100, 100, screen_size.width()/1.2, screen_size.height()/1.2)
        
        # Store OCR workers
        self.ocr_workers = ocr_workers if ocr_workers else []
        self.current_worker_index = 0  # Index for round-robin worker selection
        self.app_version = app_version
        # Initialize UI first
        self.init_ui()
        
        # Store screen dimensions for reference
        self.max_width = screen_size.width()/2
        self.max_height = screen_size.height()/2
        self.last_result_image = None
        self.last_result_text = None

    def init_ui(self):
        main_layout = QHBoxLayout()
        left_layout = QVBoxLayout()

        # Swap positions: Text first, then Preview
        info_title = QLabel("Inference Result Text")
        info_title.setStyleSheet("""
            QLabel {
                font-size: 13px;
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
        # Set line spacing to increase readability
        self.info_text.setStyleSheet("""
            QTextEdit {
                line-height: 1.2;
                padding: 4px;
                font-family: 'Consolas', 'Monaco', 'Courier New', monospace;
                font-size: 12px;
            }
        """)

        text_scroll_area = QScrollArea()
        text_scroll_area.setWidgetResizable(True)
        text_scroll_area.setWidget(self.info_text)

        left_layout.addWidget(text_scroll_area, stretch=3)

        preview_title = QLabel("Processing Result Preview")
        preview_title.setStyleSheet("""
            QLabel {
                font-size: 13px;
                color: black;
                padding: 2px;
                border-radius: 2px;
                margin-bottom: 2px;
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

        main_image_title = QLabel(f"Main Image Viewer ({self.app_version})")
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
        right_layout.addWidget(self.performance_widget, stretch=2)
        # Add Accuracy Information widget before File Management
        self.envInfo_widget = EnvironmentInfoWidget()
        right_layout.addWidget(self.envInfo_widget, stretch=1)
        # Add Accuracy Information widget before File Management
        self.accuracy_widget = AccuracyInfoWidget()
        right_layout.addWidget(self.accuracy_widget, stretch=2)

        file_title = QLabel("File Management")
        file_title.setStyleSheet("""
            QLabel {
                font-size: 13px;
                color: black;
                padding: 2px;
                border-radius: 2px;
                margin-bottom: 2px;
            }
        """)
        right_layout.addWidget(file_title)

        # Use ThumbnailListWidget with smaller thumbnail size for right layout
        self.file_list = ThumbnailListWidget(self.handle_file_list_click, thumbnail_size=50)
        right_layout.addWidget(self.file_list, stretch=8)

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
        files, _ = QFileDialog.getOpenFileNames(
            self, "Select Image Files", "", "Images (*.png *.jpg *.jpeg *.bmp *.gif)"
        )
        for file_path in files:
            # Use add_thumbnail method for better visual representation
            self.file_list.add_thumbnail(file_path)

    def handle_file_list_click(self, file_path):
        # Updated to handle file_path directly from ThumbnailListWidget
        self.display_image(file_path)
    
    def handle_preview_click(self, index: int):
        image, text, _ = preview_image_q[index]  # Extract as numpy array
        self.last_result_image = image           # Can be displayed directly or overwritten with inference results
        self.last_result_text = text
        self.update_display_from_cached()

    def ocr_run(self, image, file_path):
        worker = self.get_next_worker()
        if worker is None:
            print("No OCR workers available")
            return [], [], []
        
        try:
            boxes, rotated_crops, rec_results = worker(image)
            self.performance_widget.update_performance_data(
                det_npu=worker.detection_time_duration / worker.ocr_run_count,
                cls_npu=worker.classification_time_duration / worker.ocr_run_count,
                rec_npu=worker.recognition_time_duration / worker.ocr_run_count,
                min_rec_npu=worker.min_recognition_time_duration / worker.ocr_run_count
            )
            return boxes, rotated_crops, rec_results
        except Exception as e:
            print(f"Error during OCR inference: {e}")
            return [], [], []
    
    def run_imagelist(self):
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
        text_lines += "   recognized text : confidence score\n\n"
        for i, text in enumerate(self.last_result_text):
            # Extract recognized text and confidence score from OCR result
            recognized_text = text[0][TEXT_INDEX]
            confidence_score = text[0][SCORE_INDEX]
            if confidence_score > CONFIDENCE_THRESHOLD:
                text_lines += f"{i+1}. {recognized_text} : {confidence_score:.2f}\n"
        self.info_text.setText(text_lines)
        
    def set_left_image(self, image_path):
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

    def ocr2image(self, org_image, boxes:list, rotated_crops:list, rec_results:list):
        from PIL import Image
        image = org_image[:, :, ::-1]
        ret_boxes = [line for line in boxes]
        # Extract recognized text and confidence scores from OCR results
        # rec_results format: [[text, confidence_score], ...]
        txts = [line[0][TEXT_INDEX] for line in rec_results]  # recognized text
        scores = [line[0][SCORE_INDEX] for line in rec_results]  # confidence scores
        bbox_text_poly_shape_quadruplets = []
        for i in range(len(ret_boxes)):
            bbox_text_poly_shape_quadruplets.append(
                ([np.array(ret_boxes[i]).flatten()], txts[i], image.shape, image.shape)
            )
        im_sample = draw_with_poly_enhanced(image, bbox_text_poly_shape_quadruplets)
        return np.array(im_sample)

    def display_image(self, file_path):
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
        if self.image_label.pixmap():
            if self.file_list.currentItem():
                # Get file path from item data instead of text
                item = self.file_list.currentItem()
                path = item.data(Qt.ItemDataRole.UserRole) if item else None
            self.update_display_from_cached()
            self.preview_grid.set_images(preview_image_q)
        super().resizeEvent(event)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--version', default='v4', choices=['v4', 'v5'])
    
    args = parser.parse_args()
    
    '''
    @brief:
    @param:
        det_model_path: str, the path of the detection model : det.dxnn
        cls_model_path: str, the path of the classification model : cls.dxnn
        rec_model_path: str, the path of the recognition model
                                rec_ratio_5_height_10.dxnn
                                rec_ratio_25_height_30.dxnn
                                rec_ratio_25_height_20.dxnn
                                rec_ratio_25_height_10.dxnn
                                rec_ratio_5_height_30.dxnn
                                rec_ratio_15_height_30.dxnn
                                rec_ratio_15_height_10.dxnn
                                rec_ratio_5_height_20.dxnn
                                rec_ratio_15_height_20.dxnn
    @return:
        None
    '''
    if args.version == 'v4':
        det_model_path = "engine/model_files/v4/det.dxnn"
        cls_model_path = "engine/model_files/v4/cls.dxnn"
        rec_model_dirname = "engine/model_files/v4/"
    elif args.version == 'v5':
        det_model_path = "engine/model_files/v5/det_v5.dxnn"
        cls_model_path = "engine/model_files/v5/cls_v5.dxnn"
        rec_model_dirname = "engine/model_files/v5/"
    
    det_model = IE(det_model_path, IO().set_use_ort(True))
    cls_model = IE(cls_model_path, IO().set_use_ort(True))
        
    def make_rec_engines(model_dirname):
        prefix = ''
        if model_dirname.find('v5') != -1:
            prefix = '_v5'
        io = IO().set_use_ort(True)
        rec_model_map = {}
        ratio_interval = 10
        max_ratio = 30

        for i in range(ratio_interval // 2, max_ratio, ratio_interval):
            rec_model_map[i] = {}
            for height in [10, 20, 30]:
                model_path = f"{model_dirname}/rec{prefix}_ratio_{i}_height_{height}.dxnn"
                rec_model_map[i][height] = IE(model_path, io)
        return rec_model_map

    rec_models:dict = make_rec_engines(rec_model_dirname)
    ocr_workers = [PaddleOcr(det_model, cls_model, rec_models, args.version) for _ in range(3)]
    
    app = QApplication(sys.argv)
    viewer = ImageViewerApp(ocr_workers=ocr_workers, app_version=args.version)
    viewer.show()
    sys.exit(app.exec())