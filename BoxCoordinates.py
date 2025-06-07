import sys
import os
import time
import io
import json
import threading
import traceback

# --- PySide6 Imports ---
from PySide6.QtWidgets import (QApplication, QMainWindow, QWidget, QVBoxLayout,
                               QHBoxLayout, QPushButton, QLabel, QLineEdit,
                               QFileDialog, QMessageBox, QGroupBox, QFormLayout,
                               QPlainTextEdit, QFrame, QGridLayout)
from PySide6.QtGui import QIcon, QPixmap, QImage, QClipboard # <-- QIcon is imported here
from PySide6.QtCore import Qt, QThread, QObject, Signal, Slot

# --- Core Processing Imports ---
import torch
try:
    from segment_anything import SamAutomaticMaskGenerator, sam_model_registry
except ImportError:
    try:
        app = QApplication(sys.argv)
        error_title = "SAM Lid Coordinate Extractor - Critical Error"
        error_message = "Required library 'segment_anything' is not installed.\nPlease install it by running:\npip install git+https://github.com/facebookresearch/segment-anything.git"
        QMessageBox.critical(None, error_title, error_message)
    except Exception as e:
        print(f"CRITICAL ERROR: 'segment_anything' is not installed. {e}")
    sys.exit(1)

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import requests
from PIL import Image
from PIL.ImageQt import ImageQt

# --- Configuration & Global Variables ---
APP_NAME = "Coordinates extractor using SAM"
MODEL_TYPE = "vit_b"
SAM_CHECKPOINT_FILENAME = "sam_vit_b_01ec64.pth"
ICON_FILENAME = "icon.ico"  # <-- Your icon file name
SAM_MODEL_DOWNLOAD_URL = f"https://dl.fbaipublicfiles.com/segment_anything/{SAM_CHECKPOINT_FILENAME}"
MAX_IMAGE_DIMENSION = 1280
EXPORT_DIR = "exports"

def get_base_path():
    """Gets the base path, whether running from a script or a frozen executable."""
    if getattr(sys, 'frozen', False) and hasattr(sys, '_MEIPASS'):
        return sys._MEIPASS
    return os.path.dirname(os.path.abspath(__file__))

BASE_PATH = get_base_path()
ICON_PATH = os.path.join(BASE_PATH, ICON_FILENAME) # <-- Full path to the icon
SAM_CHECKPOINT_PATH = os.path.join(BASE_PATH, SAM_CHECKPOINT_FILENAME)
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

DEFAULT_PARAMS = {
    "NUM_LIDS_TO_PROCESS": 6, "MIN_LID_AREA_FRACTION": 0.01, "MAX_LID_AREA_FRACTION": 0.30,
    "MIN_ASPECT_RATIO_FILTER": 0.5, "SAM_POINTS_PER_SIDE": 32, "SAM_MIN_MASK_REGION_AREA": 100
}
PARAM_EXPLANATIONS = {
    "NUM_LIDS_TO_PROCESS": "Target number of most prominent lids to identify.",
    "MIN_LID_AREA_FRACTION": "Min segment area (fraction of image area, 0.0-1.0). Increase to filter small items.",
    "MAX_LID_AREA_FRACTION": "Max segment area (fraction of image area, 0.0-1.0). Decrease to filter large items.",
    "MIN_ASPECT_RATIO_FILTER": "Min aspect ratio (short_side/long_side, 0.0-1.0). 1.0 is square. Increase for more square-like lids.",
    "SAM_POINTS_PER_SIDE": "SAM generator: Points per side for grid sampling. Higher is more detailed but slower (e.g., 16-100).",
    "SAM_MIN_MASK_REGION_AREA": "SAM generator: Discards raw mask regions smaller than this pixel area."
}

# --- Helper & Core Processing Functions (Unchanged Logic) ---
def calculate_aspect_ratio(bbox):
    x, y, w, h = bbox
    if h == 0 or w == 0: return 0
    return min(w / h, h / w)

def filter_and_sort_masks(masks, image_area_processed, num_lids_expected, min_area_frac, max_area_frac, min_aspect_ratio):
    lid_candidates = []
    for ann in masks:
        area = ann['area']
        bbox = ann['bbox']
        if not (min_area_frac * image_area_processed < area < max_area_frac * image_area_processed):
            continue
        aspect_ratio = calculate_aspect_ratio(bbox)
        if not (aspect_ratio >= min_aspect_ratio):
            continue
        lid_candidates.append(ann)
    lid_candidates.sort(key=lambda ann: (ann['bbox'][1], ann['bbox'][0]))
    if len(lid_candidates) > num_lids_expected:
        lid_candidates.sort(key=lambda ann: ann['area'], reverse=True)
        selected_candidates = lid_candidates[:num_lids_expected]
        selected_candidates.sort(key=lambda ann: (ann['bbox'][1], ann['bbox'][0]))
        lid_candidates = selected_candidates
    elif 0 < len(lid_candidates) < num_lids_expected :
        print(f"Warning: Found {len(lid_candidates)} candidates, less than targeted {num_lids_expected}.")
    return lid_candidates

def run_sam_processing(image_path, params, worker_signals):
    try:
        worker_signals.status_updated.emit(f"Using device: {DEVICE}", "#1C223A")
        worker_signals.status_updated.emit(f"Loading image: {os.path.basename(image_path)}...", "#1C223A")
        image_pil_original = Image.open(image_path).convert("RGB")
        original_width, original_height = image_pil_original.size
        worker_signals.status_updated.emit(f"Original image size: {original_width}x{original_height}", "#1C223A")
        scale_w, scale_h = 1.0, 1.0
        if original_width > MAX_IMAGE_DIMENSION or original_height > MAX_IMAGE_DIMENSION:
            worker_signals.status_updated.emit(f"Resizing to max {MAX_IMAGE_DIMENSION}px on longest side...", "#1C223A")
            if original_width > original_height:
                new_width = MAX_IMAGE_DIMENSION
                new_height = int(original_height * (MAX_IMAGE_DIMENSION / original_width))
            else:
                new_height = MAX_IMAGE_DIMENSION
                new_width = int(original_width * (MAX_IMAGE_DIMENSION / original_height))
            image_pil_for_sam = image_pil_original.resize((new_width, new_height), Image.Resampling.LANCZOS)
            scale_w = original_width / new_width
            scale_h = original_height / new_height
            worker_signals.status_updated.emit(f"Resized for SAM to {new_width}x{new_height}. Scale: W={scale_w:.2f}, H={scale_h:.2f}", "#1C223A")
        else:
            image_pil_for_sam = image_pil_original.copy()
        image_rgb_np_sam = np.array(image_pil_for_sam)
        image_area_sam = image_rgb_np_sam.shape[0] * image_rgb_np_sam.shape[1]
        worker_signals.status_updated.emit(f"Loading SAM model ({MODEL_TYPE})...", "#1C223A")
        if not os.path.exists(SAM_CHECKPOINT_PATH):
            worker_signals.error.emit(f"SAM checkpoint missing: {SAM_CHECKPOINT_PATH}.")
            return
        sam = sam_model_registry[MODEL_TYPE](checkpoint=SAM_CHECKPOINT_PATH)
        sam.to(device=DEVICE)
        sam.eval()
        mask_generator = SamAutomaticMaskGenerator(
            model=sam, points_per_side=params["SAM_POINTS_PER_SIDE"],
            pred_iou_thresh=params.get("SAM_PRED_IOU_THRESH", 0.88),
            stability_score_thresh=params.get("SAM_STABILITY_SCORE_THRESH", 0.95),
            min_mask_region_area=params["SAM_MIN_MASK_REGION_AREA"]
        )
        worker_signals.status_updated.emit("SAM model loaded.", "#1C223A")
        worker_signals.status_updated.emit("Generating all masks with SAM...", "#1C223A")
        all_masks_data = mask_generator.generate(image_rgb_np_sam)
        worker_signals.status_updated.emit(f"Generated {len(all_masks_data)} masks initially.", "#1C223A")
        if not all_masks_data:
            worker_signals.error.emit("No masks were generated by SAM.")
            return
        identified_lids_resized_coords = filter_and_sort_masks(
            all_masks_data, image_area_sam, params["NUM_LIDS_TO_PROCESS"],
            params["MIN_LID_AREA_FRACTION"], params["MAX_LID_AREA_FRACTION"],
            params["MIN_ASPECT_RATIO_FILTER"]
        )
        if not identified_lids_resized_coords:
            worker_signals.finished.emit([], None, params)
            worker_signals.error.emit("Could not identify lid candidates after filtering.")
            return
        worker_signals.status_updated.emit(f"Identified {len(identified_lids_resized_coords)} lid candidates.", "#1C223A")
        lid_coordinates_list_original_scale = []
        for i, ann_resized in enumerate(identified_lids_resized_coords):
            x_r, y_r, w_r, h_r = ann_resized['bbox']
            lid_coordinates_list_original_scale.append({
                "lid_index": i, "xmin": int(x_r * scale_w), "ymin": int(y_r * scale_h),
                "xmax": int((x_r + w_r) * scale_w), "ymax": int((y_r + h_r) * scale_h),
                "area": int(ann_resized['area'] * scale_w * scale_h),
                "predicted_iou": f"{ann_resized.get('predicted_iou', 0):.4f}"
            })
        fig, ax = plt.subplots(1, figsize=(10, 8), facecolor='#FDFBF5')
        ax.imshow(image_rgb_np_sam)
        ax.set_autoscale_on(False)
        ax.axis('off')
        for i, ann_resized in enumerate(identified_lids_resized_coords):
            m = ann_resized['segmentation']
            random_color = (np.random.random(), np.random.random(), np.random.random())
            color_overlay_viz = np.ones((m.shape[0], m.shape[1], 3))
            color_overlay_viz[:,:,0], color_overlay_viz[:,:,1], color_overlay_viz[:,:,2] = random_color
            ax.imshow(np.dstack((color_overlay_viz, m * 0.5)))
            x_bbox, y_bbox, w_bbox, h_bbox = ann_resized['bbox']
            rect = patches.Rectangle((x_bbox, y_bbox), w_bbox, h_bbox, linewidth=2, edgecolor=random_color, facecolor='none')
            ax.add_patch(rect)
            ax.text(x_bbox + w_bbox / 2, y_bbox + h_bbox / 2, str(i), color='white', fontsize=12, ha='center', va='center',
                    bbox=dict(facecolor=random_color, alpha=0.8, pad=1, edgecolor='none'))
        img_buffer = io.BytesIO()
        fig.savefig(img_buffer, format='PNG', bbox_inches='tight', pad_inches=0, dpi=120)
        plt.close(fig)
        img_buffer.seek(0)
        annotated_image_pil = Image.open(img_buffer)
        annotated_qimage = ImageQt(annotated_image_pil)
        worker_signals.finished.emit(lid_coordinates_list_original_scale, annotated_qimage, params)
    except Exception as e:
        worker_signals.error.emit(f"An error occurred: {e}\n{traceback.format_exc()}")

# --- PySide6 Worker for Threading ---
class WorkerSignals(QObject):
    finished = Signal(list, QImage, dict)
    error = Signal(str)
    status_updated = Signal(str, str)
    download_progress = Signal(str, str)
    download_finished = Signal(str, str)

class SamWorker(QObject):
    def __init__(self):
        super().__init__()
        self.signals = WorkerSignals()
    @Slot(str, dict)
    def run_processing(self, image_path, params):
        run_sam_processing(image_path, params, self.signals)
    @Slot()
    def download_model(self):
        try:
            self.signals.download_progress.emit("Downloading SAM checkpoint...", "#1C223A")
            response = requests.get(SAM_MODEL_DOWNLOAD_URL, stream=True, timeout=300)
            response.raise_for_status()
            total_size = int(response.headers.get('content-length', 0))
            bytes_downloaded = 0
            start_time = time.time()
            with open(SAM_CHECKPOINT_PATH, 'wb') as f:
                for chunk in response.iter_content(chunk_size=8192 * 16):
                    if chunk:
                        f.write(chunk)
                        bytes_downloaded += len(chunk)
                        if total_size > 0:
                            progress = (bytes_downloaded / total_size) * 100
                            elapsed_time = time.time() - start_time
                            speed = bytes_downloaded / elapsed_time / (1024*1024) if elapsed_time > 0 else 0
                            self.signals.download_progress.emit(f"Downloading... {progress:.1f}% ({speed:.2f} MB/s)", "#1C223A")
            self.signals.download_finished.emit("SAM Checkpoint downloaded. Ready.", "#2E7D32")
        except Exception as e:
            self.signals.download_finished.emit(f"Error downloading model: {e}", "#FF7900")
            if os.path.exists(SAM_CHECKPOINT_PATH):
                try: os.remove(SAM_CHECKPOINT_PATH)
                except Exception as rm_e: print(f"Could not remove partial download: {rm_e}")

# --- PySide6 GUI Application ---
class ModernSamApp(QMainWindow):
    start_processing_signal = Signal(str, dict)
    start_download_signal = Signal()

    def __init__(self):
        super().__init__()
        
        # --- SET THE APPLICATION ICON ---
        if os.path.exists(ICON_PATH):
            self.setWindowIcon(QIcon(ICON_PATH))
        else:
            print(f"Warning: Application icon not found at '{ICON_PATH}'")
        # --------------------------------

        self.setWindowTitle(APP_NAME)
        self.setGeometry(100, 100, 1000, 900)
        self.last_image_path = None
        self.central_widget = QWidget()
        self.setCentralWidget(self.central_widget)
        self.setStyleSheet(self.get_stylesheet())
        main_layout = QVBoxLayout(self.central_widget)
        main_layout.setContentsMargins(20, 20, 20, 20)
        main_layout.setSpacing(15)
        param_group = self._create_params_group()
        control_group = self._create_controls_group()
        main_content_layout = QHBoxLayout()
        main_content_layout.addWidget(self._create_image_panel(), 1)
        right_column_layout = QVBoxLayout()
        right_column_layout.addWidget(self._create_coordinate_output_panel())
        right_column_layout.addWidget(self._create_results_text_panel(), 1)
        main_content_layout.addLayout(right_column_layout)
        main_layout.addWidget(param_group)
        main_layout.addWidget(control_group)
        main_layout.addLayout(main_content_layout, 1)
        self._setup_worker_thread()
        self.check_sam_model()

    def get_stylesheet(self):
        return """
            QWidget {
                background-color: #E5D4B6;
                font-family: 'Fabio XM', 'Segoe UI', 'Roboto', 'Helvetica', 'Arial', sans-serif;
            }
            QGroupBox {
                color: #1C223A;
                font-size: 14px;
                font-weight: bold;
                border: 1px solid #c4b9a7;
                border-radius: 8px;
                padding: 8px 0px;
                background-color: transparent;
            }
            QGroupBox::title {
                subcontrol-origin: margin;
                subcontrol-position: top left;
                margin-left: 10px;
                margin-top: 10px;
                background-color: transparent;
            }
            QLabel {
                color: #1C223A;
                font-size: 13px;
                background-color: transparent;
            }
            QLabel#paramLabel {
                font-weight: bold;
            }
            QLabel#statusLabel {
                font-size: 14px;
                font-weight: normal;
                font-style: italic;
            }
            QLabel#imagePanelLabel {
                background-color: transparent;
                border: 1px solid #c4b9a7;
                border-radius: 8px;
            }
            QLineEdit {
                background-color: transparent;
                color: #1C223A;
                border: 1px solid #1C223A;
                border-radius: 4px;
                padding: 5px;
                font-size: 13px;
            }
            QLineEdit:focus {
                border: 2px solid #1C223A;
            }
            QPushButton {
                background-color: transparent;
                color: #1C223A;
                font-size: 14px;
                font-weight: bold;
                border: 1px solid #1C223A;
                border-radius: 5px;
                padding: 10px 20px;
                min-height: 25px;
            }
            QPushButton:hover {
                background-color: #1C223A;
                color: #E5D4B6;
            }
            QPushButton:disabled {
                background-color: #998c7d;
                color: #c4b9a7;
            }
            QPushButton#copyButton {
                font-size: 12px;
                padding: 5px 10px;
                min-height: 0;
            }
            QPlainTextEdit {
                background-color: transparent;
                color: #1C223A;
                border: none;
                font-family: 'Courier New', monospace;
                font-size: 12px;
            }
            QToolTip {
                background-color: #1C223A;
                color: #E5D4B6;
                border: none;
                padding: 5px;
                border-radius: 3px;
            }
        """

    def _create_params_group(self):
        group = QGroupBox("Tuning Parameters")
        layout = QGridLayout(group)
        layout.setSpacing(10)
        layout.setContentsMargins(15, 25, 15, 15)
        self.param_inputs = {}
        param_config = [
            ("NUM_LIDS_TO_PROCESS", "Num Lids Target:"), ("MIN_LID_AREA_FRACTION", "Min Area Fraction:"), ("MAX_LID_AREA_FRACTION", "Max Area Fraction:"),
            ("MIN_ASPECT_RATIO_FILTER", "Min Aspect Ratio:"), ("SAM_POINTS_PER_SIDE", "SAM Pts/Side:"), ("SAM_MIN_MASK_REGION_AREA", "SAM Min Mask Area:")
        ]
        num_items = len(param_config)
        items_per_row = (num_items + 1) // 2
        for i, (key, label_text) in enumerate(param_config):
            row = i // items_per_row
            col = (i % items_per_row) * 2
            label = QLabel(label_text)
            label.setObjectName("paramLabel")
            line_edit = QLineEdit(str(DEFAULT_PARAMS[key]))
            line_edit.setToolTip(PARAM_EXPLANATIONS[key])
            self.param_inputs[key] = line_edit
            layout.addWidget(label, row, col)
            layout.addWidget(line_edit, row, col + 1)
        return group

    def _create_controls_group(self):
        group = QWidget()
        layout = QVBoxLayout(group)
        layout.setContentsMargins(0,0,0,0)
        button_layout = QHBoxLayout()
        self.select_button = QPushButton("Select Image & Process")
        self.select_button.clicked.connect(self.select_image_and_process)
        self.retry_button = QPushButton("Retry with Current Params")
        self.retry_button.clicked.connect(self.retry_processing)
        self.retry_button.setEnabled(False)
        button_layout.addStretch()
        button_layout.addWidget(self.select_button)
        button_layout.addWidget(self.retry_button)
        button_layout.addStretch()
        self.status_label = QLabel("Status: Initializing...")
        self.status_label.setObjectName("statusLabel")
        self.status_label.setAlignment(Qt.AlignCenter)
        layout.addLayout(button_layout)
        layout.addWidget(self.status_label)
        return group

    def _create_image_panel(self):
        self.image_panel = QLabel("Annotated image will be displayed here.")
        self.image_panel.setObjectName("imagePanelLabel")
        self.image_panel.setAlignment(Qt.AlignCenter)
        self.image_panel.setMinimumSize(400, 300)
        return self.image_panel
    
    def _create_coordinate_output_panel(self):
        group = QGroupBox("Machine-Readable Coordinates")
        layout = QVBoxLayout(group)
        layout.setContentsMargins(10, 20, 10, 10)
        self.coordinate_text = QPlainTextEdit()
        self.coordinate_text.setReadOnly(True)
        self.coordinate_text.setPlaceholderText("xmin,ymin,xmax,ymax")
        copy_button = QPushButton("Copy")
        copy_button.setObjectName("copyButton")
        copy_button.clicked.connect(self.copy_coordinates)
        button_layout = QHBoxLayout()
        button_layout.addStretch()
        button_layout.addWidget(copy_button)
        layout.addWidget(self.coordinate_text)
        layout.addLayout(button_layout)
        return group

    def _create_results_text_panel(self):
        group = QGroupBox("Processing Log & Full Results")
        layout = QVBoxLayout(group)
        layout.setContentsMargins(10, 20, 10, 10)
        self.result_text = QPlainTextEdit()
        self.result_text.setReadOnly(True)
        self.result_text.setPlaceholderText("Lid coordinates and logs will appear here...")
        layout.addWidget(self.result_text)
        return group

    def _setup_worker_thread(self):
        self.worker_thread = QThread()
        self.sam_worker = SamWorker()
        self.sam_worker.moveToThread(self.worker_thread)
        self.sam_worker.signals.finished.connect(self.display_results)
        self.sam_worker.signals.error.connect(self.handle_error)
        self.sam_worker.signals.status_updated.connect(self.update_status)
        self.sam_worker.signals.download_progress.connect(self.update_status)
        self.sam_worker.signals.download_finished.connect(self.on_download_finished)
        self.start_processing_signal.connect(self.sam_worker.run_processing)
        self.start_download_signal.connect(self.sam_worker.download_model)
        self.worker_thread.start()

    def check_sam_model(self):
        if not os.path.exists(SAM_CHECKPOINT_PATH):
            self.update_status(f"SAM Checkpoint '{SAM_CHECKPOINT_FILENAME}' not found.", "#FF7900")
            reply = QMessageBox.question(self, "Download Model?",
                                         f"SAM checkpoint '{SAM_CHECKPOINT_FILENAME}' not found.\nDownload it now? (~350MB)",
                                         QMessageBox.Yes | QMessageBox.No, QMessageBox.Yes)
            if reply == QMessageBox.Yes:
                self.select_button.setEnabled(False)
                self.retry_button.setEnabled(False)
                self.start_download_signal.emit()
            else:
                self.update_status("SAM Checkpoint missing. Cannot proceed.", "#FF7900")
        else:
            self.update_status(f"SAM Checkpoint found ({DEVICE}). Ready.", "#2E7D32")

    @Slot(str, str)
    def update_status(self, message, color):
        self.status_label.setText(f"Status: {message}")
        self.status_label.setStyleSheet(f"color: {color}; font-style: italic; font-size: 14px; font-weight:normal;")

    @Slot(str, str)
    def on_download_finished(self, message, color):
        self.update_status(message, color)
        self.select_button.setEnabled(True)
        if self.last_image_path:
            self.retry_button.setEnabled(True)

    @Slot(list, QImage, dict)
    def display_results(self, coordinates, annotated_qimage, used_params):
        self.result_text.clear()
        self.coordinate_text.clear()
        self.result_text.appendPlainText("--- Parameters Used for this Run ---")
        self.result_text.appendPlainText(json.dumps(used_params, indent=2) + "\n")
        if coordinates:
            coord_lines = []
            for lid_info in coordinates:
                line = f"{lid_info['xmin']},{lid_info['ymin']},{lid_info['xmax']},{lid_info['ymax']}"
                coord_lines.append(line)
            self.coordinate_text.setPlainText("\n".join(coord_lines))
            self.result_text.appendPlainText("--- Identified Lid Coordinates (Original Image Scale) ---")
            for lid_info in coordinates:
                self.result_text.appendPlainText(f"Lid {lid_info['lid_index']}:")
                self.result_text.appendPlainText(f"  Top-Left (xmin, ymin): ({lid_info['xmin']}, {lid_info['ymin']})")
                self.result_text.appendPlainText(f"  Bottom-Right (xmax, ymax): ({lid_info['xmax']}, {lid_info['ymax']})")
                self.result_text.appendPlainText(f"  Area (original scale): {lid_info['area']} pixels")
                self.result_text.appendPlainText(f"  Predicted IoU (SAM): {lid_info['predicted_iou']}\n")
        else:
            self.result_text.appendPlainText("No lid coordinates found with the current parameters.")
        status_message_suffix = ""
        if annotated_qimage and self.last_image_path:
            pil_image_to_save = Image.fromqimage(annotated_qimage)
            os.makedirs(EXPORT_DIR, exist_ok=True)
            base, _ = os.path.splitext(os.path.basename(self.last_image_path))
            save_filename = f"{base}_annotated_{int(time.time())}.png"
            save_path = os.path.join(EXPORT_DIR, save_filename)
            try:
                pil_image_to_save.save(save_path)
                status_message_suffix = f" Annotated image saved to {save_path}"
            except Exception as e:
                status_message_suffix = " Error saving annotated image."
                print(f"Error saving annotated image: {e}")
        self.update_status(f"Processing complete.{status_message_suffix}", "#1C223A")
        if annotated_qimage:
            pixmap = QPixmap.fromImage(annotated_qimage)
            self.image_panel.setPixmap(pixmap.scaled(self.image_panel.size(),
                                                     Qt.KeepAspectRatio, Qt.SmoothTransformation))
        else:
            self.image_panel.setText("No annotation image generated.")
        self.select_button.setEnabled(True)
        if self.last_image_path: self.retry_button.setEnabled(True)

    @Slot(str)
    def handle_error(self, error_message):
        self.result_text.clear()
        self.coordinate_text.clear()
        self.result_text.appendPlainText(f"ERROR:\n{error_message}")
        self.update_status("Error during processing.", "#FF7900")
        QMessageBox.critical(self, "Processing Error", str(error_message).split('\n')[0])
        self.select_button.setEnabled(True)
        if self.last_image_path: self.retry_button.setEnabled(True)
        
    def copy_coordinates(self):
        clipboard = QApplication.clipboard()
        text_to_copy = self.coordinate_text.toPlainText()
        if text_to_copy:
            clipboard.setText(text_to_copy)
            self.update_status("Coordinates copied to clipboard!", "#5E2750")
        else:
            self.update_status("No coordinates to copy.", "#FF7900")

    def _get_current_params_from_gui(self):
        current_params = {}
        try:
            for key, widget in self.param_inputs.items():
                if key in ["NUM_LIDS_TO_PROCESS", "SAM_POINTS_PER_SIDE", "SAM_MIN_MASK_REGION_AREA"]:
                    current_params[key] = int(widget.text())
                else:
                    current_params[key] = float(widget.text())
            if not (0 < current_params["MIN_LID_AREA_FRACTION"] < current_params["MAX_LID_AREA_FRACTION"] < 1):
                QMessageBox.warning(self, "Input Error", "Area Fractions must satisfy: 0 < Min < Max < 1")
                return None
            if not (0 < current_params["MIN_ASPECT_RATIO_FILTER"] <= 1):
                QMessageBox.warning(self, "Input Error", "Min Aspect Ratio must be between 0 and 1")
                return None
            return current_params
        except ValueError:
            QMessageBox.critical(self, "Input Error", "Invalid number found in parameters.")
            return None
        
    def _start_processing(self, image_path):
        current_params = self._get_current_params_from_gui()
        if current_params is None: return
        self.select_button.setEnabled(False)
        self.retry_button.setEnabled(False)
        self.image_panel.setText("Processing... Please wait.")
        self.image_panel.setPixmap(QPixmap())
        self.result_text.clear()
        self.coordinate_text.clear()
        self.result_text.appendPlainText("Processing, please wait...\nThis can take some time, especially on CPU or for large images.\n")
        self.result_text.appendPlainText("Parameters for this run:")
        self.result_text.appendPlainText(json.dumps(current_params, indent=2) + "\n")
        self.update_status(f"Processing '{os.path.basename(image_path)}'...", "#1C223A")
        self.start_processing_signal.emit(image_path, current_params)

    def select_image_and_process(self):
        if not os.path.exists(SAM_CHECKPOINT_PATH):
            QMessageBox.critical(self, "Model Missing", f"SAM checkpoint '{SAM_CHECKPOINT_FILENAME}' not found.")
            self.check_sam_model()
            return
        file_path, _ = QFileDialog.getOpenFileName(self, "Select an Image File", "",
                                                     "Image Files (*.tif *.tiff *.jpg *.jpeg *.png *.bmp);;All Files (*)")
        if not file_path: return
        self.last_image_path = file_path
        self._start_processing(self.last_image_path)

    def retry_processing(self):
        if self.last_image_path is None or not os.path.exists(self.last_image_path):
            QMessageBox.warning(self, "File Not Found", "Please select an image first.")
            self.retry_button.setEnabled(False)
            return
        self._start_processing(self.last_image_path)

    def closeEvent(self, event):
        self.worker_thread.quit()
        self.worker_thread.wait()
        event.accept()

if __name__ == "__main__":
    if DEVICE.type == 'cuda':
        os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'expandable_segments:True'
    os.makedirs(EXPORT_DIR, exist_ok=True)
    app = QApplication(sys.argv)
    window = ModernSamApp()
    window.show()
    sys.exit(app.exec())