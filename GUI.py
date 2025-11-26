# GUI.py - Object Detection Desktop Application
"""
GUI untuk aplikasi Object Detection menggunakan PyQt5
Support: Gambar dan Video dengan berbagai enhancement options
"""

import sys
import cv2
import numpy as np
from pathlib import Path
from PyQt5.QtWidgets import (QApplication, QMainWindow, QWidget, QVBoxLayout, 
                             QHBoxLayout, QPushButton, QLabel, QComboBox, 
                             QSlider, QFileDialog, QGroupBox, QProgressBar,
                             QMessageBox, QScrollArea, QTextEdit, QRadioButton)
from PyQt5.QtCore import Qt, QThread, pyqtSignal
from PyQt5.QtGui import QPixmap, QImage, QPalette, QColor, QFont
import torch

# Import fungsi run dari detect.py
try:
    from detect import run
except ImportError:
    print("Error: detect.py tidak ditemukan. Pastikan file detect.py ada di direktori yang sama.")
    sys.exit(1)

class DetectionThread(QThread):
    """Thread terpisah untuk proses deteksi agar GUI tetap responsif"""
    finished = pyqtSignal(str)
    progress = pyqtSignal(int)
    error = pyqtSignal(str)
    log = pyqtSignal(str)
    
    def __init__(self, model_path, source, conf_thres, enhancement, device):
        super().__init__()
        self.model_path = model_path
        self.source = source
        self.conf_thres = conf_thres
        self.enhancement = enhancement
        self.device = device

    def run(self):
        try:
            self.log.emit(f"Memulai deteksi...")
            self.log.emit(f"Model: {Path(self.model_path).name}")
            self.log.emit(f"Source: {Path(self.source).name}")
            self.log.emit(f"Confidence: {self.conf_thres:.2f}")
            self.log.emit(f"Device: {self.device.upper()}")
            if self.enhancement:
                self.log.emit(f"Enhancement: {self.enhancement}")
            
            # Jalankan deteksi dengan parameter yang dipilih
            output_path = run(
                weights=self.model_path,
                source=self.source,
                conf_thres=self.conf_thres,
                enhancement=self.enhancement,
                device=self.device
            )

            self.log.emit(f"Deteksi selesai!")
            self.finished.emit(output_path)
        except Exception as e:
            self.log.emit(f"Error: {str(e)}")
            self.error.emit(str(e))

class ObjectDetectionApp(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Object Detection Application")
        self.setGeometry(100, 100, 1200, 800)
        
        # Path dan variabel
        self.model_dir = Path("model")
        self.input_path = None
        self.output_path = None
        self.detection_thread = None
        self.current_device = 'cpu'
        
        # Setup UI
        self.init_ui()
        self.apply_styles()
    
    def on_device_changed(self):
        if self.cpu_radio.isChecked():
            self.current_device = "cpu"
            self.add_log("Device: CPU")
        elif self.gpu_radio.isChecked():
            self.current_device = "cuda"
            try:
                if torch.cuda.is_available():
                    gpu_name = torch.cuda.get_device_name(0)
                else:
                    gpu_name = "(CUDA not available)"
            except Exception as e:
                gpu_name = f"(error: {e})"
            self.add_log(f"Device: GPU ({gpu_name})")

    def init_ui(self):
        """Inisialisasi semua komponen UI"""
        central_widget = QWidget()
        self.setCentralWidget(central_widget)
        main_layout = QHBoxLayout(central_widget)
        
        # Panel Kiri - Kontrol
        left_panel = self.create_control_panel()
        main_layout.addWidget(left_panel, 1)
        
        # Panel Kanan - Display
        right_panel = self.create_display_panel()
        main_layout.addWidget(right_panel, 2)

        self.on_device_changed()
        
    def create_control_panel(self):
        """Membuat panel kontrol di sebelah kiri"""
        panel = QWidget()
        layout = QVBoxLayout(panel)
        layout.setSpacing(15)
        
        # Header
        header = QLabel("Pengaturan Deteksi")
        header.setStyleSheet("font-size: 18px; font-weight: bold; padding: 10px;")
        layout.addWidget(header)
        
        # 1. Pilih Model
        model_group = QGroupBox("1. Pilih Model")
        model_layout = QVBoxLayout()
        
        self.model_combo = QComboBox()
        self.load_available_models()
        model_layout.addWidget(self.model_combo)
        
        model_group.setLayout(model_layout)
        layout.addWidget(model_group)
        
        # 2. Upload File
        upload_group = QGroupBox("2. Upload File")
        upload_layout = QVBoxLayout()
        
        self.upload_btn = QPushButton("📁 Pilih Gambar/Video")
        self.upload_btn.clicked.connect(self.upload_file)
        upload_layout.addWidget(self.upload_btn)
        
        self.file_label = QLabel("Belum ada file dipilih")
        self.file_label.setWordWrap(True)
        self.file_label.setStyleSheet("padding: 5px; color: #666;")
        upload_layout.addWidget(self.file_label)
        
        upload_group.setLayout(upload_layout)
        layout.addWidget(upload_group)
        
        # 3. Enhancement
        enhance_group = QGroupBox("3. Image Enhancement")
        enhance_layout = QVBoxLayout()
        
        self.enhance_combo = QComboBox()
        self.enhance_combo.addItems([
            "Tidak Ada Enhancement",
            "Histogram Equalization",
            "CLAHE (Adaptive Histogram)",
            "Brightness & Contrast",
            "Sharpening",
            "Denoising"
        ])
        enhance_layout.addWidget(self.enhance_combo)
        
        enhance_group.setLayout(enhance_layout)
        layout.addWidget(enhance_group)
        
        # 4. Confidence Threshold
        conf_group = QGroupBox("4. Confidence Threshold")
        conf_layout = QVBoxLayout()
        
        self.conf_slider = QSlider(Qt.Orientation.Horizontal)
        self.conf_slider.setMinimum(1)
        self.conf_slider.setMaximum(100)
        self.conf_slider.setValue(25)
        self.conf_slider.setTickPosition(QSlider.TicksBelow)
        self.conf_slider.setTickInterval(10)
        self.conf_slider.valueChanged.connect(self.update_conf_label)
        conf_layout.addWidget(self.conf_slider)
        
        self.conf_label = QLabel("Confidence: 0.25")
        self.conf_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        conf_layout.addWidget(self.conf_label)
        
        conf_group.setLayout(conf_layout)
        layout.addWidget(conf_group)

        # 5. Device Selection (GPU/CPU)
        device_group = QGroupBox("5. Pilih Device")
        device_layout = QVBoxLayout()

        self.cpu_radio = QRadioButton("🖥️ CPU")
        self.cpu_radio.setChecked(True)

        self.gpu_radio = QRadioButton("🚀 GPU (CUDA)")
        self.gpu_radio.setEnabled(torch.cuda.is_available())

        self.cpu_radio.toggled.connect(self.on_device_changed)
        self.gpu_radio.toggled.connect(self.on_device_changed)

        device_layout.addWidget(self.cpu_radio)
        device_layout.addWidget(self.gpu_radio)

        self.device_info_label = QLabel()
        if torch.cuda.is_available():
            try:
                gpu_name = torch.cuda.get_device_name(0)
                self.device_info_label.setText(f"GPU Terdeteksi: {gpu_name}")
            except Exception:
                self.device_info_label.setText("GPU terdeteksi (nama tidak tersedia)")
        else:
            self.device_info_label.setText("GPU tidak tersedia")

        device_layout.addWidget(self.device_info_label)
        device_group.setLayout(device_layout)

        layout.addWidget(device_group)

        # 6. Tombol Deteksi
        self.detect_btn = QPushButton("Mulai Deteksi")
        self.detect_btn.clicked.connect(self.start_detection)
        self.detect_btn.setStyleSheet("""
            QPushButton {
                background-color: #4CAF50;
                color: white;
                font-size: 16px;
                font-weight: bold;
                padding: 12px;
                border-radius: 5px;
            }
            QPushButton:hover {
                background-color: #45a049;
            }
            QPushButton:disabled {
                background-color: #cccccc;
            }
        """)
        layout.addWidget(self.detect_btn)
        
        # Progress Bar
        self.progress_bar = QProgressBar()
        self.progress_bar.setVisible(False)
        layout.addWidget(self.progress_bar)
        
        # 6. Tombol Download
        self.download_btn = QPushButton("💾 Download Hasil")
        self.download_btn.clicked.connect(self.download_result)
        self.download_btn.setEnabled(False)
        self.download_btn.setStyleSheet("""
            QPushButton {
                background-color: #2196F3;
                color: white;
                font-size: 14px;
                padding: 10px;
                border-radius: 5px;
            }
            QPushButton:hover {
                background-color: #0b7dda;
            }
            QPushButton:disabled {
                background-color: #cccccc;
            }
        """)
        layout.addWidget(self.download_btn)
        
        layout.addStretch()
        return panel
    
    def create_display_panel(self):
        """Membuat panel display di sebelah kanan"""
        panel = QWidget()
        layout = QVBoxLayout(panel)
        
        # Header
        header = QLabel("Hasil Deteksi")
        header.setStyleSheet("font-size: 18px; font-weight: bold; padding: 10px;")
        layout.addWidget(header)
        
        # Scroll Area untuk gambar
        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setStyleSheet("background-color: #f5f5f5; border: 2px solid #ddd;")
        
        self.image_label = QLabel()
        self.image_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.image_label.setText("\n\nHasil deteksi akan muncul di sini")
        self.image_label.setStyleSheet("font-size: 16px; color: #999; padding: 50px;")
        
        scroll.setWidget(self.image_label)
        layout.addWidget(scroll, 3)
        
        # Info hasil
        self.info_label = QLabel()
        self.info_label.setStyleSheet("padding: 10px; background-color: #e3f2fd; border-radius: 5px;")
        self.info_label.setVisible(False)
        layout.addWidget(self.info_label)
        
        # Log area
        log_label = QLabel("Log Proses")
        log_label.setStyleSheet("font-size: 14px; font-weight: bold; margin-top: 10px;")
        layout.addWidget(log_label)
        
        self.log_text = QTextEdit()
        self.log_text.setReadOnly(True)
        self.log_text.setMaximumHeight(150)
        self.log_text.setStyleSheet("""
            QTextEdit {
                background-color: #263238;
                color: #aed581;
                font-family: 'Courier New', monospace;
                font-size: 12px;
                border: 1px solid #37474f;
                border-radius: 4px;
                padding: 5px;
            }
        """)
        layout.addWidget(self.log_text, 1)
        
        return panel
    
    def load_available_models(self):
        """Memuat daftar model yang tersedia"""
        if self.model_dir.exists():
            models = list(self.model_dir.glob("*.pt"))
            if models:
                for model in models:
                    self.model_combo.addItem(model.name, str(model))
            else:
                self.model_combo.addItem("Tidak ada model ditemukan")
        else:
            self.model_combo.addItem("Folder model tidak ditemukan")
    
    def update_conf_label(self):
        """Update label confidence threshold"""
        value = self.conf_slider.value() / 100
        self.conf_label.setText(f"Confidence: {value:.2f}")
    
    def upload_file(self):
        """Upload gambar atau video"""
        file_filter = "Media Files (*.jpg *.jpeg *.png *.mp4 *.avi *.mov);;All Files (*.*)"
        file_path, _ = QFileDialog.getOpenFileName(
            self, "Pilih File", "", file_filter
        )
        
        if file_path:
            self.input_path = file_path
            file_name = Path(file_path).name
            self.file_label.setText(f"✓ {file_name}")
            self.file_label.setStyleSheet("padding: 5px; color: #4CAF50; font-weight: bold;")
            
            # Preview untuk gambar
            if file_path.lower().endswith(('.jpg', '.jpeg', '.png')):
                self.display_image(file_path)
    
    def display_image(self, image_path):
        """Menampilkan gambar di panel display"""
        pixmap = QPixmap(image_path)
        if not pixmap.isNull():
            scaled_pixmap = pixmap.scaled(
                self.image_label.size(), 
                Qt.AspectRatioMode.KeepAspectRatio, 
                Qt.TransformationMode.SmoothTransformation
            )
            self.image_label.setPixmap(scaled_pixmap)
    
    def get_enhancement_type(self):
        """Mendapatkan jenis enhancement yang dipilih"""
        enhancement_map = {
            "Tidak Ada Enhancement": None,
            "Histogram Equalization": "hist_eq",
            "CLAHE (Adaptive Histogram)": "clahe",
            "Brightness & Contrast": "brightness",
            "Sharpening": "sharpen",
            "Denoising": "denoise"
        }
        return enhancement_map[self.enhance_combo.currentText()]
    
    def start_detection(self):
        """Memulai proses deteksi"""
        if not self.input_path:
            QMessageBox.warning(self, "Peringatan", "Silakan pilih file terlebih dahulu!")
            return
        
        if self.model_combo.currentIndex() == -1:
            QMessageBox.warning(self, "Peringatan", "Silakan pilih model terlebih dahulu!")
            return
        
        # Clear log
        self.log_text.clear()
        self.add_log("=" * 50)
        self.add_log("MEMULAI PROSES DETEKSI")
        self.add_log("=" * 50)
        
        # Disable tombol dan tampilkan progress
        self.detect_btn.setEnabled(False)
        self.progress_bar.setVisible(True)
        self.progress_bar.setRange(0, 0)  # Indeterminate mode
        
        # Dapatkan parameter
        model_path = self.model_combo.currentData()
        conf_thres = self.conf_slider.value() / 100
        enhancement = self.get_enhancement_type()
        
        # Jalankan deteksi di thread terpisah
        self.detection_thread = DetectionThread(
            model_path, self.input_path, conf_thres, enhancement, self.current_device
        )
        self.detection_thread.finished.connect(self.on_detection_finished)
        self.detection_thread.error.connect(self.on_detection_error)
        self.detection_thread.log.connect(self.add_log)
        self.detection_thread.start()
    
    def add_log(self, message):
        """Tambahkan pesan ke log"""
        self.log_text.append(message)
        # Auto scroll ke bawah
        scrollbar = self.log_text.verticalScrollBar()
        if scrollbar:
            scrollbar.setValue(scrollbar.maximum())
    
    def on_detection_finished(self, output_path):
        """Callback ketika deteksi selesai"""
        self.output_path = output_path
        
        # Enable tombol kembali
        self.detect_btn.setEnabled(True)
        self.progress_bar.setVisible(False)
        self.download_btn.setEnabled(True)
        
        # Tampilkan hasil
        if output_path.lower().endswith(('.jpg', '.jpeg', '.png')):
            self.display_image(output_path)
            self.info_label.setText("✓ Deteksi berhasil! Gambar siap diunduh.")
        else:
            self.info_label.setText("✓ Deteksi berhasil! Video siap diunduh.")
        
        self.info_label.setVisible(True)
        
        QMessageBox.information(self, "Sukses", "Deteksi selesai!")
    
    def on_detection_error(self, error_msg):
        """Callback ketika terjadi error"""
        self.detect_btn.setEnabled(True)
        self.progress_bar.setVisible(False)
        QMessageBox.critical(self, "Error", f"Terjadi kesalahan:\n{error_msg}")
    
    def download_result(self):
        """Download hasil deteksi"""
        if not self.output_path:
            return
        
        file_ext = Path(self.output_path).suffix
        file_filter = f"File (*{file_ext})"
        
        save_path, _ = QFileDialog.getSaveFileName(
            self, "Simpan Hasil", f"result{file_ext}", file_filter
        )
        
        if save_path:
            try:
                import shutil
                shutil.copy2(self.output_path, save_path)
                QMessageBox.information(self, "Sukses", f"File berhasil disimpan ke:\n{save_path}")
            except Exception as e:
                QMessageBox.critical(self, "Error", f"Gagal menyimpan file:\n{str(e)}")
    
    def apply_styles(self):
        """Menerapkan style ke aplikasi"""
        self.setStyleSheet("""
            QMainWindow {
                background-color: #fafafa;
            }
            QGroupBox {
                font-weight: bold;
                border: 2px solid #ddd;
                border-radius: 5px;
                margin-top: 10px;
                padding-top: 10px;
            }
            QGroupBox::title {
                subcontrol-origin: margin;
                left: 10px;
                padding: 0 5px;
            }
            QComboBox, QPushButton {
                padding: 8px;
                border: 1px solid #ddd;
                border-radius: 4px;
            }
            QComboBox:hover, QPushButton:hover {
                border-color: #2196F3;
            }
        """)

def main():
    app = QApplication(sys.argv)
    win = ObjectDetectionApp()
    win.show()
    sys.exit(app.exec_())

if __name__ == "__main__":
    main()