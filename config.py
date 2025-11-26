"""
Konfigurasi dan utility functions untuk aplikasi Object Detection
"""

from pathlib import Path
import json
from datetime import datetime

# =====================
# PATH CONFIGURATIONS
# =====================
BASE_DIR = Path(__file__).parent
MODEL_DIR = BASE_DIR / 'model'
OUTPUT_DIR = BASE_DIR / 'runs' / 'detect'
CONFIG_FILE = BASE_DIR / 'app_config.json'

# =====================
# DEFAULT SETTINGS
# =====================
DEFAULT_SETTINGS = {
    'default_model': '5m.pt',
    'default_confidence': 0.25,
    'default_enhancement': None,
    'save_settings': True,
    'auto_open_result': False,
    'theme': 'light'
}

# =====================
# ENHANCEMENT CONFIGS
# =====================
ENHANCEMENT_CONFIGS = {
    'hist_eq': {
        'name': 'Histogram Equalization',
        'description': 'Meningkatkan kontras gambar secara global',
        'best_for': 'Gambar dengan pencahayaan tidak merata'
    },
    'clahe': {
        'name': 'CLAHE',
        'description': 'Adaptive histogram equalization',
        'best_for': 'Gambar dengan detail yang kurang jelas',
        'clip_limit': 2.0,
        'tile_grid_size': (8, 8)
    },
    'brightness': {
        'name': 'Brightness & Contrast',
        'description': 'Menyesuaikan kecerahan dan kontras',
        'best_for': 'Gambar yang terlalu gelap atau terang',
        'alpha': 1.2,  # Contrast
        'beta': 20     # Brightness
    },
    'sharpen': {
        'name': 'Sharpening',
        'description': 'Mempertajam tepi dan detail',
        'best_for': 'Gambar yang blur atau kurang tajam'
    },
    'denoise': {
        'name': 'Denoising',
        'description': 'Mengurangi noise pada gambar',
        'best_for': 'Gambar dengan banyak noise atau grain',
        'h': 10,
        'template_window_size': 7,
        'search_window_size': 21
    }
}

# =====================
# SUPPORTED FORMATS
# =====================
SUPPORTED_IMAGE_FORMATS = ['.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.webp']
SUPPORTED_VIDEO_FORMATS = ['.mp4', '.avi', '.mov', '.mkv', '.flv', '.wmv']

# =====================
# MODEL CONFIGURATIONS
# =====================
MODEL_INFO = {
    '5n.pt': {
        'name': 'YOLOv5 Nano',
        'size': '1.9 MB',
        'speed': 'Very Fast',
        'accuracy': 'Low-Medium',
        'description': 'Model paling ringan, cocok untuk real-time'
    },
    '5s.pt': {
        'name': 'YOLOv5 Small',
        'size': '7.2 MB',
        'speed': 'Fast',
        'accuracy': 'Medium',
        'description': 'Balance antara kecepatan dan akurasi'
    },
    '5m.pt': {
        'name': 'YOLOv5 Medium',
        'size': '21.2 MB',
        'speed': 'Medium',
        'accuracy': 'Medium-High',
        'description': 'Pilihan terbaik untuk kebanyakan kasus'
    },
    '5l.pt': {
        'name': 'YOLOv5 Large',
        'size': '46.5 MB',
        'speed': 'Slow',
        'accuracy': 'High',
        'description': 'Akurasi tinggi, butuh resource lebih besar'
    },
    '11m.pt': {
        'name': 'YOLOv11 Medium',
        'size': '~20 MB',
        'speed': 'Medium',
        'accuracy': 'High',
        'description': 'Model YOLOv11 versi medium'
    },
    '11n.pt': {
        'name': 'YOLOv11 Nano',
        'size': '~3 MB',
        'speed': 'Very Fast',
        'accuracy': 'Medium',
        'description': 'Model YOLOv11 versi nano'
    },
    '11s.pt': {
        'name': 'YOLOv11 Small',
        'size': '~10 MB',
        'speed': 'Fast',
        'accuracy': 'Medium-High',
        'description': 'Model YOLOv11 versi small'
    }
}

# =====================
# UTILITY FUNCTIONS
# =====================

def load_config():
    """Load konfigurasi dari file"""
    if CONFIG_FILE.exists():
        try:
            with open(CONFIG_FILE, 'r') as f:
                config = json.load(f)
                # Merge dengan default settings
                return {**DEFAULT_SETTINGS, **config}
        except Exception as e:
            print(f"Warning: Tidak bisa load config: {e}")
    return DEFAULT_SETTINGS.copy()

def save_config(config):
    """Simpan konfigurasi ke file"""
    try:
        with open(CONFIG_FILE, 'w') as f:
            json.dump(config, f, indent=4)
        return True
    except Exception as e:
        print(f"Warning: Tidak bisa save config: {e}")
        return False

def get_available_models():
    """Dapatkan list model yang tersedia"""
    if not MODEL_DIR.exists():
        MODEL_DIR.mkdir(parents=True, exist_ok=True)
        return []
    
    models = []
    for model_file in MODEL_DIR.glob('*.pt'):
        model_name = model_file.name
        info = MODEL_INFO.get(model_name, {
            'name': model_name,
            'size': 'Unknown',
            'speed': 'Unknown',
            'accuracy': 'Unknown',
            'description': 'Custom model'
        })
        models.append({
            'filename': model_name,
            'path': str(model_file),
            'info': info
        })
    
    return models

def create_output_filename(source_path, prefix='result'):
    """Generate output filename dengan timestamp"""
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    source_name = Path(source_path).stem
    ext = Path(source_path).suffix
    return f"{prefix}_{source_name}_{timestamp}{ext}"

def is_image(file_path):
    """Cek apakah file adalah gambar"""
    return Path(file_path).suffix.lower() in SUPPORTED_IMAGE_FORMATS

def is_video(file_path):
    """Cek apakah file adalah video"""
    return Path(file_path).suffix.lower() in SUPPORTED_VIDEO_FORMATS

def get_file_size(file_path):
    """Dapatkan ukuran file dalam format readable"""
    size_bytes = Path(file_path).stat().st_size
    
    for unit in ['B', 'KB', 'MB', 'GB']:
        if size_bytes < 1024.0:
            return f"{size_bytes:.2f} {unit}"
        size_bytes /= 1024.0
    
    return f"{size_bytes:.2f} TB"

def validate_confidence(conf_value):
    """Validasi nilai confidence threshold"""
    try:
        conf = float(conf_value)
        return max(0.01, min(1.0, conf))
    except:
        return 0.25

def get_enhancement_info(enhancement_type):
    """Dapatkan informasi tentang enhancement"""
    return ENHANCEMENT_CONFIGS.get(enhancement_type, None)

def format_duration(seconds):
    """Format durasi dalam format readable"""
    if seconds < 60:
        return f"{seconds:.2f} detik"
    elif seconds < 3600:
        minutes = seconds // 60
        secs = seconds % 60
        return f"{int(minutes)} menit {int(secs)} detik"
    else:
        hours = seconds // 3600
        minutes = (seconds % 3600) // 60
        return f"{int(hours)} jam {int(minutes)} menit"

def ensure_directories():
    """Pastikan semua direktori yang diperlukan ada"""
    directories = [MODEL_DIR, OUTPUT_DIR]
    
    for directory in directories:
        directory.mkdir(parents=True, exist_ok=True)

# =====================
# LOG UTILITIES
# =====================

class Logger:
    """Simple logger untuk tracking process"""
    
    def __init__(self, log_file=None):
        self.log_file = log_file
        self.logs = []
    
    def log(self, message, level='INFO'):
        """Add log message"""
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        log_entry = f"[{timestamp}] [{level}] {message}"
        self.logs.append(log_entry)
        
        print(log_entry)
        
        if self.log_file:
            try:
                with open(self.log_file, 'a') as f:
                    f.write(log_entry + '\n')
            except:
                pass
    
    def info(self, message):
        """Log info message"""
        self.log(message, 'INFO')
    
    def warning(self, message):
        """Log warning message"""
        self.log(message, 'WARNING')
    
    def error(self, message):
        """Log error message"""
        self.log(message, 'ERROR')
    
    def success(self, message):
        """Log success message"""
        self.log(message, 'SUCCESS')
    
    def get_logs(self):
        """Get all logs"""
        return '\n'.join(self.logs)
    
    def clear(self):
        """Clear logs"""
        self.logs = []

# =====================
# INITIALIZATION
# =====================

# Pastikan direktori ada saat module di-import
ensure_directories()

# Export semua yang penting
__all__ = [
    'MODEL_DIR',
    'OUTPUT_DIR',
    'DEFAULT_SETTINGS',
    'ENHANCEMENT_CONFIGS',
    'SUPPORTED_IMAGE_FORMATS',
    'SUPPORTED_VIDEO_FORMATS',
    'MODEL_INFO',
    'load_config',
    'save_config',
    'get_available_models',
    'create_output_filename',
    'is_image',
    'is_video',
    'get_file_size',
    'validate_confidence',
    'get_enhancement_info',
    'format_duration',
    'Logger'
]