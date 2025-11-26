import sys
from pathlib import Path

def check_dependencies():
    """Cek apakah semua dependency tersedia"""
    required_modules = ['PyQt5', 'cv2', 'torch', 'numpy']
    missing_modules = []
    
    for module in required_modules:
        try:
            __import__(module)
        except ImportError:
            missing_modules.append(module)
    
    if missing_modules:
        print("=" * 60)
        print("ERROR: Module yang diperlukan tidak ditemukan!")
        print("=" * 60)
        print("\nModule yang hilang:")
        for module in missing_modules:
            print(f"  - {module}")
        print("\nSilakan install dengan perintah:")
        print("  pip install -r requirements.txt")
        print("=" * 60)
        return False
    
    return True

def check_files():
    """Cek apakah file-file yang diperlukan ada"""
    required_files = ['GUI.py', 'detect.py']
    missing_files = []
    
    for file in required_files:
        if not Path(file).exists():
            missing_files.append(file)
    
    if missing_files:
        print("=" * 60)
        print("ERROR: File yang diperlukan tidak ditemukan!")
        print("=" * 60)
        print("\nFile yang hilang:")
        for file in missing_files:
            print(f"  - {file}")
        print("=" * 60)
        return False
    
    # Cek folder model
    model_dir = Path("model")
    if not model_dir.exists():
        print("=" * 60)
        print("WARNING: Folder 'model' tidak ditemukan!")
        print("=" * 60)
        print("\nMembuat folder 'model'...")
        model_dir.mkdir(parents=True, exist_ok=True)
        print("✓ Folder 'model' telah dibuat.")
        print("\nSilakan letakkan file model (.pt) ke dalam folder 'model'")
        print("=" * 60)
    
    return True

def main():
    """Fungsi utama"""
    print("\n" + "=" * 60)
    print("OBJECT DETECTION APPLICATION")
    print("=" * 60)
    print()
    
    # Cek dependencies
    print("Memeriksa dependencies...")
    if not check_dependencies():
        sys.exit(1)
    print("✓ Semua dependencies tersedia\n")
    
    # Cek files
    print("Memeriksa file...")
    if not check_files():
        sys.exit(1)
    print("✓ Semua file tersedia\n")
    
    # Import GUI setelah semua pengecekan selesai
    try:
        from GUI import ObjectDetectionApp
        from PyQt5.QtWidgets import QApplication
    except ImportError as e:
        print(f"Error importing modules: {e}")
        sys.exit(1)
    
    # Jalankan aplikasi
    print("Meluncurkan aplikasi...")
    print("=" * 60)
    print()
    
    app = QApplication(sys.argv)
    app.setStyle('Fusion')  # Gunakan style Fusion untuk tampilan yang konsisten
    
    window = ObjectDetectionApp()
    window.show()
    
    sys.exit(app.exec_())

if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n\nAplikasi dihentikan oleh user")
        sys.exit(0)
    except Exception as e:
        print(f"\nError tidak terduga: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)